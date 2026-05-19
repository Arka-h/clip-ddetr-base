"""
Sketch-query detection baseline — inference & evaluation.

For each (image, sketch-query) pair in the val set:
  1. Run def-DETR → query_clip_embeds [B, 300, 512]  &  pred_boxes [B, 300, 4]
  2. Encode sketch with CLIP ViT-B/32 → sketch_embed [B, 512]
  3. Score proposals by cosine similarity, rank descending
  4. Evaluate:
       • Recall@k (k=1,5,10) at IoU ≥ 0.5
       • COCO-style mAP / AP@50 / APS / APM / APL  (class-agnostic, category_id=1)

Usage:
    python eval_sketch_baseline.py \
        --coco_path /home/rahul/coco \
        --resume outputs/clip_proj_aligned/checkpoint.pth \
        --sketch_dataset qd \
        --sketch_root /path/to/quickdraw_npy \
        --clip_checkpoint checkpoints/clip_model/ViT-B-32.pt \
        --topk 1 5 10 --iou_thresh 0.5
"""

import argparse
import contextlib
import copy
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

import util.misc as utils
from datasets.coco import CocoDetectionQD, CocoDetectionSketchy, make_coco_transforms
from models import build_model
from util.box_ops import box_cxcywh_to_xyxy, box_iou


# ── Helpers ────────────────────────────────────────────────────────────────────

def box_cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, orig_size: torch.Tensor) -> torch.Tensor:
    """
    boxes: [N, 4]  normalized cxcywh (values in [0,1])
    orig_size: [2]  (H, W)
    returns: [N, 4]  absolute xyxy
    """
    H, W = orig_size[0].item(), orig_size[1].item()
    scale = boxes.new_tensor([W, H, W, H])
    return box_cxcywh_to_xyxy(boxes * scale)


def compute_recall_at_k(
    pred_boxes_xyxy: torch.Tensor,   # [N_pred, 4] sorted descending by score
    gt_boxes_xyxy: torch.Tensor,     # [N_gt, 4]
    ks: list,
    iou_thresh: float,
) -> dict:
    """Returns {k: 1 if any top-k pred hits any GT at IoU >= iou_thresh else 0}."""
    results = {}
    if len(gt_boxes_xyxy) == 0:
        return {k: 0 for k in ks}
    for k in ks:
        top_k = pred_boxes_xyxy[:k]                            # [min(k,N), 4]
        iou, _ = box_iou(top_k, gt_boxes_xyxy)                # [k, N_gt]
        results[k] = int((iou.max() >= iou_thresh).item())
    return results


def load_clip_visual_encoder(checkpoint_path: str, device):
    """Load CLIP ViT-B/32 and return visual encoder + dtype."""
    import clip
    model, _ = clip.load(checkpoint_path, device=device)
    model.eval()
    return model


# ── Argument parsing ────────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser('Sketch-query detection baseline evaluation')

    # Required
    p.add_argument('--resume', required=True, help='checkpoint (def-DETR + trained query_clip_proj)')
    p.add_argument('--coco_path', required=True)

    # Dataset
    p.add_argument('--sketch_dataset', default='qd', choices=['qd', 'sk'])
    p.add_argument('--sketch_root', default=None)
    p.add_argument('--train_scheme_world', default='open', choices=['open', 'closed'])
    p.add_argument('--data_frac', default=1.0, type=float)
    p.add_argument('--debug_size', default=0, type=int)
    p.add_argument('--multi_sketch', default=1, type=int)
    p.add_argument('--seed', default=42, type=int)

    # CLIP
    p.add_argument('--clip_checkpoint', default='checkpoints/clip_model/ViT-B-32.pt')
    p.add_argument('--random_proj', action='store_true',
                   help='Reinitialise query_clip_proj with random weights after loading '
                        'the checkpoint. Use as a chance-level sanity check.')

    # Evaluation
    p.add_argument('--topk', nargs='+', type=int, default=[1, 5, 10])
    p.add_argument('--iou_thresh', default=0.5, type=float)
    p.add_argument('--batch_size', default=1, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output_dir', default='outputs/eval_baseline')
    p.add_argument('--clip_dim', default=512, type=int)

    # Def-DETR model args (must match checkpoint)
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--dilation', action='store_true')
    p.add_argument('--position_embedding', default='sine', choices=('sine', 'learned'))
    p.add_argument('--position_embedding_scale', default=2 * math.pi, type=float)
    p.add_argument('--num_feature_levels', default=4, type=int)
    p.add_argument('--enc_layers', default=6, type=int)
    p.add_argument('--dec_layers', default=6, type=int)
    p.add_argument('--dim_feedforward', default=1024, type=int)
    p.add_argument('--hidden_dim', default=256, type=int)
    p.add_argument('--dropout', default=0.1, type=float)
    p.add_argument('--nheads', default=8, type=int)
    p.add_argument('--num_queries', default=300, type=int)
    p.add_argument('--dec_n_points', default=4, type=int)
    p.add_argument('--enc_n_points', default=4, type=int)
    p.add_argument('--with_box_refine', action='store_true')
    p.add_argument('--two_stage', action='store_true')
    p.add_argument('--aux_loss', action='store_true', default=True)
    p.add_argument('--masks', action='store_true')
    p.add_argument('--dataset_file', default='coco')
    p.add_argument('--frozen_weights', default=None, type=str)
    p.add_argument('--lr_backbone', default=2e-5, type=float)

    # Loss coefficients (required by build_model / SetCriterion)
    p.add_argument('--cls_loss_coef', default=2.0, type=float)
    p.add_argument('--bbox_loss_coef', default=5.0, type=float)
    p.add_argument('--giou_loss_coef', default=2.0, type=float)
    p.add_argument('--mask_loss_coef', default=1.0, type=float)
    p.add_argument('--dice_loss_coef', default=1.0, type=float)
    p.add_argument('--focal_alpha', default=0.25, type=float)

    # Matcher args
    p.add_argument('--set_cost_class', default=2.0, type=float)
    p.add_argument('--set_cost_bbox', default=5.0, type=float)
    p.add_argument('--set_cost_giou', default=2.0, type=float)

    return p


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = get_args_parser().parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model, _, _ = build_model(args)
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Loaded checkpoint. Missing: {missing}  Unexpected: {unexpected}")

    if args.random_proj:
        model.query_clip_proj.reset_parameters()
        print("query_clip_proj reinitialised with random weights (chance-level baseline)")

    # ── CLIP ─────────────────────────────────────────────────────────────────
    print(f"Loading CLIP from {args.clip_checkpoint} ...")
    clip_model = load_clip_visual_encoder(args.clip_checkpoint, device)

    # ── Dataset ──────────────────────────────────────────────────────────────
    root = Path(args.coco_path)
    img_folder = root / 'val2017'
    ann_file = root / 'annotations' / 'instances_val2017.json'
    DatasetClass = CocoDetectionQD if args.sketch_dataset == 'qd' else CocoDetectionSketchy
    dataset = DatasetClass(
        'val',
        str(img_folder),
        str(ann_file),
        transforms=make_coco_transforms('val'),
        return_masks=False,
        unroll_all_cats=True,
        data_frac=args.data_frac,
        train_scheme_world=args.train_scheme_world,
        ds_len=args.debug_size,
        multi_sketch=args.multi_sketch,
        seed=args.seed,
        inference=False,
        sketch_root=args.sketch_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    print(f"Val dataset: {len(dataset)} (image, category) pairs")

    # ── Evaluation accumulators ───────────────────────────────────────────────
    recall_hits = {k: 0 for k in args.topk}
    total_samples = 0

    # For COCO mAP we build a synthetic single-class GT and collect predictions
    coco_gt_images = []
    coco_gt_anns = []
    coco_preds = []
    ann_id_counter = 0
    # synthetic_id increments per sample (avoids duplicate image_ids for same COCO image)
    synthetic_id = 0

    # ── Inference loop ────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in loader:
            samples, targets, sketch_list_batch, _cat_embeds, _cat_ids = batch

            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # sketch_list_batch: tuple of [K, 3, 224, 224] tensors (one per image in batch)

            outputs = model(samples)
            query_clip_embeds = outputs['query_clip_embeds']   # [B, 300, 512]
            pred_boxes_norm = outputs['pred_boxes']             # [B, 300, 4] cxcywh norm

            B = len(targets)
            for i in range(B):
                orig_size = targets[i]['orig_size']            # [H, W]

                # ── Encode sketch with CLIP ──────────────────────────────
                sk = sketch_list_batch[i].to(device)           # [K, 3, 224, 224]
                sketch_feats = clip_model.encode_image(sk)     # [K, 512]
                sketch_embed = F.normalize(
                    sketch_feats.float().mean(0, keepdim=True), dim=-1
                )                                              # [1, 512] (avg over K sketches)

                # ── Cosine similarity & ranking ──────────────────────────
                q_embeds = query_clip_embeds[i]                # [300, 512]
                sim_scores = (q_embeds * sketch_embed).sum(-1) # [300]
                rank_idx = sim_scores.argsort(descending=True)
                ranked_scores = sim_scores[rank_idx]           # [300]
                ranked_boxes_norm = pred_boxes_norm[i][rank_idx]  # [300, 4]

                # ── Convert predicted boxes to absolute xyxy ─────────────
                ranked_boxes_abs = box_cxcywh_norm_to_xyxy_abs(
                    ranked_boxes_norm, orig_size
                )                                              # [300, 4]

                # ── GT boxes: val dataset reverts to pre-transform absolute xyxy ──
                gt_boxes_abs = targets[i]['boxes'].to(ranked_boxes_abs)  # [N_gt, 4]

                # ── Recall@k ─────────────────────────────────────────────
                hits = compute_recall_at_k(
                    ranked_boxes_abs, gt_boxes_abs,
                    args.topk, args.iou_thresh,
                )
                for k in args.topk:
                    recall_hits[k] += hits[k]
                total_samples += 1

                # ── Accumulate for COCO mAP ──────────────────────────────
                sid = synthetic_id
                synthetic_id += 1

                H, W = orig_size[0].item(), orig_size[1].item()
                coco_gt_images.append({'id': sid, 'height': H, 'width': W})

                for box_xyxy in gt_boxes_abs.cpu():
                    x1, y1, x2, y2 = box_xyxy.tolist()
                    w, h = x2 - x1, y2 - y1
                    coco_gt_anns.append({
                        'id': ann_id_counter,
                        'image_id': sid,
                        'category_id': 1,
                        'bbox': [x1, y1, w, h],
                        'area': w * h,
                        'iscrowd': 0,
                    })
                    ann_id_counter += 1

                for rank_j in range(len(ranked_boxes_abs)):
                    x1, y1, x2, y2 = ranked_boxes_abs[rank_j].cpu().tolist()
                    coco_preds.append({
                        'image_id': sid,
                        'category_id': 1,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': ranked_scores[rank_j].item(),
                    })

    # ── Report Recall@k ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Sketch-Query Detection Baseline  |  {total_samples} samples")
    print(f"{'='*60}")
    for k in args.topk:
        r = recall_hits[k] / max(total_samples, 1)
        print(f"  Recall@{k:2d} (IoU≥{args.iou_thresh}): {r:.4f}  ({recall_hits[k]}/{total_samples})")

    # ── COCO mAP ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COCO-style class-agnostic mAP (cosine-sim as confidence score):")
    print(f"{'='*60}")

    coco_gt_dict = {
        'images': coco_gt_images,
        'annotations': coco_gt_anns,
        'categories': [{'id': 1, 'name': 'object', 'supercategory': 'object'}],
    }

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            coco_gt_obj = COCO()
            coco_gt_obj.dataset = coco_gt_dict
            coco_gt_obj.createIndex()
            coco_dt_obj = coco_gt_obj.loadRes(coco_preds)

    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # [mAP, AP@50, AP@75, APS, APM, APL, ...]
    names = ['mAP', 'AP@50', 'AP@75', 'APS', 'APM', 'APL']
    print()
    for name, val in zip(names, stats[:6]):
        print(f"  {name:<8}: {val:.4f}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
