"""
FiLM global conditioning — training script.

Architecture (frozen ViT patch-pool → global γ/β applied uniformly to all decoder layers):

    sketch → CLIP ViT-B/32 patch tokens [num_tokens, 768]
           → mean-pool                   [768]
           → film_mlp_gamma / film_mlp_beta
           → γ [B, 256],  β [B, 256]
    for every decoder layer:
        query = γ * query + β
        query = layer(query, image_memory)
    → query_clip_proj → [B, 300, 512]  (cosine scored at eval)

Trained parameters: film_mlp_gamma, film_mlp_beta, query_clip_proj  (~656 K)
Frozen:             everything else in def-DETR

Optimization:
  - Sketch CLIP pools pre-cached per QuickDraw category at epoch start (768-d, ~80 cats, tiny)
  - Backbone + encoder params are frozen → no grad tracked through them automatically
  - AMP autocast (--amp) for faster forward/backward

Loss:
  Hungarian-matched queries vs CLIP text embed (default) or CLIP sketch embed (--sketch_targets).

Usage:
    python train_film.py \\
        --resume checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \\
        --coco_path $COCO_HOME \\
        --sketch_dataset qd \\
        --sketch_root $SKETCH_HOME \\
        --clip_checkpoint checkpoints/clip_model/ViT-B-32.pt \\
        --output_dir outputs/film_global_open_qd \\
        --epochs 10 --lr 1e-3 --batch_size 4
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import clip
import wandb

import util.misc as utils
from datasets.coco import CocoDetectionQD, CocoDetectionSketchy, make_coco_transforms, rasterize_stroke3
from models import build_model
from models.matcher import build_matcher
from train_clip_proj import (
    _build_sketch_file_map, _build_nearest_sketch_map, load_clip_model,
)


# ── ViT patch pool extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_vit_pool(clip_model, sketch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run sketch through CLIP ViT-B/32 and return mean-pooled patch features
    before the projection head.

    sketch_tensor : [B, 3, 224, 224]
    returns       : [B, 768]  (ViT hidden dim, float32)
    """
    vit = clip_model.visual
    dtype = vit.conv1.weight.dtype
    x = sketch_tensor.to(dtype)

    # patchify
    x = vit.conv1(x).flatten(2).permute(0, 2, 1)                     # [B, N, 768]
    cls = vit.class_embedding.to(dtype)[None, None].expand(len(x), -1, -1)
    x = torch.cat([cls, x], dim=1)                                    # [B, 1+N, 768]
    x = x + vit.positional_embedding.to(dtype)
    x = vit.ln_pre(x).permute(1, 0, 2)                                # LND

    x = vit.transformer(x).permute(1, 0, 2)                           # [B, 1+N, 768]
    x = vit.ln_post(x)                                                 # LN applied to all tokens
    return x.float().mean(dim=1)                                       # [B, 768]


# ── Per-category sketch pool cache ─────────────────────────────────────────────

@torch.no_grad()
def build_film_pool_cache(
    sketch_root: str,
    clip_model,
    preprocess,
    device,
    k: int = 5,
    sim_threshold: float = 0.85,
) -> dict:
    """
    Build {cat_name: pool_tensor [768]} for every available QuickDraw category,
    using nearest-category fallback for any categories without sketch files.

    Returns a dict ready for lookup by (normalised) COCO category name.
    """
    cat_file_map = _build_sketch_file_map(sketch_root)
    if not cat_file_map:
        print("WARNING: no QuickDraw sketch files found in sketch_root")
        return {}

    pool_cache: dict = {}
    n_cats = len(cat_file_map)
    print(f"Building FiLM sketch pool cache ({n_cats} categories, k={k} sketches each) …")

    for i, (cat, (ptr_path, strokes_path)) in enumerate(cat_file_map.items(), 1):
        ptr     = np.load(ptr_path)
        strokes = np.load(strokes_path, mmap_mode='r')
        n_sk = len(ptr) - 1
        idx  = np.random.choice(n_sk, min(k, n_sk), replace=False)
        imgs = [rasterize_stroke3(strokes[ptr[j]:ptr[j + 1]]) for j in idx]
        tensors = torch.stack([preprocess(img) for img in imgs]).to(device)
        pool_cache[cat] = extract_vit_pool(clip_model, tensors).mean(0).cpu()  # [768]

        if i % 20 == 0 or i == n_cats:
            print(f"  [{i}/{n_cats}] cached")

    print(f"Pool cache ready: {len(pool_cache)} categories")
    return pool_cache


def resolve_pool(cat_name: str, pool_cache: dict) -> torch.Tensor | None:
    """Return 768-d pool for cat_name, or None if not found."""
    key = cat_name.lower().replace(' ', '_')
    return pool_cache.get(key, pool_cache.get(key.replace('_', ' '), None))


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser('FiLM global conditioning training')

    # Required
    p.add_argument('--resume', required=True)
    p.add_argument('--coco_path', required=True)

    # CLIP (required for sketch pool extraction)
    p.add_argument('--clip_checkpoint', required=True,
                   help='Path to CLIP ViT-B/32 checkpoint')
    p.add_argument('--film_clip_dim', default=768, type=int,
                   help='ViT hidden dim (768 for ViT-B/32)')

    # Dataset
    p.add_argument('--sketch_dataset', default='qd', choices=['qd', 'sk'])
    p.add_argument('--sketch_root', default=None)
    p.add_argument('--train_scheme_world', default='open', choices=['open', 'closed'])
    p.add_argument('--data_frac', default=1.0, type=float)
    p.add_argument('--debug_size', default=0, type=int)
    p.add_argument('--multi_sketch', default=1, type=int)
    p.add_argument('--seed', default=42, type=int)

    # Sketch pool cache
    p.add_argument('--sketch_embed_k', default=5, type=int,
                   help='Sketches averaged per category for pool cache')
    p.add_argument('--sketch_map_threshold', default=0.85, type=float,
                   help='Min CLIP text-sim to accept a nearest-category proxy')

    # Loss
    p.add_argument('--sketch_targets', action='store_true', default=False,
                   help='Use CLIP sketch embed (512-d) as loss target instead of text embed')
    p.add_argument('--clip_dim', default=512, type=int)

    # Training
    p.add_argument('--epochs', default=10, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--amp', action='store_true', default=False,
                   help='Mixed-precision training (torch.cuda.amp)')
    p.add_argument('--output_dir', default='outputs/film_global_open_qd')
    p.add_argument('--device', default='cuda')
    p.add_argument('--print_freq', default=50, type=int)

    # Wandb
    p.add_argument('--wandb', action='store_true', default=False)
    p.add_argument('--wandb_user', default='')
    p.add_argument('--wandb_project', default='clip-ddetr-base')
    p.add_argument('--wandb_name', default='film_global')

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
    p.add_argument('--frozen_weights', default=None)
    p.add_argument('--lr_backbone', default=2e-5, type=float)
    p.add_argument('--cls_loss_coef', default=2.0, type=float)
    p.add_argument('--bbox_loss_coef', default=5.0, type=float)
    p.add_argument('--giou_loss_coef', default=2.0, type=float)
    p.add_argument('--mask_loss_coef', default=1.0, type=float)
    p.add_argument('--dice_loss_coef', default=1.0, type=float)
    p.add_argument('--focal_alpha', default=0.25, type=float)
    p.add_argument('--set_cost_class', default=2.0, type=float)
    p.add_argument('--set_cost_bbox', default=5.0, type=float)
    p.add_argument('--set_cost_giou', default=2.0, type=float)

    return p


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args_parser().parse_args()
    args.film_conditioning = True   # always on for this script
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   entity=args.wandb_user or None, config=vars(args))

    # ── Model ─────────────────────────────────────────────────────────────────
    model, _, _ = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Loaded checkpoint. Missing: {missing}")
    print(f"Unexpected: {unexpected}")

    # Freeze everything except film MLPs + query_clip_proj
    train_keys = ('film_mlp_gamma', 'film_mlp_beta', 'query_clip_proj')
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in train_keys)
    proj_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in proj_params)
    print(f"Training {n_params:,} parameters: {train_keys}")

    model.train()
    optimizer = torch.optim.Adam(proj_params, lr=args.lr)
    scaler = GradScaler(enabled=False)  # MSDeformAttn doesn't support float16
    matcher = build_matcher(args)

    # ── CLIP ──────────────────────────────────────────────────────────────────
    print(f"Loading CLIP from {args.clip_checkpoint} …")
    clip_model, preprocess = load_clip_model(args.clip_checkpoint, device)

    # ── Pre-cache sketch pools per category (768-d, one-time) ─────────────────
    assert args.sketch_root, "--sketch_root is required for FiLM training"
    pool_cache = build_film_pool_cache(
        args.sketch_root, clip_model, preprocess, device,
        k=args.sketch_embed_k, sim_threshold=args.sketch_map_threshold,
    )

    # sketch_targets: also cache 512-d CLIP sketch embeds for the loss
    sketch_embed_cache: dict = {}
    if args.sketch_targets:
        from train_clip_proj import _sketch_embed_for_category, _build_sketch_file_map as bsfm
        cat_file_map = bsfm(args.sketch_root)
        print("Pre-caching 512-d sketch embeds for loss targets …")
        for cat, (ptr_p, str_p) in cat_file_map.items():
            sketch_embed_cache[cat] = _sketch_embed_for_category(
                cat, ptr_p, str_p, clip_model, preprocess, device, args.sketch_embed_k,
            ).cpu()  # [512]
        print(f"  {len(sketch_embed_cache)} categories cached")

    # ── Dataset ───────────────────────────────────────────────────────────────
    root = Path(args.coco_path)
    DatasetClass = CocoDetectionQD if args.sketch_dataset == 'qd' else CocoDetectionSketchy
    dataset = DatasetClass(
        'train',
        str(root / 'train2017'),
        str(root / 'annotations' / 'instances_train2017.json'),
        transforms=make_coco_transforms('train'),
        return_masks=False,
        unroll_all_cats=False,
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
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    # COCO id → category name for pool lookup
    from pycocotools.coco import COCO
    coco_api = COCO(str(root / 'annotations' / 'instances_train2017.json'))
    id_to_name = {c['id']: c['name'] for c in coco_api.loadCats(coco_api.getCatIds())}

    n_total = len(loader)
    best_loss = float('inf')
    global_step = 0

    print(f"\nDataset: {len(dataset)} samples | {n_total} batches/epoch")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        total_loss = 0.0
        running_loss = 0.0
        n_batches = 0

        print(f"\nEpoch [{epoch + 1}/{args.epochs}]  —  {n_total} batches")

        for batch in loader:
            samples, targets, sketch_list_batch, cat_embeds, cat_ids = batch

            samples = samples.to(device)
            targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # ── Build sketch_pool [B, 768] from cache ─────────────────────────
            pools = []
            for i in range(len(targets)):
                cat_id = cat_ids[i].item() if isinstance(cat_ids[i], torch.Tensor) else int(cat_ids[i])
                cat_name = id_to_name.get(cat_id, '').lower().replace(' ', '_')
                pool = resolve_pool(cat_name, pool_cache)
                if pool is None:
                    # category not in QuickDraw — zero pool (FiLM becomes identity-ish)
                    pool = torch.zeros(args.film_clip_dim)
                pools.append(pool)
            sketch_pool = torch.stack(pools).to(device)    # [B, 768]

            # ── Forward ───────────────────────────────────────────────────────
            # AMP is intentionally disabled: MSDeformAttn custom CUDA kernel
            # does not support float16. --amp has no effect on the model forward.
            outputs = model(samples, sketch_pool=sketch_pool)

            # ── Hungarian matching (no grad needed) ───────────────────────────
            with torch.no_grad():
                indices = matcher(
                    {'pred_logits': outputs['pred_logits'].detach(),
                     'pred_boxes':  outputs['pred_boxes'].detach()},
                    targets_dev,
                )

            # ── Cosine alignment loss on matched queries ───────────────────────
            hs_last = outputs['hs_last']    # [B, 300, 256]
            loss = torch.tensor(0.0, device=device)
            n_matched = 0

            for i, (src_idx, _) in enumerate(indices):
                if len(src_idx) == 0:
                    continue

                hs_matched = hs_last[i][src_idx]                          # [n_gt, 256]
                proj = F.normalize(model.query_clip_proj(hs_matched), dim=-1)  # [n_gt, 512]

                if args.sketch_targets:
                    cat_id = cat_ids[i].item() if isinstance(cat_ids[i], torch.Tensor) else int(cat_ids[i])
                    cat_name = id_to_name.get(cat_id, '').lower().replace(' ', '_')
                    tgt_raw = sketch_embed_cache.get(cat_name)
                    if tgt_raw is None:
                        continue
                    tgt = F.normalize(tgt_raw.float().to(device).flatten(), dim=0).unsqueeze(0)
                else:
                    cat_embed = cat_embeds[i]
                    if not isinstance(cat_embed, torch.Tensor):
                        continue
                    tgt = F.normalize(cat_embed.float().to(device).flatten(), dim=0).unsqueeze(0)

                loss = loss + (1.0 - (proj * tgt).sum(-1)).mean()
                n_matched += 1

            if n_matched == 0:
                continue

            loss = loss / n_matched
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss  += loss.item()
            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            if n_batches % args.print_freq == 0:
                avg = running_loss / args.print_freq
                print(f"  step [{n_batches:5d}/{n_total}] | loss: {avg:.4f}")
                if args.wandb:
                    wandb.log({'train/loss_step': avg}, step=global_step)
                running_loss = 0.0

        epoch_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1} done | avg loss: {epoch_loss:.4f}")
        if args.wandb:
            wandb.log({'train/loss_epoch': epoch_loss, 'epoch': epoch + 1}, step=global_step)

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = {'model': model.state_dict(), 'epoch': epoch, 'loss': epoch_loss, 'args': args}
        torch.save(ckpt, os.path.join(args.output_dir, f'checkpoint{epoch:04d}.pth'))
        torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint.pth'))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            print(f"  ✦ New best loss: {best_loss:.4f}")

    print("\nTraining complete.")

    # ── Auto-eval on best checkpoint ──────────────────────────────────────────
    best_ckpt = os.path.join(args.output_dir, 'checkpoint_best.pth')
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(args.output_dir, 'checkpoint.pth')

    eval_out = os.path.join(args.output_dir.replace('film_global', 'eval_film'), '')
    os.makedirs(eval_out, exist_ok=True)

    import subprocess, sys
    cmd = [
        sys.executable, '-u', 'eval_sketch_baseline.py',
        '--resume',             best_ckpt,
        '--coco_path',          args.coco_path,
        '--sketch_dataset',     args.sketch_dataset,
        '--sketch_root',        args.sketch_root or '',
        '--train_scheme_world', args.train_scheme_world,
        '--clip_checkpoint',    args.clip_checkpoint,
        '--topk', '1', '5', '10',
        '--iou_thresh', '0.5',
        '--with_box_refine',
        '--film_conditioning',
        '--film_clip_dim', str(args.film_clip_dim),
        '--output_dir', eval_out,
    ]
    subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
