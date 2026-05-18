"""
Alignment training for the query_clip_proj layer only.

Freezes the entire def-DETR model and trains only the 256->512 linear projection
that maps decoder query features to CLIP's embedding space.

Training signal: cosine similarity between projected matched-query features and
the pre-computed CLIP text embedding for the GT category.

Speed mode (--cache_features, recommended):
  On the first run the frozen def-DETR forward pass is executed once per sample
  and hs_last [300, 256] is cached to disk. All subsequent epochs skip the
  backbone/encoder/decoder entirely (~10-20x faster per epoch). AMP is used
  for the extraction pass.

Usage:
    python train_clip_proj.py \
        --coco_path /home/rahul/coco \
        --resume checkpoints/r50_deformable_detr.pth \
        --sketch_dataset qd \
        --sketch_root /path/to/quickdraw_npy \
        --epochs 10 --lr 1e-3 \
        --cache_features \
        --output_dir outputs/clip_proj_aligned
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import wandb

import util.misc as utils
from datasets.coco import CocoDetectionQD, CocoDetectionSketchy, make_coco_transforms
from models import build_model
from models.matcher import build_matcher


# ── Feature cache dataset ──────────────────────────────────────────────────────

class CachedFeatureDataset(Dataset):
    """
    Wraps the pre-extracted (hs_matched, cat_embed) pairs.
    Bypasses the backbone/encoder/decoder entirely during training epochs.
    """
    def __init__(self, records):
        # records: list of {'hs_matched': [n_gt, 256], 'cat_embed': [512]}
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def cache_collate(batch):
    return batch  # list of dicts — variable-length src_idx, no padding needed


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, matcher, loader, device, cache_path):
    """
    One pass over the dataset. Stores only the Hungarian-matched query rows to keep
    memory small: {'hs_matched': [n_gt, 256], 'cat_embed': [512]} per sample.
    """
    print(f"Extracting features ({len(loader)} batches) → {cache_path}")
    model.eval()
    records = []

    n_total = len(loader)
    for batch_idx, batch in enumerate(loader):
        samples, targets, _sketches, cat_embeds, _cat_ids = batch
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        indices = matcher(outputs, targets)

        hs_last = outputs['hs_last']  # [B, 300, 256] — on GPU, slice before moving to CPU

        for i, (src_idx, _) in enumerate(indices):
            cat_embed = cat_embeds[i]
            if len(src_idx) == 0 or not isinstance(cat_embed, torch.Tensor):
                continue
            records.append({
                'hs_matched': hs_last[i][src_idx].cpu(),  # [n_gt, 256] — only matched rows
                'cat_embed': cat_embed.cpu(),              # [512]
            })

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_total:
            print(f"  [{batch_idx + 1}/{n_total}] extracted {len(records)} records so far")

    torch.save(records, cache_path)
    print(f"  Cached {len(records)} samples → {cache_path}")
    return records


# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser('Sketch-CLIP projection alignment training')

    # Required
    p.add_argument('--resume', required=True, help='def-DETR checkpoint to start from')
    p.add_argument('--coco_path', required=True)

    # Dataset
    p.add_argument('--sketch_dataset', default='qd', choices=['qd', 'sk'],
                   help='qd=QuickDraw, sk=Sketchy')
    p.add_argument('--sketch_root', default=None,
                   help='Root dir for QuickDraw .npy files (required for qd)')
    p.add_argument('--train_scheme_world', default='open', choices=['open', 'closed'])
    p.add_argument('--data_frac', default=1.0, type=float)
    p.add_argument('--debug_size', default=0, type=int,
                   help='If >0, truncate dataset to this many samples (debugging)')
    p.add_argument('--multi_sketch', default=1, type=int)
    p.add_argument('--seed', default=42, type=int)

    # Speed
    p.add_argument('--cache_features', action='store_true', default=False,
                   help='Pre-extract hs_last once and cache to disk. '
                        'Skips the backbone/encoder/decoder on all subsequent epochs '
                        '(~10-20x faster). Uses fixed val-style transforms for extraction.')

    # Training
    p.add_argument('--epochs', default=10, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--output_dir', default='outputs/clip_proj_aligned')
    p.add_argument('--device', default='cuda')
    p.add_argument('--clip_dim', default=512, type=int)
    p.add_argument('--print_freq', default=50, type=int,
                   help='Print loss every N batches')

    # Wandb
    p.add_argument('--wandb', action='store_true', default=False)
    p.add_argument('--wandb_user', default='', help='wandb entity / username')
    p.add_argument('--wandb_project', default='clip-ddetr-base')
    p.add_argument('--wandb_name', default='train_proj')

    # Def-DETR model args (must match checkpoint)
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--dilation', action='store_true')
    p.add_argument('--position_embedding', default='sine',
                   choices=('sine', 'learned'))
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args_parser().parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Wandb ──────────────────────────────────────────────────────────────────
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_user or None,
            config=vars(args),
        )

    # ── Build model ────────────────────────────────────────────────────────────
    model, _, _ = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Loaded checkpoint. Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    # Freeze everything except query_clip_proj
    for name, param in model.named_parameters():
        param.requires_grad = 'query_clip_proj' in name
    proj_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in proj_params)
    print(f"Training {n_params:,} parameters (query_clip_proj only)")

    model.eval()

    optimizer = torch.optim.Adam(proj_params, lr=args.lr)
    matcher = build_matcher(args)

    # ── Dataset ────────────────────────────────────────────────────────────────
    root = Path(args.coco_path)
    img_folder = root / 'train2017'
    ann_file = root / 'annotations' / 'instances_train2017.json'
    DatasetClass = CocoDetectionQD if args.sketch_dataset == 'qd' else CocoDetectionSketchy

    # Use val-style (fixed) transforms for feature extraction to keep cache deterministic
    extract_transforms = make_coco_transforms('val')
    train_transforms = make_coco_transforms('train')

    def build_dataset(transforms):
        return DatasetClass(
            'train',
            str(img_folder),
            str(ann_file),
            transforms=transforms,
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

    # ── Feature caching ────────────────────────────────────────────────────────
    if args.cache_features:
        cache_path = os.path.join(args.output_dir, 'features_cache.pt')

        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            records = torch.load(cache_path, weights_only=False)
        else:
            # Extract once with fixed transforms + AMP
            extract_ds = build_dataset(extract_transforms)
            extract_loader = DataLoader(
                extract_ds,
                batch_size=args.batch_size * 2,  # larger batch fine, no grad storage
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=utils.collate_fn,
                pin_memory=True,
            )
            records = extract_features(model, matcher, extract_loader, device, cache_path)

        train_loader = DataLoader(
            CachedFeatureDataset(records),
            batch_size=args.batch_size * 4,  # much larger: only a linear layer runs
            shuffle=True,
            num_workers=0,          # records are already in memory
            collate_fn=cache_collate,
        )
        use_cache = True
        print(f"Cache mode: {len(records)} records | batch size: {args.batch_size * 4}")

    else:
        # Standard mode: run full model every step
        dataset = build_dataset(train_transforms)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
            drop_last=True,
            pin_memory=True,
        )
        use_cache = False
        print(f"Standard mode: {len(dataset)} samples | batches/epoch: {len(train_loader)}")

    # ── Training loop ──────────────────────────────────────────────────────────
    n_total = len(train_loader)
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        total_loss = 0.0
        running_loss = 0.0
        n_batches = 0

        print(f"\nEpoch [{epoch + 1}/{args.epochs}]  —  {n_total} batches")

        for batch in train_loader:

            if use_cache:
                # batch is a list of record dicts — skip the model forward entirely
                loss = torch.tensor(0.0, device=device)
                n_matched = 0
                for rec in batch:
                    hs_matched = rec['hs_matched'].to(device)          # [n_gt, 256]
                    proj = F.normalize(model.query_clip_proj(hs_matched), dim=-1)
                    text_tgt = F.normalize(rec['cat_embed'].float().to(device).unsqueeze(0), dim=-1)
                    loss = loss + (1.0 - (proj * text_tgt).sum(-1)).mean()
                    n_matched += 1

            else:
                samples, targets, _sketches, cat_embeds, _cat_ids = batch
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                cat_embeds = [
                    e.to(device, non_blocking=True) if isinstance(e, torch.Tensor) else None
                    for e in cat_embeds
                ]

                with torch.no_grad():
                    outputs = model(samples)
                    indices = matcher(outputs, targets)

                hs_last = outputs['hs_last']

                loss = torch.tensor(0.0, device=device)
                n_matched = 0
                for i, (src_idx, _) in enumerate(indices):
                    if len(src_idx) == 0 or cat_embeds[i] is None:
                        continue
                    hs_matched = hs_last[i][src_idx]
                    proj = F.normalize(model.query_clip_proj(hs_matched), dim=-1)
                    text_tgt = F.normalize(cat_embeds[i].float().unsqueeze(0), dim=-1)
                    loss = loss + (1.0 - (proj * text_tgt).sum(-1)).mean()
                    n_matched += 1

            if n_matched == 0:
                continue

            loss = loss / n_matched
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()
            n_batches += 1

            if n_batches % args.print_freq == 0:
                avg_running = running_loss / args.print_freq
                print(f"  step [{n_batches:5d}/{n_total}] | loss: {avg_running:.4f}")
                if args.wandb:
                    wandb.log({'train/loss_step': avg_running}, step=global_step)
                running_loss = 0.0

            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch [{epoch + 1}/{args.epochs}] done | avg loss: {avg_loss:.4f}")
        if args.wandb:
            wandb.log({'train/loss_epoch': avg_loss, 'epoch': epoch + 1}, step=global_step)

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint.pth'))
        torch.save(ckpt, os.path.join(args.output_dir, f'checkpoint{epoch:04d}.pth'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            print(f"  → best checkpoint (loss: {best_loss:.4f}) saved to {args.output_dir}/checkpoint_best.pth")
        else:
            print(f"  → saved to {args.output_dir}/checkpoint.pth")

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
