"""
Alignment training for the query_clip_proj layer only.

Freezes the entire def-DETR model and trains only the 256->512 linear projection
that maps decoder query features to CLIP's embedding space.

Training signal (v1 default):
  Cosine similarity between projected matched-query features and the
  pre-computed CLIP text embedding for the GT category.

v2 flags (can be combined):
  --sketch_targets    Use CLIP sketch embeddings as training targets instead
                      of text embeddings (closes train/eval domain gap).
                      Builds features_cache_v2.pt on first run by augmenting
                      the existing features_cache.pt with CLIP sketch embeds.
  --contrastive_loss  Replace per-sample cosine loss with in-batch InfoNCE:
                      matched queries should rank highest for their own target.

Speed mode (--cache_features, recommended):
  On the first run the frozen def-DETR forward pass is executed once per sample
  and hs_matched [n_gt, 256] is cached to disk. All subsequent epochs skip the
  backbone/encoder/decoder entirely (~10-20x faster per epoch).

Usage:
    # v1 (text targets, cosine loss)
    python train_clip_proj.py --resume ckpt.pth --coco_path /coco \
        --sketch_dataset qd --sketch_root /npy --epochs 10 --cache_features

    # v2 sketch targets only
    python train_clip_proj.py ... --sketch_targets \
        --clip_checkpoint checkpoints/clip_model/ViT-B-32.pt

    # v2 contrastive loss only (reuses v1 cache)
    python train_clip_proj.py ... --contrastive_loss

    # v2 both (highest expected gain)
    python train_clip_proj.py ... --sketch_targets --contrastive_loss \
        --clip_checkpoint checkpoints/clip_model/ViT-B-32.pt
"""

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import clip
import wandb

import util.misc as utils
from datasets.coco import CocoDetectionQD, CocoDetectionSketchy, make_coco_transforms, rasterize_stroke3
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


# ── v2 helpers ────────────────────────────────────────────────────────────────

def load_clip_model(checkpoint_path, device):
    """Return (clip_model, preprocess) for ViT-B/32."""
    model, preprocess = clip.load(checkpoint_path, device=device)
    model.eval()
    return model, preprocess


def _build_sketch_file_map(sketch_root):
    """
    Scan sketch_root for QuickDraw ptr+strokes pairs and return
    {category_name: (ptr_path, strokes_path)}.
    Prefers 'train' split, falls back to 'valid' then 'test'.
    """
    import glob
    # Collect all ptr files; each implies a paired strokes file
    cat_map = {}
    for split in ('train', 'valid', 'test'):
        for ptr_path in glob.glob(os.path.join(sketch_root, f'*.{split}.ptr.npy')):
            stem = os.path.basename(ptr_path)[:-4]          # e.g. 'cat.train.ptr'
            cat = stem.rsplit('.', 2)[0]                     # 'cat'
            strokes_path = ptr_path.replace('.ptr.npy', '.strokes.npy')
            if cat not in cat_map and os.path.exists(strokes_path):
                cat_map[cat] = (ptr_path, strokes_path)
    return cat_map


@torch.no_grad()
def _sketch_embed_for_category(cat_name, ptr_path, strokes_path, clip_model, preprocess,
                                device, k):
    """Sample k QuickDraw sketches using ptr+strokes format → mean normalised CLIP embed [512]."""
    ptr     = np.load(ptr_path)
    strokes = np.load(strokes_path, mmap_mode='r')
    n_sketches = len(ptr) - 1
    idx = np.random.choice(n_sketches, min(k, n_sketches), replace=False)
    imgs = [rasterize_stroke3(strokes[ptr[i]:ptr[i + 1]]) for i in idx]
    tensors = torch.stack([preprocess(img) for img in imgs]).to(device)
    feats = clip_model.encode_image(tensors)
    return F.normalize(feats.float().mean(0), dim=-1).cpu()


def _build_nearest_sketch_map(missing_cats, available_cats, text_embeds, clip_model, device,
                              sim_threshold=0.85):
    """
    For each category in missing_cats (no .npy file), find the nearest available
    QuickDraw category by cosine similarity in CLIP text space.
    If the best similarity is below sim_threshold, maps to None (caller falls back
    to the CLIP text embedding for that category).
    Returns {missing_cat: nearest_available_cat | None}.
    """
    tokens = clip.tokenize(available_cats).to(device)
    with torch.no_grad():
        avail_vecs = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)  # [N_avail, 512]

    nearest_map = {}
    print(f"  Resolving {len(missing_cats)} missing categories "
          f"(sim_threshold={sim_threshold}):")
    for cat in sorted(missing_cats):
        cat_vec = F.normalize(
            torch.as_tensor(text_embeds[cat], dtype=torch.float32).flatten().to(device), dim=0
        )
        sims = avail_vecs @ cat_vec                     # [N_avail]
        best_sim, best_idx = sims.max(0)
        best_sim = best_sim.item()

        if best_sim >= sim_threshold:
            nearest = available_cats[best_idx.item()]
            nearest_map[cat] = nearest
            print(f"    '{cat}' → '{nearest}'  (sim={best_sim:.3f})")
        else:
            nearest_map[cat] = None                     # fallback: use text embed
            print(f"    '{cat}' → [text embed]  (best sim={best_sim:.3f} < {sim_threshold})")
    return nearest_map


@torch.no_grad()
def augment_cache_with_sketches(records, clip_model, preprocess,
                                sketch_root, text_embeds_pkl,
                                device, out_path, k=5, sim_threshold=0.85):
    """
    Build features_cache_v2.pt by adding a 'sketch_embed' [512] field to every
    existing record.  Category is recovered via cosine reverse-lookup against
    text_embeddings.pkl so the expensive hs_matched extraction is not repeated.
    """
    with open(text_embeds_pkl, 'rb') as f:
        text_embeds = pickle.load(f)

    cat_names = list(text_embeds.keys())
    # Flatten each embed to [D] before stacking — guards against [1, D] entries in the pkl
    cat_vecs = F.normalize(
        torch.stack([torch.as_tensor(text_embeds[c], dtype=torch.float32).flatten()
                     for c in cat_names]),
        dim=-1,
    )  # [N_cats, D]  guaranteed 2D

    # Build canonical category → file path map (handles split/format suffixes)
    cat_file_map = _build_sketch_file_map(sketch_root)
    available_set = set(cat_file_map.keys())

    missing_cats = {c for c in cat_names if c not in available_set}
    nearest_map: dict = {}   # missing_cat → nearest available cat name | None
    if missing_cats:
        nearest_map = _build_nearest_sketch_map(
            missing_cats, sorted(available_set), text_embeds, clip_model, device,
            sim_threshold=sim_threshold)

    per_cat: dict = {}   # cat_name → sketch_embed tensor [512]
    augmented = []
    n = len(records)
    print(f"Augmenting {n} records | {len(available_set)} cats available, "
          f"{len(missing_cats)} truly missing …")

    for i, rec in enumerate(records):
        cat_flat = F.normalize(rec['cat_embed'].float().flatten(), dim=0)  # [D]
        cat_name = cat_names[(cat_vecs @ cat_flat).argmax().item()]
        effective_cat = nearest_map.get(cat_name, cat_name)  # None = below threshold

        if effective_cat is None:
            # Truly missing and no close proxy — use normalised text embed
            sketch_embed = F.normalize(rec['cat_embed'].float().flatten(), dim=0).cpu()
        else:
            if effective_cat not in per_cat:
                ptr_path, strokes_path = cat_file_map[effective_cat]
                per_cat[effective_cat] = _sketch_embed_for_category(
                    effective_cat, ptr_path, strokes_path,
                    clip_model, preprocess, device, k,
                )
            sketch_embed = per_cat[effective_cat]

        augmented.append({
            'hs_matched':   rec['hs_matched'],
            'cat_embed':    rec['cat_embed'],
            'sketch_embed': sketch_embed,
        })

        if (i + 1) % 5000 == 0 or (i + 1) == n:
            print(f"  [{i + 1}/{n}]  {len(per_cat)} categories resolved")

    torch.save(augmented, out_path)
    print(f"Saved v2 cache ({len(augmented)} records, {len(per_cat)} cats) → {out_path}")
    return augmented


def infonce_loss_fn(projs_list, targets_stacked, temperature):
    """
    In-batch InfoNCE: each matched query should rank highest for its own target.

    projs_list:      list of N tensors, each [n_i, 512] L2-normalised
    targets_stacked: [N, 512] L2-normalised (one target per record)

    Returns un-normalised sum loss (caller divides by N for logging parity).
    """
    loss = torch.tensor(0.0, device=targets_stacked.device)
    for i, proj in enumerate(projs_list):
        logits = (proj @ targets_stacked.mT) / temperature   # [n_i, N]
        labels = torch.full((len(proj),), i,
                            dtype=torch.long, device=targets_stacked.device)
        loss = loss + F.cross_entropy(logits, labels)
    return loss


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
    p.add_argument('--base_cache_dir', default='',
                   help='Directory that already contains features_cache.pt from a v1 run. '
                        'Used by v2 variants to reuse the expensive detector extraction.')

    # v2: sketch targets
    p.add_argument('--sketch_targets', action='store_true', default=False,
                   help='Use CLIP sketch embeddings as training targets instead of text '
                        'embeddings. Requires --clip_checkpoint. Builds '
                        'features_cache_v2.pt on first run (reuses hs_matched from v1 cache).')
    p.add_argument('--clip_checkpoint', default='',
                   help='Path to CLIP ViT-B/32 .pt file. Required for --sketch_targets.')
    p.add_argument('--text_embeds', default='',
                   help='Path to text_embeddings.pkl for category reverse-lookup when '
                        'building v2 cache. Defaults to <clip_checkpoint_dir>/text_embeddings.pkl.')
    p.add_argument('--sketch_embed_k', default=5, type=int,
                   help='Number of sketches to average per category in v2 cache.')
    p.add_argument('--sketch_map_threshold', default=0.85, type=float,
                   help='Min cosine similarity (CLIP text space) to accept a nearest-neighbour '
                        'QuickDraw substitute for a missing category. '
                        'Below this threshold the CLIP text embedding is used instead.')

    # v2: contrastive loss
    p.add_argument('--contrastive_loss', action='store_true', default=False,
                   help='Replace per-sample cosine loss with in-batch InfoNCE. '
                        'Matched queries must rank above all other queries in the batch.')
    p.add_argument('--temperature', default=0.07, type=float,
                   help='Softmax temperature for InfoNCE loss.')

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
        # ── v1 cache (hs_matched + cat_embed) ─────────────────────────────────
        base_dir = args.base_cache_dir or args.output_dir
        v1_cache = os.path.join(base_dir, 'features_cache.pt')
        v2_cache = os.path.join(args.output_dir, 'features_cache_v2.pt')

        # Determine which cache file we actually need
        need_v2_cache = args.sketch_targets
        active_cache = v2_cache if need_v2_cache else v1_cache

        if os.path.exists(active_cache):
            print(f"Loading cache from {active_cache}")
            records = torch.load(active_cache, weights_only=False)
        else:
            # Ensure v1 cache exists (extract if missing)
            if os.path.exists(v1_cache):
                print(f"Loading v1 cache from {v1_cache}")
                records = torch.load(v1_cache, weights_only=False)
            else:
                extract_ds = build_dataset(extract_transforms)
                extract_loader = DataLoader(
                    extract_ds,
                    batch_size=args.batch_size * 2,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=utils.collate_fn,
                    pin_memory=True,
                )
                records = extract_features(model, matcher, extract_loader, device, v1_cache)

            if need_v2_cache:
                # Augment v1 records with CLIP sketch embeddings
                assert args.clip_checkpoint, "--clip_checkpoint is required for --sketch_targets"
                assert args.sketch_root,     "--sketch_root is required for --sketch_targets"
                text_embeds_pkl = args.text_embeds or os.path.join(
                    os.path.dirname(args.clip_checkpoint), 'text_embeddings.pkl')

                print(f"Loading CLIP for sketch augmentation from {args.clip_checkpoint} …")
                clip_model_aug, preprocess_aug = load_clip_model(args.clip_checkpoint, device)
                records = augment_cache_with_sketches(
                    records, clip_model_aug, preprocess_aug,
                    args.sketch_root, text_embeds_pkl,
                    device, v2_cache, k=args.sketch_embed_k,
                    sim_threshold=args.sketch_map_threshold,
                )
                del clip_model_aug, preprocess_aug
                torch.cuda.empty_cache()

        train_loader = DataLoader(
            CachedFeatureDataset(records),
            batch_size=args.batch_size * 4,  # much larger: only a linear layer runs
            shuffle=True,
            num_workers=0,          # records are already in memory
            collate_fn=cache_collate,
        )
        use_cache = True
        variant_tag = []
        if args.sketch_targets:   variant_tag.append('sketch-targets')
        if args.contrastive_loss: variant_tag.append('InfoNCE')
        vtag = '+'.join(variant_tag) or 'text-targets+cosine'
        print(f"Cache mode [{vtag}]: {len(records)} records | batch {args.batch_size * 4}")

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
                # ── select target key ──────────────────────────────────────────
                tgt_key = 'sketch_embed' if args.sketch_targets else 'cat_embed'

                if args.contrastive_loss:
                    # In-batch InfoNCE: gather all projs + targets then compute once
                    projs_list = []
                    targets_list = []
                    for rec in batch:
                        proj = F.normalize(
                            model.query_clip_proj(rec['hs_matched'].to(device)), dim=-1)
                        # flatten to [512] regardless of whether stored as [512] or [1,512]
                        tgt = F.normalize(rec[tgt_key].float().to(device).flatten(), dim=0)
                        projs_list.append(proj)
                        targets_list.append(tgt)

                    n_matched = len(projs_list)
                    if n_matched < 2:
                        continue  # need ≥2 records for meaningful contrastive negatives

                    targets_stacked = torch.stack(targets_list)          # [N, 512]
                    loss = infonce_loss_fn(projs_list, targets_stacked, args.temperature)

                else:
                    # Per-sample cosine loss (v1 or sketch-targets only)
                    loss = torch.tensor(0.0, device=device)
                    n_matched = 0
                    for rec in batch:
                        hs_matched = rec['hs_matched'].to(device)        # [n_gt, 256]
                        proj = F.normalize(model.query_clip_proj(hs_matched), dim=-1)
                        tgt = F.normalize(
                            rec[tgt_key].float().to(device).flatten(), dim=0).unsqueeze(0)
                        loss = loss + (1.0 - (proj * tgt).sum(-1)).mean()
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
