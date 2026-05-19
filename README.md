# Sketch-Query Detection Baseline

This section describes the minimal sketch-guided detection baseline built on top of Deformable DETR.

**Idea.** After the decoder produces 300 object-query features (`hs[-1]`, 256-d), a linear projection layer maps them into CLIP's 512-d embedding space. At inference, each proposal is scored by its cosine similarity with the CLIP visual embedding of a sketch query. No category labels are used at test time.

```
Image ──▶ Def-DETR backbone + decoder ──▶ query features [B, 300, 256]
                                                    │ query_clip_proj (256→512)
                                                    ▼
                                          query embeddings [B, 300, 512]
                                                    │ cosine similarity
Sketch ──▶ CLIP ViT-B/32 ───────────────▶ sketch embedding [B, 512]
                                                    │
                                          ranked proposals → mAP / Recall@k
```

## Environment setup

All scripts source `.env` from the project root for the following variables:

```bash
# .env  (already present — edit paths if needed)
export COCO_HOME="/mnt/1tb/data/coco"
export SKETCH_HOME="/mnt/1tb/data/quickdraw/sketchrnn"   # QuickDraw .npy files
export PROJECT_HOME="/home/rahul/arka/clip_ddetr_base"
```

Activate the conda environment before running anything:

```bash
conda activate clip_ddetr
```

## Checkpoints layout

```
checkpoints/
├── clip_model/
│   ├── ViT-B-32.pt               # CLIP visual encoder
│   └── text_embeddings.pkl       # pre-computed CLIP text embeddings
├── r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
└── r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth
```

## World-split variants

| Mode | Description | When to use |
|---|---|---|
| `open` **(default)** | Train on **seen** categories only; eval categories are held out | Rigorous baseline — tests generalisation to unseen sketch queries |
| `closed` | Train on **all** categories, including eval ones | Upper-bound reference; numbers will be inflated for held-out categories |

Both scripts accept `WORLD` as the third positional argument.

## Step 1 — Train the projection layer

Freezes all def-DETR weights and trains only `query_clip_proj` (256 → 512, ~131 K params).

Each script carries its own `#SBATCH` headers so it runs identically whether invoked locally or submitted to Slurm:

```bash
bash  scripts/train_proj.sh [RESUME] [SKETCH_DS] [WORLD] [VARIANT]   # local GPU
sbatch scripts/train_proj.sh                                           # Slurm
```

| Positional arg | Default | Options |
|---|---|---|
| `RESUME` | `checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth` | any def-DETR `.pth` |
| `SKETCH_DS` | `qd` | `qd` (QuickDraw), `sk` (Sketchy) |
| `WORLD` | `open` | `open`, `closed` |
| `VARIANT` | `v1` | `v1`, `sketch`, `contrast`, `both` — see table below |

### Training variants

| `VARIANT` | Training target | Loss | W&B run name | Notes |
|---|---|---|---|---|
| `v1` | CLIP text embedding | cosine | `base_qd_ow_v1` | Baseline. Reuses `features_cache.pt`. |
| `sketch` | CLIP **sketch** embedding | cosine | `v2_sketch_qd_ow` | Closes train/eval domain gap. Builds `features_cache_v2.pt` once by augmenting the v1 cache with CLIP visual encodings of sketches (cheap — detector not re-run). |
| `contrast` | CLIP text embedding | **InfoNCE** | `v2_contrast_qd_ow` | In-batch contrastive: matched queries rank above all others. Reuses v1 cache directly. |
| `both` | CLIP **sketch** embedding | **InfoNCE** | `v2_both_qd_ow` | Combines both improvements. Highest expected gain. |

W&B run names substitute `qd` with `sk` when `SKETCH_DS=sk`.

**Examples**

```bash
# v1 baseline (open-world, QuickDraw)
bash scripts/train_proj.sh

# v2 — sketch targets only
bash scripts/train_proj.sh "" qd open sketch

# v2 — InfoNCE contrastive loss only (reuses existing features_cache.pt)
bash scripts/train_proj.sh "" qd open contrast

# v2 — both (recommended for best performance)
bash scripts/train_proj.sh "" qd open both

# Slurm — submit any variant
sbatch scripts/train_proj.sh "" qd open both

# Closed-world upper-bound
bash scripts/train_proj.sh "" qd closed v1

# Extra Python flags (append after VARIANT)
bash scripts/train_proj.sh "" qd open both --epochs 20 --lr 5e-4
```

### Cache reuse

On the first run, the frozen def-DETR pass extracts matched decoder query features once and saves them to `features_cache.pt` (~10–20× speedup for all subsequent epochs). v2 variants build on this:

```
features_cache.pt    ← extracted once (full def-DETR forward, slow)
features_cache_v2.pt ← augmented from v1 (CLIP visual pass only, fast)
                        built automatically when VARIANT=sketch or both
```

Both variants point `--base_cache_dir` at the v1 output directory so they reuse the existing `features_cache.pt` without re-running the detector.

Checkpoints are saved after every epoch:

```
outputs/clip_proj_aligned_open_qd/          # v1
outputs/clip_proj_aligned_open_qd_sketch/   # sketch variant
outputs/clip_proj_aligned_open_qd_contrast/ # contrast variant
outputs/clip_proj_aligned_open_qd_both/     # both
├── checkpoint.pth          # latest epoch
├── checkpoint_best.pth     # best epoch by val loss
├── checkpoint0000.pth
└── ...
```

Key Python flags (pass after `VARIANT`):

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch_size` | `4` | Batch size |
| `--temperature` | `0.07` | InfoNCE softmax temperature (`contrast` / `both` only) |
| `--sketch_embed_k` | `5` | Sketches averaged per category in v2 cache |
| `--debug_size` | `0` | Truncate dataset to N samples (0 = full) |

---

## Step 2 — Evaluate the baseline

Ranks each image's 300 proposals by cosine similarity with the sketch CLIP embedding and reports:

- **Recall@1 / @5 / @10** at IoU ≥ 0.5
- **COCO-style mAP**, AP@50, APS, APM, APL (class-agnostic, cosine-sim as confidence score)

```bash
bash  scripts/eval_baseline.sh [RESUME] [SKETCH_DS] [WORLD]   # local GPU
sbatch scripts/eval_baseline.sh                                # Slurm
```

| Positional arg | Default | Options |
|---|---|---|
| `RESUME` | `outputs/clip_proj_aligned_{WORLD}_{SKETCH_DS}/checkpoint.pth` | any trained `.pth` |
| `SKETCH_DS` | `qd` | `qd` (QuickDraw), `sk` (Sketchy) |
| `WORLD` | `open` | `open`, `closed` |

**Examples**

```bash
# Open-world QuickDraw — local GPU (matches default training run)
bash scripts/eval_baseline.sh

# Same, submitted to Slurm
sbatch scripts/eval_baseline.sh

# Evaluate the closed-world checkpoint
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_closed_qd/checkpoint.pth qd closed

# Specific epoch checkpoint
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_open_qd/checkpoint0004.pth qd open

# Sketchy, open-world
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_open_sk/checkpoint.pth sk open

# Change IoU threshold or top-k values
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_open_qd/checkpoint.pth qd open \
    --iou_thresh 0.25 --topk 1 3 5 10
```

**Expected output format**

```
============================================================
Sketch-Query Detection Baseline  |  N samples
============================================================
  Recall@ 1 (IoU≥0.5): 0.XXXX  (X/N)
  Recall@ 5 (IoU≥0.5): 0.XXXX  (X/N)
  Recall@10 (IoU≥0.5): 0.XXXX  (X/N)

============================================================
COCO-style class-agnostic mAP (cosine-sim as confidence score):
============================================================
 Average Precision  (AP) @[ IoU=0.50:0.95 | ... ]
 ...
  mAP    : 0.XXXX
  AP@50  : 0.XXXX
  AP@75  : 0.XXXX
  APS    : 0.XXXX
  APM    : 0.XXXX
  APL    : 0.XXXX
```

Key flags (pass after positional args to override defaults):

| Flag | Default | Description |
|---|---|---|
| `--topk` | `1 5 10` | Space-separated list of k values for Recall@k |
| `--iou_thresh` | `0.5` | IoU threshold for Recall@k |
| `--batch_size` | `1` | Inference batch size |
| `--clip_checkpoint` | `checkpoints/clip_model/ViT-B-32.pt` | CLIP model path |
| `--output_dir` | `outputs/eval_baseline` | Output directory |
| `--random_proj` | off | Reinitialise `query_clip_proj` with random weights after loading the checkpoint — use as a chance-level sanity check |

**Random projection baseline**

Measures what recall/mAP looks like with a randomly initialised projection (scores are meaningless cosine similarities). Any trained checkpoint can be used since only the def-DETR backbone weights are kept.

```bash
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_open_qd/checkpoint.pth qd open --random_proj
```

---

## Quick end-to-end run

```bash
# ── v1 baseline (open-world, QuickDraw) ──────────────────────────────────────
bash scripts/train_proj.sh              # trains + auto-evals
bash scripts/eval_baseline.sh          # standalone eval

# Slurm (no changes needed — #SBATCH headers built into scripts)
sbatch scripts/train_proj.sh
sbatch scripts/eval_baseline.sh

# ── v2 — full (sketch targets + InfoNCE) ─────────────────────────────────────
# Requires v1 to have run first so features_cache.pt exists
bash scripts/train_proj.sh "" qd open both

# ── Upper-bound comparison (closed-world) ─────────────────────────────────────
bash scripts/train_proj.sh "" qd closed v1
bash scripts/eval_baseline.sh outputs/clip_proj_aligned_closed_qd/checkpoint.pth qd closed
```

### Baseline results (open-world, QuickDraw, 2111 val samples)

| Variant | Recall@1 | Recall@5 | Recall@10 | mAP | AP@50 | AR@100 |
|---|---|---|---|---|---|---|
| v1 (text targets, cosine) | 8.5% | 23.2% | 32.3% | 1.35% | 2.28% | 47.8% |
| sketch | — | — | — | — | — | — |
| contrast | — | — | — | — | — | — |
| both | — | — | — | — | — | — |

*AR@100 = 47.8% means the detector proposals cover GT boxes well; the gap to AR@10 shows ranking quality is the bottleneck.*

### Citing Deformable DETR
If you find Deformable DETR useful in your research, please consider citing:
```bibtex
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```