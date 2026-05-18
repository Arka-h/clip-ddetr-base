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

Freezes all def-DETR weights and trains only `query_clip_proj` (256 → 512, ~131 K params) using pre-computed CLIP text embeddings as alignment targets.

Each script carries its own `#SBATCH` headers so it runs identically whether invoked locally or submitted to Slurm:

```bash
bash  scripts/train_proj.sh [RESUME] [SKETCH_DS] [WORLD]   # local GPU
sbatch scripts/train_proj.sh                                # Slurm (uses #SBATCH defaults)
```

| Positional arg | Default | Options |
|---|---|---|
| `RESUME` | `checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth` | any def-DETR `.pth` |
| `SKETCH_DS` | `qd` | `qd` (QuickDraw), `sk` (Sketchy) |
| `WORLD` | `open` | `open`, `closed` |

**Examples**

```bash
# Open-world QuickDraw — local GPU
bash scripts/train_proj.sh

# Same, submitted to Slurm
sbatch scripts/train_proj.sh

# Closed-world upper-bound
bash scripts/train_proj.sh checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth qd closed

# Sketchy, open-world
bash scripts/train_proj.sh checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth sk open

# Pass extra Python flags after the three positional args
bash scripts/train_proj.sh checkpoints/....pth qd open --epochs 20 --lr 5e-4
```

Checkpoints are saved to `outputs/clip_proj_aligned_{WORLD}_{SKETCH_DS}/` after every epoch:

```
outputs/clip_proj_aligned_open_qd/
├── checkpoint.pth        # latest epoch (overwritten each epoch)
├── checkpoint0000.pth
├── checkpoint0001.pth
└── ...
```

Key Python flags (pass after the three positional args):

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch_size` | `4` | Batch size |
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

---

## Quick end-to-end run

```bash
# Rigorous baseline (open-world, QuickDraw) — local GPU
bash scripts/train_proj.sh     # → outputs/clip_proj_aligned_open_qd/
bash scripts/eval_baseline.sh

# Same, on Slurm (no changes needed — #SBATCH headers are built into the scripts)
sbatch scripts/train_proj.sh
sbatch scripts/eval_baseline.sh

# Upper-bound comparison (closed-world, same dataset)
bash  scripts/train_proj.sh checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth qd closed
bash  scripts/eval_baseline.sh outputs/clip_proj_aligned_closed_qd/checkpoint.pth qd closed
```

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