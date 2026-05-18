#!/bin/bash
#SBATCH --job-name=train_proj
#SBATCH --output=outputs/train_proj_%j.log
#SBATCH --error=outputs/train_proj_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=ada
#SBATCH --gres=gpu:ADA6000:1
# =============================================================
# Train the query_clip_proj alignment layer (all other weights frozen).
#
# World-split variants:
#   open   (default) — train on seen categories only; held-out categories
#                      are used for eval. Tests generalisation to unseen
#                      sketch queries. This is the rigorous baseline setting.
#   closed           — train on ALL categories including eval ones.
#                      Useful as an upper-bound reference.
#
# Usage:
#   bash  scripts/train_proj.sh [RESUME] [SKETCH_DS] [WORLD]  # local GPU
#   sbatch scripts/train_proj.sh                               # Slurm
# =============================================================

echo "job: $SLURM_JOB_NAME"
# >>> Conda setup <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr

# Job execution commands
. ./.env
echo $COCO_HOME
echo $SLURM_JOBID

# 1) Find a free port by binding to port 0
export MASTER_PORT=$(python - <<'EOF'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
port = s.getsockname()[1]
s.close()
print(port)
EOF
)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_NNODES=${SLURM_NNODES:-1}
export SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-1}
echo "nnodes: $SLURM_NNODES"
echo "nproc_per_node: $SLURM_GPUS_ON_NODE"
echo "master port: $MASTER_PORT"

# ── Args ───────────────────────────────────────────────────────────────────────
RESUME="${1:-$PROJECT_HOME/checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth}"
SKETCH_DS="${2:-qd}"    # qd | sk
WORLD="${3:-open}"      # open | closed
OUTPUT_DIR="$PROJECT_HOME/outputs/clip_proj_aligned_${WORLD}_${SKETCH_DS}"

echo "========================================"
echo " Projection alignment training"
echo "  world split : $WORLD"
echo "  sketch data : $SKETCH_DS"
echo "  resume      : $RESUME"
echo "  output      : $OUTPUT_DIR"
echo "========================================"

python -u train_clip_proj.py \
    --resume             "$RESUME" \
    --coco_path          "$COCO_HOME" \
    --sketch_dataset     "$SKETCH_DS" \
    --sketch_root        "$SKETCH_HOME" \
    --train_scheme_world "$WORLD" \
    --epochs             10 \
    --lr                 1e-3 \
    --batch_size         4 \
    --num_workers        8 \
    --with_box_refine \
    --cache_features \
    --output_dir         "$OUTPUT_DIR" \
    --wandb \
    --wandb_user         "aurkohaldi" \
    --wandb_name         "base_qd_ow" \
    --wandb_project      "clip-ddetr-base" \
    "${@:4}"

# ── Eval after training ────────────────────────────────────────────────────────
echo "========================================"
echo " Running eval on trained checkpoint"
echo "========================================"

EVAL_CKPT="$OUTPUT_DIR/checkpoint_best.pth"
if [ ! -f "$EVAL_CKPT" ]; then
    echo "WARNING: checkpoint_best.pth not found, falling back to checkpoint.pth"
    EVAL_CKPT="$OUTPUT_DIR/checkpoint.pth"
fi

python -u eval_sketch_baseline.py \
    --resume             "$EVAL_CKPT" \
    --coco_path          "$COCO_HOME" \
    --sketch_dataset     "$SKETCH_DS" \
    --sketch_root        "$SKETCH_HOME" \
    --train_scheme_world "$WORLD" \
    --clip_checkpoint    "$PROJECT_HOME/checkpoints/clip_model/ViT-B-32.pt" \
    --topk 1 5 10 \
    --iou_thresh         0.5 \
    --with_box_refine \
    --output_dir         "$PROJECT_HOME/outputs/eval_baseline_${WORLD}_${SKETCH_DS}"
