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
# Positional args:
#   1  RESUME     checkpoint to start from (default: iterative-bbox-refine ckpt)
#   2  SKETCH_DS  qd | sk                 (default: qd)
#   3  WORLD      open | closed           (default: open)
#   4  VARIANT    v1 | sketch | contrast | both  (default: v1)
#              v1       — text targets + cosine loss  [baseline]
#              sketch   — sketch targets + cosine loss
#              contrast — text targets   + InfoNCE loss
#              both     — sketch targets + InfoNCE loss  [highest expected gain]
#
# Usage:
#   bash  scripts/train_proj.sh [RESUME] [SKETCH_DS] [WORLD] [VARIANT]
#   sbatch scripts/train_proj.sh
#
# Examples:
#   bash scripts/train_proj.sh "" qd open v1      # reproduce baseline
#   bash scripts/train_proj.sh "" qd open both    # v2 full
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
VARIANT="${4:-v1}"      # v1 | sketch | contrast | both

CLIP_CKPT="$PROJECT_HOME/checkpoints/clip_model/ViT-B-32.pt"
V1_OUTPUT_DIR="$PROJECT_HOME/outputs/clip_proj_aligned_${WORLD}_${SKETCH_DS}"

# ── Variant config ─────────────────────────────────────────────────────────────
case "$VARIANT" in
    v1)
        OUTPUT_DIR="$V1_OUTPUT_DIR"
        BASE_CACHE_FLAG=""
        V2_FLAGS=""
        WANDB_NAME="base_${SKETCH_DS}_ow_v1"
        ;;
    sketch)
        OUTPUT_DIR="${V1_OUTPUT_DIR}_sketch"
        BASE_CACHE_FLAG="--base_cache_dir $V1_OUTPUT_DIR"
        V2_FLAGS="--sketch_targets --clip_checkpoint $CLIP_CKPT"
        WANDB_NAME="v2_sketch_${SKETCH_DS}_ow"
        ;;
    contrast)
        OUTPUT_DIR="${V1_OUTPUT_DIR}_contrast"
        BASE_CACHE_FLAG="--base_cache_dir $V1_OUTPUT_DIR"
        V2_FLAGS="--contrastive_loss"
        WANDB_NAME="v2_contrast_${SKETCH_DS}_ow"
        ;;
    both)
        OUTPUT_DIR="${V1_OUTPUT_DIR}_both"
        BASE_CACHE_FLAG="--base_cache_dir $V1_OUTPUT_DIR"
        V2_FLAGS="--sketch_targets --contrastive_loss --clip_checkpoint $CLIP_CKPT"
        WANDB_NAME="v2_both_${SKETCH_DS}_ow"
        ;;
    *)
        echo "ERROR: Unknown VARIANT '$VARIANT'. Use v1 | sketch | contrast | both"
        exit 1
        ;;
esac

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo " Projection alignment training"
echo "  world split : $WORLD"
echo "  sketch data : $SKETCH_DS"
echo "  variant     : $VARIANT"
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
    $BASE_CACHE_FLAG \
    $V2_FLAGS \
    --output_dir         "$OUTPUT_DIR" \
    --wandb \
    --wandb_user         "aurkohaldi" \
    --wandb_name         "$WANDB_NAME" \
    --wandb_project      "clip-ddetr-base" \
    "${@:5}"

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
    --clip_checkpoint    "$CLIP_CKPT" \
    --topk 1 5 10 \
    --iou_thresh         0.5 \
    --with_box_refine \
    --output_dir         "$PROJECT_HOME/outputs/eval_baseline_${WORLD}_${SKETCH_DS}_${VARIANT}"
