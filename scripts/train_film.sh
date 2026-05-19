#!/bin/bash
#SBATCH --job-name=train_film
#SBATCH --output=outputs/train_film_%j.log
#SBATCH --error=outputs/train_film_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=ada
#SBATCH --gres=gpu:ADA6000:1
# =============================================================
# Train the FiLM global conditioning baseline.
#
# Trains: film_mlp_gamma, film_mlp_beta, query_clip_proj (~656 K params)
# Freezes: all def-DETR backbone / encoder / decoder weights
#
# Positional args:
#   1  RESUME     def-DETR checkpoint  (default: iterative-bbox-refine ckpt)
#   2  SKETCH_DS  qd | sk              (default: qd)
#   3  WORLD      open | closed        (default: open)
#
# Usage:
#   bash  scripts/train_film.sh [RESUME] [SKETCH_DS] [WORLD]
#   sbatch scripts/train_film.sh
#
# Extra Python flags can be appended after WORLD:
#   bash scripts/train_film.sh "" qd open --epochs 20 --amp
# =============================================================

echo "job: $SLURM_JOB_NAME"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clip_ddetr

. ./.env
echo $COCO_HOME
echo $SLURM_JOBID

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
echo "nnodes: $SLURM_NNODES  nproc_per_node: $SLURM_GPUS_ON_NODE  port: $MASTER_PORT"

# ── Args ───────────────────────────────────────────────────────────────────────
RESUME="${1:-$PROJECT_HOME/checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth}"
SKETCH_DS="${2:-qd}"
WORLD="${3:-open}"

CLIP_CKPT="$PROJECT_HOME/checkpoints/clip_model/ViT-B-32.pt"
OUTPUT_DIR="$PROJECT_HOME/outputs/film_global_${WORLD}_${SKETCH_DS}"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo " FiLM global conditioning training"
echo "  world split : $WORLD"
echo "  sketch data : $SKETCH_DS"
echo "  resume      : $RESUME"
echo "  output      : $OUTPUT_DIR"
echo "========================================"

python -u train_film.py \
    --resume             "$RESUME" \
    --coco_path          "$COCO_HOME" \
    --sketch_dataset     "$SKETCH_DS" \
    --sketch_root        "$SKETCH_HOME" \
    --train_scheme_world "$WORLD" \
    --clip_checkpoint    "$CLIP_CKPT" \
    --epochs             10 \
    --lr                 1e-3 \
    --batch_size         4 \
    --num_workers        8 \
    --amp \
    --with_box_refine \
    --output_dir         "$OUTPUT_DIR" \
    --wandb \
    --wandb_user         "aurkohaldi" \
    --wandb_name         "v2_film_${SKETCH_DS}_ow" \
    --wandb_project      "clip-ddetr-base" \
    "${@:4}"
