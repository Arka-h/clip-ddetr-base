#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --output=outputs/eval_baseline_%j.log
#SBATCH --error=outputs/eval_baseline_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=ada
#SBATCH --gres=gpu:ADA6000:1
# =============================================================
# Evaluate the sketch-query detection baseline.
# Ranks def-DETR proposals by cosine similarity with a CLIP sketch embedding.
# Reports Recall@1/5/10 and full COCO mAP / AP@50 / APS / APM / APL.
#
# Usage:
#   bash  scripts/eval_baseline.sh [RESUME] [SKETCH_DS] [WORLD]  # local GPU
#   sbatch scripts/eval_baseline.sh                               # Slurm
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
SKETCH_DS="${2:-qd}"    # qd | sk
WORLD="${3:-open}"      # open | closed
RESUME="${1:-$PROJECT_HOME/outputs/clip_proj_aligned_${WORLD}_${SKETCH_DS}/checkpoint.pth}"
OUTPUT_DIR="$PROJECT_HOME/outputs/eval_baseline_${WORLD}_${SKETCH_DS}"

echo "========================================"
echo " Sketch-query baseline evaluation"
echo "  world split : $WORLD"
echo "  sketch data : $SKETCH_DS"
echo "  resume      : $RESUME"
echo "  output      : $OUTPUT_DIR"
echo "========================================"

python -u eval_sketch_baseline.py \
    --resume             "$RESUME" \
    --coco_path          "$COCO_HOME" \
    --sketch_dataset     "$SKETCH_DS" \
    --sketch_root        "$SKETCH_HOME" \
    --train_scheme_world "$WORLD" \
    --clip_checkpoint    "$PROJECT_HOME/checkpoints/clip_model/ViT-B-32.pt" \
    --topk 1 5 10 \
    --iou_thresh         0.5 \
    --with_box_refine \
    --output_dir         "$OUTPUT_DIR" \
    "${@:4}"
