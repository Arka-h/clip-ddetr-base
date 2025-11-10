#!/bin/bash
#SBATCH --job-name=cd_cw_N2_sk                    # Job name
#SBATCH --output=outputs/cd_cw_N2_sk_%j.log        # Standard output log (%j = job ID)
#SBATCH --error=outputs/cd_cw_N2_sk_%j.err         # Standard error log
#SBATCH --time=2-00:00:00                     # Time limit (dd-hh:mm:ss)
#SBATCH --ntasks=2                            # Number of tasks (typically 1 for single-node jobs)
#SBATCH --cpus-per-task=8                     # Number of CPUs per task
#SBATCH --mem=48GB                            # Memory allocation
#SBATCH --partition=ada                       # Partition (long/queue)
#SBATCH --gres=gpu:ADA6000:2                  # GPU allocation (if needed, modify accordingly)
# #SBATCH --nodelist=cn8                        # Node to run on (modify as needed)
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

python -u -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port $MASTER_PORT \
    main.py \
    --epochs 50 \
    --batch_size 8 \
    --num_workers 8 \
    --with_box_refine \
    --coco_path $COCO_HOME \
    --output_dir $PROJECT_HOME/outputs/open_set_ddetr_box_refine \
    --start_epoch 0 \
    --wandb_user "aurkohaldi" \
    --wandb_name "DDETR_ow_open" \
    --wandb_project "ddetr" \
    --wandb \
# --resume $PROJECT_HOME/checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth