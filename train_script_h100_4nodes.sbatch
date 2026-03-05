#!/bin/bash
#SBATCH -J h100_origin
#SBATCH -p accelerated-h100          #
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1         # 
#SBATCH --cpus-per-task=64
#SBATCH --time=0-6:00:00
#SBATCH --mem=0
#SBATCH -o run_logs/%x-%j.out
#SBATCH -e run_logs/%x-%j.err

set -euo pipefail

# ========= 1) 环境 =========
source ~/conda/etc/profile.d/conda.sh
conda activate video   # 按你的环境名改

# ========= 2) 变量 =========
CONFIG=configs/longlive_train_long.yaml
LOGDIR=logs_h100_v2
WANDB_SAVE_DIR=wandb
mkdir -p "$LOGDIR" "$WANDB_SAVE_DIR"

# ========= 4) 启动 =========
# 推荐用 srun，slurm 会更好地绑核/绑卡

# MASTER_ADDR="localhost:29500"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"


srun torchrun \
  --nnodes=4 --node_rank="$SLURM_NODEID" --nproc_per_node=4 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint "$RDZV_ENDPOINT" \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \