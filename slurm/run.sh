#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/data/models/graybox_ws/slurm_logs/%x-%j.out
# Usage: sbatch -J <name> slurm/run.sh python scripts/<script>.py key=value ...   (ONE GPU per job, run sequentially)
set -euo pipefail
cd /home/arya/graybox
source /data/models/cape_ws/venv/bin/activate
export PATH=$PATH:/data/models/cider_ws/envs/cider/bin
export LD_LIBRARY_PATH=/data/models/cider_ws/envs/cider/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_CACHE=/data/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT=/data/models/graybox_ws/cache/vllm TRITON_CACHE_DIR=/data/models/graybox_ws/cache/triton TORCHINDUCTOR_CACHE_DIR=/data/models/graybox_ws/cache/inductor XDG_CACHE_HOME=/data/models/graybox_ws/cache
echo "job $SLURM_JOB_ID on $(hostname) gpu=$CUDA_VISIBLE_DEVICES start $(date)"
"$@"
echo "end $(date)"
