#!/bin/bash
# One L2 model: black-box interrogation (vLLM) then ground-truth hidden states (HF). Usage: sbatch slurm/run.sh bash slurm/l2_model.sh <model>
set -uo pipefail
python scripts/12_l2_run.py model=$1 stage=interrogate || echo "L2 interrogate $1 FAILED"
python scripts/12_l2_run.py model=$1 stage=hidden || echo "L2 hidden $1 FAILED"
python scripts/12_l2_run.py model=$1 stage=generate || echo "L2 generate $1 FAILED"
