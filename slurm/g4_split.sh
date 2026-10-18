#!/bin/bash
# G4 data for one split: answers + replies (vLLM), hidden states (HF), reply embeddings; models in sequence on one GPU
set -uo pipefail
SPLIT=$1
for M in qwen3_8b llama31_8b mistral7b gemma3_4b phi4mini olmo3_7b qwen25_7b gptoss20b; do
  [ -f /data/models/graybox_ws/outputs/g4/$SPLIT/$M/replies_pirate.parquet ] || python scripts/132_g4_shift.py stage=gen model=$M split=$SPLIT || echo "GEN FAILED $M"
  [ -f /data/models/graybox_ws/outputs/g4/$SPLIT/$M/hidden_pirate.npz ] || python scripts/132_g4_shift.py stage=hidden model=$M split=$SPLIT || echo "HIDDEN FAILED $M"
done
python scripts/132_g4_shift.py stage=embed split=$SPLIT
