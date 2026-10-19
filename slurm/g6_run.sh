#!/bin/bash
# G6 data: one pass per model on the 1,000 fresh contexts (answers + replies, hidden), then embeddings, then the analysis
set -uo pipefail
for M in qwen3_8b llama31_8b mistral7b gemma3_4b phi4mini olmo3_7b qwen25_7b gptoss20b; do
  D=/data/models/graybox_ws/outputs/g6/run/$M
  [ -f $D/replies_none.parquet ] || python scripts/137_g6_shared_geometry.py stage=gen model=$M || echo "GEN FAILED $M"
  [ -f $D/hidden_none.npz ] || python scripts/137_g6_shared_geometry.py stage=hidden model=$M || echo "HIDDEN FAILED $M"
done
python scripts/137_g6_shared_geometry.py stage=embed || echo "EMBED FAILED"
