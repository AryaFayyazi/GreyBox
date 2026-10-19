#!/bin/bash
# G5 confirmatory data: one fresh pass per model (answers + replies, hidden), then reply embeddings
set -uo pipefail
for M in qwen3_8b llama31_8b mistral7b gemma3_4b phi4mini olmo3_7b qwen25_7b gptoss20b; do
  D=/data/models/graybox_ws/outputs/g5/confirm/$M
  [ -f $D/replies_none.parquet ] || python scripts/133_g5_answers_vs_replies.py stage=gen model=$M split=confirm || echo "GEN FAILED $M"
  [ -f $D/hidden_none.npz ] || python scripts/133_g5_answers_vs_replies.py stage=hidden model=$M split=confirm || echo "HIDDEN FAILED $M"
done
python scripts/133_g5_answers_vs_replies.py stage=embed split=confirm
