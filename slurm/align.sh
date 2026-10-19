#!/bin/bash
set -uo pipefail
for M in qwen3_8b llama31_8b mistral7b gemma3_4b phi4mini olmo3_7b qwen25_7b gptoss20b; do
  [ -f /data/models/graybox_ws/outputs/g7/dev_align/$M.npz ] || python scripts/144_question_alignment_exploratory.py stage=gen model=$M || echo "GEN FAILED $M"
done
python scripts/144_question_alignment_exploratory.py stage=analyze || echo "ANALYZE FAILED"
