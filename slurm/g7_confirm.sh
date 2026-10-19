#!/bin/bash
# G7 confirmatory data on the 1,000 fresh G7 contexts: answers (vLLM) and channels + hidden (HF), all 8 models; analysis
set -uo pipefail
for M in qwen3_8b llama31_8b mistral7b gemma3_4b phi4mini olmo3_7b qwen25_7b gptoss20b; do
  [ -f /data/models/graybox_ws/outputs/g7/g7_answers/$M/psi_none.npz ] || python scripts/138_g7_channels.py stage=answers model=$M set=g7 || echo "ANSWERS FAILED $M"
  [ -f /data/models/graybox_ws/outputs/g7/g7/$M/channels.npz ] || python scripts/138_g7_channels.py stage=gen model=$M set=g7 || echo "GEN FAILED $M"
done
python scripts/138_g7_channels.py stage=analyze set=g7 || echo "ANALYZE FAILED"
