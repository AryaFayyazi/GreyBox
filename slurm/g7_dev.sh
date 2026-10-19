#!/bin/bash
# G7 dry run on 600 L2 development contexts: channels + hidden at the same position, all 8 models, then the analysis
set -uo pipefail
for M in qwen3_8b llama31_8b mistral7b gemma3_4b phi4mini olmo3_7b qwen25_7b gptoss20b; do
  [ -f /data/models/graybox_ws/outputs/g7/dev/$M/channels.npz ] || python scripts/138_g7_channels.py stage=gen model=$M set=dev || echo "GEN FAILED $M"
done
python scripts/138_g7_channels.py stage=analyze set=dev n_boot=200 || echo "ANALYZE FAILED"
