#!/bin/bash
set -uo pipefail
for a in gemma4 gptoss20b qwen38; do
  [ -f /data/models/graybox_ws/outputs/g43/$a.parquet ] && continue
  python scripts/219_g43_situation_battery.py stage=battery actor=$a || echo "G43 $a FAILED"
done
[ -f /data/models/graybox_ws/outputs/g43/museglimmer.parquet ] || python scripts/219_g43_situation_battery.py stage=battery actor=museglimmer vllm_python=/data/models/cape_ws/envs/v030/bin/python || echo "G43 museglimmer FAILED"
