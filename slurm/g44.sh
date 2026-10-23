#!/bin/bash
set -uo pipefail
for t in gpt gptoss20b qwen38 museglimmer; do
  [ -f /data/models/graybox_ws/outputs/g44/${t}_by_gemma4.parquet ] && continue
  python scripts/220_g44_self_vs_other_forecast.py stage=battery target=$t || echo "G44 $t FAILED"
done
