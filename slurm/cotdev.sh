#!/bin/bash
set -uo pipefail
for A in gptoss20b qwen38 museglimmer; do
  [ -f /data/models/graybox_ws/outputs/cot_boundary/${A}_cot_dev.npz ] || python scripts/149_cot_boundary.py stage=features agent=$A split=dev || echo "FEAT FAILED $A"
done
