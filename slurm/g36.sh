#!/bin/bash
set -uo pipefail
for r in gemma3_4b gptoss20b llama31_8b mistral7b olmo3_7b phi4mini qwen25_7b qwen3_8b; do
  [ -f /data/models/graybox_ws/outputs/g36/$r.npz ] && continue
  python scripts/209_g36_decision_triangulation.py stage=hidden reader=$r || echo "G36 $r FAILED"
done
