#!/bin/bash
set -uo pipefail
for M in qwen3_14b qwen3_32b gemma4_26b; do
  D=/data/models/graybox_ws/outputs/g6/run/$M
  [ -f $D/psi_none.npz ] || python scripts/142_scale_exploratory.py stage=gen model=$M || echo "GEN FAILED $M"
  [ -f $D/hidden_none.npz ] || python scripts/142_scale_exploratory.py stage=hidden model=$M || echo "HIDDEN FAILED $M"
done
python scripts/142_scale_exploratory.py stage=analyze || echo "ANALYZE FAILED"
