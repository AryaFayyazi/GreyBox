#!/bin/bash
set -uo pipefail
M=llama33_70b; D=/data/models/graybox_ws/outputs/g6/run/$M
[ -f $D/psi_none.npz ] || python scripts/142_scale_exploratory.py stage=gen model=$M || echo "GEN FAILED $M"
[ -f $D/hidden_none.npz ] || python scripts/142_scale_exploratory.py stage=hidden model=$M || echo "HIDDEN FAILED $M"
python scripts/142_scale_exploratory.py stage=analyze || echo "ANALYZE FAILED"
