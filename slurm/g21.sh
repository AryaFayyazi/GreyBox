#!/bin/bash
set -uo pipefail
python scripts/188_g21_tau2_open_answerer.py stage=run || echo "G21 run FAILED"
python scripts/188_g21_tau2_open_answerer.py stage=analyze || echo "G21 analyze FAILED"
