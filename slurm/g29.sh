#!/bin/bash
set -uo pipefail
python scripts/198_g29_live_forecast_gate.py stage=run || echo "G29 run FAILED"
python scripts/198_g29_live_forecast_gate.py stage=analyze || echo "G29 analyze FAILED"
