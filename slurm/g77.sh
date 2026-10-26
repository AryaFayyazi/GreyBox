#!/bin/bash
set -uo pipefail
for t in gpt tau2 adgemma4 adgptoss20b; do
  [ -f /data/models/graybox_ws/outputs/g77/judge_$t.parquet ] || python scripts/269_g77_fill_cells.py stage=judge target=$t || echo "G77 judge $t FAILED"
done
python scripts/269_g77_fill_cells.py stage=other65 || echo "G77 other65 FAILED"
python scripts/269_g77_fill_cells.py stage=rules target=all || echo "G77 rules FAILED"
python scripts/269_g77_fill_cells.py stage=analyze || echo "G77 analyze FAILED"
