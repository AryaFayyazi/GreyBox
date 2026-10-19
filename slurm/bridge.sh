#!/bin/bash
set -uo pipefail
for R in minilm mpnet qemb06 q06 q17 q4 q14; do
  [ -f /data/models/graybox_ws/outputs/bridge/${R}_museglimmer_dev.npz ] || python scripts/148_bridge_geometry_transfer.py stage=features reader=$R split=dev || echo "FEAT FAILED $R"
done
python scripts/148_bridge_geometry_transfer.py stage=analyze split=dev || echo "ANALYZE FAILED"
