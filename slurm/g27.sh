#!/bin/bash
set -uo pipefail
python scripts/194_g27_state_tomography.py stage=battery || echo "G27 battery FAILED"
python scripts/194_g27_state_tomography.py stage=analyze || echo "G27 analyze FAILED"
