#!/bin/bash
set -uo pipefail
python scripts/198c_g29b_precall_gate.py stage=run || echo "G29b run FAILED"
python scripts/198c_g29b_precall_gate.py stage=analyze || echo "G29b analyze FAILED"
