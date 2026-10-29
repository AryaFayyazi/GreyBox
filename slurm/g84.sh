#!/bin/bash
set -uo pipefail
python scripts/286_g84_open_judge_vs_geometry.py stage=judge || echo "G84 judge FAILED"
python scripts/286_g84_open_judge_vs_geometry.py stage=analyze || echo "G84 analyze FAILED"
