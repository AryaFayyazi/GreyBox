#!/bin/bash
set -uo pipefail
A=$1; shift
python scripts/197_g27b_replication.py stage=battery actor=$A "$@" || echo "G27b $A FAILED"
