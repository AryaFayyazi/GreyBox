#!/bin/bash
set -uo pipefail
A=$1; shift
python scripts/203_g31_dither.py stage=battery actor=$A "$@" || echo "G31 $A FAILED"
