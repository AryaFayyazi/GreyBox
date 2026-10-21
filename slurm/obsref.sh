#!/bin/bash
set -uo pipefail
python web/build_observable_ref.py stage=run || echo "OBSREF RUN FAILED"
python web/build_observable_ref.py stage=build || echo "OBSREF BUILD FAILED"
