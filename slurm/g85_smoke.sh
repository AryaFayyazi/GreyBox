#!/bin/bash
set -uo pipefail
for a in gemma4 gptoss20b qwen38 museglimmer; do python scripts/287_g85_open_self_monitor.py stage=smoke actor=$a || echo "SMOKE $a FAILED"; done
