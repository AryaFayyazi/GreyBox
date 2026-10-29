#!/bin/bash
set -uo pipefail
for a in gemma4 gptoss20b qwen38 museglimmer; do python scripts/287_g85_open_self_monitor.py stage=judge actor=$a || echo "G85 judge $a FAILED"; done
python scripts/287_g85_open_self_monitor.py stage=analyze || echo "G85 analyze FAILED"
