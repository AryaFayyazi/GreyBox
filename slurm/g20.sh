#!/bin/bash
set -uo pipefail
for t in gptoss20b qwen38 museglimmer gemma4 claude; do
  python scripts/187_g20_reasoning_judge.py stage=judge target=$t || echo "G20 $t FAILED"
done
python scripts/187_g20_reasoning_judge.py stage=analyze || echo "G20 analyze FAILED"
