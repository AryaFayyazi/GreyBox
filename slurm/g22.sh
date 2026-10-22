#!/bin/bash
set -uo pipefail
for t in tau2 gptoss20b qwen38 museglimmer claude; do
  python scripts/189_g22_universal_questions.py stage=run target=$t || echo "G22 $t FAILED"
done
python scripts/189_g22_universal_questions.py stage=analyze || echo "G22 analyze FAILED"
