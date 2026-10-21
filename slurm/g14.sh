#!/bin/bash
set -uo pipefail
python scripts/171_decision_greybox.py stage=battery actor=gemma4 split=fresh || echo "G14 gemma4 FAILED"
python scripts/171_decision_greybox.py stage=battery actor=gptoss20b split=heldout || echo "G14 gptoss20b FAILED"
python scripts/171_decision_greybox.py stage=battery actor=museglimmer split=heldout vllm_python=/data/models/cape_ws/envs/v030/bin/python || echo "G14 museglimmer FAILED"
python scripts/171_decision_greybox.py stage=battery actor=qwen38 split=heldout || echo "G14 qwen38 FAILED"
