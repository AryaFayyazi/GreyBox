#!/bin/bash
set -uo pipefail
python scripts/171_decision_greybox.py stage=battery actor=gptoss20b split=dev || echo "DGB DEV gptoss20b FAILED"
python scripts/171_decision_greybox.py stage=battery actor=qwen38 split=dev || echo "DGB DEV qwen38 FAILED"
python scripts/171_decision_greybox.py stage=battery actor=museglimmer split=dev vllm_python=/data/models/cape_ws/envs/v030/bin/python || echo "DGB DEV museglimmer FAILED"
