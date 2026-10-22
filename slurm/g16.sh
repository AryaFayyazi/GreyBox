#!/bin/bash
set -uo pipefail
python scripts/176_other_answerer.py answerer=gemma4 target=gptoss20b || echo "G16 gptoss FAILED"
python scripts/176_other_answerer.py answerer=gemma4 target=claude || echo "G16 claude FAILED"
python scripts/176_other_answerer.py answerer=gemma4 target=museglimmer || echo "G16 muse FAILED"
python scripts/176_other_answerer.py answerer=gemma4 target=qwen38 || echo "G16 qwen FAILED"
python scripts/176_other_answerer.py answerer=museglimmer target=gemma4 vllm_python=/data/models/cape_ws/envs/v030/bin/python || echo "G16 gemma FAILED"
