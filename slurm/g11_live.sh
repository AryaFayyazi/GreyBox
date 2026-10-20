#!/bin/bash
# G11 (docs/prereg_G11_selfpolicing_law.md): G9's gate@1.0 protocol, unchanged, on Muse-Glimmer and Qwen3.8 (v4fresh).
set -uo pipefail
python scripts/151_live_selfaudit.py actor=qwen38 split=v4fresh seeds=[0,1] limit=null arms=[none,gate] cut=1.0 escalate=review tag=g11_c100 || echo "G11 QWEN FAILED"
export VLLM_CACHE_ROOT=/data/models/cape_ws/cache/vllm030 TRITON_CACHE_DIR=/data/models/cape_ws/cache/triton030
python scripts/151_live_selfaudit.py actor=museglimmer split=v4fresh seeds=[0,1] limit=null arms=[none,gate] cut=1.0 escalate=review tag=g11_c100 vllm_python=/data/models/cape_ws/envs/v030/bin/python || echo "G11 MUSE FAILED"
python scripts/156_g11_analysis.py || echo "G11 ANALYSIS FAILED"
