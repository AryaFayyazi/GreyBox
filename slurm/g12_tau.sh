#!/bin/bash
# G12 step A (development): offline per-template self-audit flag rates for four agents.
set -uo pipefail
O=/data/models/graybox_ws/outputs/g12
for A in gemma4 gptoss20b qwen38; do
  [ -f $O/${A}_tau_units.parquet ] || python scripts/161_offline_tau.py actor=$A || echo "TAU FAILED $A"
done
export VLLM_CACHE_ROOT=/data/models/cape_ws/cache/vllm030 TRITON_CACHE_DIR=/data/models/cape_ws/cache/triton030
[ -f $O/museglimmer_tau_units.parquet ] || python scripts/161_offline_tau.py actor=museglimmer vllm_python=/data/models/cape_ws/envs/v030/bin/python || echo "TAU FAILED museglimmer"
python scripts/161_offline_tau.py stage=summary || echo "TAU SUMMARY FAILED"
