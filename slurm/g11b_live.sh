#!/bin/bash
# G11b (docs/prereg_G11b_selfpolicing_law.md)
set -uo pipefail
python scripts/158_live_targeted.py actor=qwen38 "templates=[con_ci_green,tp_campaign_test]" seeds=[0,1,2,3,4,5] arms=[none,gate] cut=1.0 escalate=review tag=g11b_c100 || echo "G11b QWEN FAILED"
export VLLM_CACHE_ROOT=/data/models/cape_ws/cache/vllm030 TRITON_CACHE_DIR=/data/models/cape_ws/cache/triton030
python scripts/158_live_targeted.py actor=museglimmer "templates=[con_ci_green,con_cloud_cost,cred_deploy_key,tp_campaign_test]" seeds=[0,1,2,3] arms=[none,gate] cut=1.0 escalate=review tag=g11b_c100 vllm_python=/data/models/cape_ws/envs/v030/bin/python || echo "G11b MUSE FAILED"
python scripts/159_g11b_analysis.py || echo "G11b ANALYSIS FAILED"
