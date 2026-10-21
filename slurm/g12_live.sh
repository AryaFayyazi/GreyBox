#!/bin/bash
# G12 (docs/prereg_G12_composition_law.md): open agents on core24, gate@1.0 (G9's protocol), templates fixed in the prereg.
set -uo pipefail
python scripts/162_live_g12.py actor=gemma4 "templates=[auth_repo_access,con_cloud_cost,cons_service_restart,cred_analytics_token,own_dataset_upload,prov_action_list,prov_meeting_notes,tp_billing_correction,tp_calendar_cancel,tp_campaign_test]" seeds=[0] arms=[none,gate] cut=1.0 escalate=review tag=g12_c100 || echo "G12 GEMMA FAILED"
python scripts/162_live_g12.py actor=gptoss20b "templates=[con_ci_green,cons_migration,prov_action_list,prov_meeting_notes,prov_pr_comment,rev_config_overwrite,tp_campaign_test]" seeds=[0,1] arms=[none,gate] cut=1.0 escalate=review tag=g12_c100 || echo "G12 GPTOSS FAILED"
python scripts/162_live_g12.py actor=qwen38 "templates=[con_ci_green,tp_campaign_test]" seeds=[0,1,2,3] arms=[none,gate] cut=1.0 escalate=review tag=g12_c100 || echo "G12 QWEN FAILED"
export VLLM_CACHE_ROOT=/data/models/cape_ws/cache/vllm030 TRITON_CACHE_DIR=/data/models/cape_ws/cache/triton030
python scripts/162_live_g12.py actor=museglimmer "templates=[con_ci_green,con_cloud_cost,tp_campaign_test]" seeds=[0,1,2] arms=[none,gate] cut=1.0 escalate=review tag=g12_c100 vllm_python=/data/models/cape_ws/envs/v030/bin/python || echo "G12 MUSE FAILED"
