#!/bin/bash
# G12 Claude arm (API, not GPU): core24, cons_service_restart + tp_campaign_test, 6 passes, G10's gate; ledger cap $38.
set -uo pipefail
for P in 0 1 2 3 4 5; do
  /data/models/cape_ws/venv/bin/python scripts/153_claude_self_audit.py stage=live scen_split=core24 split=all n_scen=1000 "templates=[cons_service_restart,tp_campaign_test]" "arms=[none,gate]" prompt=checklist cut=0.67 n_audit=3 cap=38.0 seed=$P tag=g12_claude_p$P || echo "G12 CLAUDE PASS $P FAILED"
done
