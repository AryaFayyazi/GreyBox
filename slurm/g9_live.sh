#!/bin/bash
# G9 (docs/prereg_G9_live_self_policing.md): Gemma-4 on v4fresh, 333 unsafe non-nuisance scenarios x seeds 0,1.
set -uo pipefail
python scripts/151_live_selfaudit.py actor=gemma4 split=v4fresh seeds=[0,1] limit=null arms=[none,gate] cut=1.0 escalate=review tag=g9_c100 || echo "G9 A FAILED"
python scripts/151_live_selfaudit.py actor=gemma4 split=v4fresh seeds=[0,1] limit=null arms=[S4] cut=0.75 escalate=review tag=g9_c075 || echo "G9 B FAILED"
python scripts/152_live_selfaudit_analysis.py "files=[gemma4_v4fresh_g9_c100,gemma4_v4fresh_g9_c075]" endpoints=g9 out=g9_live_self_policing.json || echo "G9 ANALYSIS FAILED"
