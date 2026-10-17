#!/bin/bash
# Live v2 pipeline (one GPU job): benchmark analysis (exact logprob tier, θ-only, θ-only text tier n=8) -> v2 readout
# bundle -> v2m truth overlay for Qwen3-8B -> recorded v2 session. Run after the livebench judge (37).
set -euo pipefail
GB_FEAT=theta_lo GB_NSAMP=0 python scripts/38_livebench_analysis.py
GB_FEAT=theta GB_NSAMP=0 python scripts/38_livebench_analysis.py
GB_FEAT=theta GB_NSAMP=8 python scripts/38_livebench_analysis.py
GB_M=24 GB_FEAT=theta GB_NSAMP=8 python scripts/38_livebench_analysis.py
GB_M=24 GB_FEAT=theta GB_NSAMP=0 python scripts/38_livebench_analysis.py
python scripts/39_live_v2_readouts.py
GB_M=24 python scripts/39_live_v2_readouts.py
python scripts/40_live_open_v2m.py model=qwen3_8b
python scripts/27_live_record.py model=qwen3_8b bundle=v2
python scripts/43_judge_sessions.py
