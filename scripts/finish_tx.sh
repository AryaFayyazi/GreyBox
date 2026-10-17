#!/bin/bash
# After the six transcript-framing runs (scripts/55): open-model analysis -> frozen v4 bundle (L3d's exact command)
# -> bundle hash into the preregistration record -> L3d analysis -> development alarm evaluation. CPU only.
set -euo pipefail
cd /home/arya/graybox
source /data/models/cape_ws/venv/bin/activate
W=/data/models/graybox_ws
export GB_LB=$W/outputs/livebench_transcript GB_LB_RES=/home/arya/graybox/results/livebench_transcript
GB_FEAT=theta GB_NSAMP=0 python scripts/38_livebench_analysis.py > $GB_LB_RES.log38 2>&1
GB_LIVE_DIR=$W/models/live_tx GB_BUNDLE_NAME=readouts_v4 GB_BATTERY_TAG=v2m_transcript \
  GB_VAL=$GB_LB_RES/livebench_theta_nexact.json python scripts/39_live_v2_readouts.py > $GB_LB_RES.log39 2>&1
chmod a-w $W/models/live_tx/readouts_v4.npz
H=$(sha256sum $W/models/live_tx/readouts_v4.npz | cut -c1-64)
echo "- **L3d bundle built $(date '+%F %T %Z')** (before scripts/57 ran): models/live_tx/readouts_v4.npz sha256 $H; read-only." >> docs/PREREGISTRATION_RECORD.md
python scripts/57_claude_transcript_analysis.py > results/l3/claude_transcript.log 2>&1
GB_FEAT=theta GB_NSAMP=0 python scripts/48_alarm_eval.py > $GB_LB_RES.log48 2>&1
echo FINISHED
