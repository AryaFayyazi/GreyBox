#!/bin/bash
# After the transcript-framing open-model runs (scripts/55, jobs 3872-3877) finish: CPU analysis chain for L3d.
# Order is fixed by docs/prereg_L3d_claude_transcript.md: bundle built (and its sha recorded) before scripts/57 runs.
set -uo pipefail
cd /home/arya/graybox
source /data/models/cape_ws/venv/bin/activate
LOG=/data/models/graybox_ws/outputs/finish_tx.log
while squeue -h -j 3877 -o %T 2>/dev/null | grep -q .; do sleep 60; done
echo "$(date '+%F %T') transcript runs finished; files: $(ls /data/models/graybox_ws/outputs/livebench_transcript)" >> $LOG
W=/data/models/graybox_ws
GB_LB=$W/outputs/livebench_transcript GB_LB_RES=results/livebench_transcript GB_FEAT=theta GB_NSAMP=0 python scripts/38_livebench_analysis.py >> $LOG 2>&1
GB_LB=$W/outputs/livebench_transcript GB_LB_RES=results/livebench_transcript GB_LIVE_DIR=$W/models/live_tx GB_BUNDLE_NAME=readouts_v4 \
  GB_BATTERY_TAG=v2m_transcript GB_VAL=results/livebench_transcript/livebench_theta_nexact.json python scripts/39_live_v2_readouts.py >> $LOG 2>&1
SHA=$(sha256sum $W/models/live_tx/readouts_v4.npz | cut -d' ' -f1)
echo "- **L3d bundle built $(date '+%F %T %Z').** models/live_tx/readouts_v4.npz sha256 $SHA (recorded before scripts/57 ran)." >> docs/PREREGISTRATION_RECORD.md
chmod a-w $W/models/live_tx/readouts_v4.npz
python scripts/57_claude_transcript_analysis.py >> $LOG 2>&1
GB_LB=$W/outputs/livebench_transcript GB_LB_RES=results/livebench_transcript GB_FEAT=theta GB_NSAMP=0 python scripts/48_alarm_eval.py >> $LOG 2>&1
python web/build_browser_live.py /tmp/claude-1006/-home-arya-graybox/db8cdb77-a6b6-4066-98d2-8165aa2edb5c/scratchpad/artifact/grey-box-live-browser.html >> $LOG 2>&1
echo "$(date '+%F %T') FINISHED" >> $LOG
