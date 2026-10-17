#!/bin/bash
# After the in-context one-call open-model runs (scripts/47, jobs 3865-3870) finish: CPU analysis chain for L3c.
# Order fixed by docs/prereg_L3c_claude_batch.md: bundle built (sha recorded) before scripts/51 runs.
set -uo pipefail
cd /home/arya/graybox
source /data/models/cape_ws/venv/bin/activate
LOG=/data/models/graybox_ws/outputs/finish_ic.log
while squeue -h -j 3870 -o %T 2>/dev/null | grep -q .; do sleep 60; done
echo "$(date '+%F %T') in-context runs finished; files: $(ls /data/models/graybox_ws/outputs/livebench_batch)" >> $LOG
W=/data/models/graybox_ws
GB_LB=$W/outputs/livebench_batch GB_LB_RES=results/livebench_batch GB_FEAT=theta GB_NSAMP=0 python scripts/38_livebench_analysis.py >> $LOG 2>&1
GB_LB=$W/outputs/livebench_batch GB_LB_RES=results/livebench_batch GB_LIVE_DIR=$W/models/live_batch GB_BUNDLE_NAME=readouts_v3 \
  GB_BATTERY_TAG=v2m_batch GB_VAL=results/livebench_batch/livebench_theta_nexact.json python scripts/39_live_v2_readouts.py >> $LOG 2>&1
SHA=$(sha256sum $W/models/live_batch/readouts_v3.npz | cut -d' ' -f1)
echo "- **L3c bundle built $(date '+%F %T %Z').** models/live_batch/readouts_v3.npz sha256 $SHA (recorded before scripts/51 ran)." >> docs/PREREGISTRATION_RECORD.md
chmod a-w $W/models/live_batch/readouts_v3.npz
python scripts/51_claude_batch_analysis.py >> $LOG 2>&1
GB_LB=$W/outputs/livebench_batch GB_LB_RES=results/livebench_batch GB_FEAT=theta GB_NSAMP=0 python scripts/48_alarm_eval.py >> $LOG 2>&1
echo "$(date '+%F %T') FINISHED" >> $LOG
