#!/bin/bash
# Resume CAPE's gpu_guard.py (paused 2026-10-16 ~01:00 with the user's approval, so it would not serialise the
# user-approved second lane) at 08:00 PDT, when the two-GPU exception ends. Logs to slurm/lane_guard.log.
LOG=/home/arya/graybox/slurm/lane_guard.log
while [ $(date +%s) -lt $(date -d "2026-10-16 08:00:00" +%s) ]; do sleep 30; done
kill -CONT 3388394 && echo "$(date '+%F %T') RESUMED CAPE gpu_guard.py (pid 3388394; soft one-GPU mode)" >> $LOG
