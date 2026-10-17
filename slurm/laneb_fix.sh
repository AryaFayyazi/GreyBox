#!/bin/bash
# Slurm 21.08 quirk (2026-10-16 00:40-00:52): lane-B jobs re-showed an old lane-A dependency (afterany:3825) and the
# hold on 3846 was dropped after scontrol updates. Until 3852 ends: every 30 s, re-clear any lane-A link from lane-B
# jobs and keep 3846 held while the Claude run file is missing. Each repair is logged to slurm/lane_guard.log.
LOG=/home/arya/graybox/slurm/lane_guard.log
log() { echo "$(date '+%F %T') LANEB_FIX $*" >> $LOG; }
PREV=( [3851]=3850 [3852]=3851 [3846]=3852 [3853]=3846 [3854]=3853 [3836]=3854 [3841]=3836 [3842]=3841 [3843]=3842 )
while squeue -h -j 3852 -o %T 2>/dev/null | grep -q .; do
  for j in 3851 3852 3846 3853 3854 3836 3841 3842 3843; do
    d=$(scontrol show job $j 2>/dev/null | grep -o -E 'Dependency=[^ ]*')
    if echo "$d" | grep -q -E 'afterany:38(05|06|22|23|24|25)'; then
      scontrol update JobId=$j Dependency= ; scontrol update JobId=$j Dependency=afterany:${PREV[$j]}
      log "job $j had [$d]; reset to afterany:${PREV[$j]}"
    fi
  done
  if [ ! -s /data/models/graybox_ws/outputs/l3/claude_haiku55_run.json ] && \
     ! scontrol show job 3846 | grep -q 'Priority=0 '; then
    scontrol hold 3846 && log "re-held 3846 (Claude run file not yet present)"
  fi
  sleep 30
done
log "3852 finished; fixer exits"
