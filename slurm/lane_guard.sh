#!/bin/bash
# Lane guard (user rule 2026-10-16 ~00:30): two GPUs allowed until 08:00 PDT 2026-10-16, then ONE GPU at a time.
#   Before 08:00 (yield): if another user's GPU job is waiting and we run 2 jobs, requeue+hold our most recently
#     started job; release it once nobody is waiting (adapted from CAPE slurm/yield_guard.sh).
#   At 08:00 (collapse, once): if lane B still runs alongside lane A, requeue+hold lane B's running job; then chain
#     lane B's first remaining job after lane A's last remaining job and release it -> strictly one lane.
# Lanes (job ids in order) are read from slurm/lanes.env. Logs: slurm/lane_guard.log. Runs detached (nohup).
ME=$USER; DIR=/home/arya/graybox/slurm; LOG=$DIR/lane_guard.log; STATE=$DIR/.lane_held
DEADLINE=$(date -d "2026-10-16 08:00:00" +%s)
touch $STATE
log() { echo "$(date '+%F %T') $*" >> $LOG; }
alive() { squeue -h -j "$1" -o "%T" 2>/dev/null | grep -q . ; }
log "guard started (pid $$); deadline $(date -d @$DEADLINE)"
COLLAPSED=0
while true; do
  source $DIR/lanes.env      # LANE_A="..." LANE_B="..."
  NOW=$(date +%s)
  MINE=$(squeue -h -u $ME -t RUNNING -o "%S|%i" 2>/dev/null | sort | awk -F'|' '{print $2}')
  NMINE=$(echo "$MINE" | grep -c .)
  if [ $NOW -ge $DEADLINE ] && [ $COLLAPSED -eq 0 ]; then
    A_LEFT=$(for j in $LANE_A; do alive $j && echo $j; done)
    B_LEFT=$(for j in $LANE_B; do alive $j && echo $j; done)
    if [ -n "$A_LEFT" ] && [ -n "$B_LEFT" ]; then
      for j in $B_LEFT; do
        if squeue -h -j $j -t RUNNING -o "%i" | grep -q .; then
          scontrol requeuehold $j 2>>$LOG && log "COLLAPSE: requeue+hold running lane-B job $j"
        fi
      done
      TAIL_A=$(echo "$A_LEFT" | tail -1); HEAD_B=$(echo "$B_LEFT" | head -1)
      scontrol update JobId=$HEAD_B Dependency=afterany:$TAIL_A 2>>$LOG && log "COLLAPSE: lane B head $HEAD_B now after lane A tail $TAIL_A"
      scontrol release $HEAD_B 2>>$LOG && log "COLLAPSE: released $HEAD_B (waits on dependency)"
    else
      log "COLLAPSE: nothing to merge (A left: $(echo $A_LEFT); B left: $(echo $B_LEFT))"
    fi
    : > $STATE
    COLLAPSED=1
  elif [ $NOW -lt $DEADLINE ]; then
    WAIT=$(squeue -h -t PENDING -o "%i|%u|%b|%r" 2>/dev/null | awk -F'|' -v me=$ME '$2!=me && $3 ~ /gpu/ && $4 !~ /Dependency|JobHeld|BeginTime/ {print $1":"$2":"$4}')
    if [ -n "$WAIT" ] && [ "$NMINE" -ge 2 ]; then
      VID=$(echo "$MINE" | tail -1)
      log "YIELD: other user waiting [$WAIT]; we run $NMINE; requeue+hold our latest $VID"
      scontrol requeuehold $VID 2>>$LOG && echo $VID >> $STATE
    elif [ -z "$WAIT" ] && [ -s $STATE ]; then
      for J in $(sort -u $STATE); do
        squeue -h -j $J -o "%r" 2>/dev/null | grep -q JobHeld && scontrol release $J 2>>$LOG && log "RESUME: released $J"
      done
      : > $STATE
    fi
  fi
  [ $COLLAPSED -eq 1 ] && [ $NMINE -le 1 ] && log "after deadline: $NMINE job(s) running" && sleep 600 && continue
  sleep 20
done
