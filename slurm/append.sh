#!/bin/bash
# Append a job to the END of my serial GPU chain (one GPU at a time), then verify at most one of my jobs runs.
# usage: slurm/append.sh <job-name> <command...>
# The tail = my pending job that no other pending job of mine depends on (or the running job if nothing is pending).
set -euo pipefail
NAME=$1; shift
cd /home/arya/graybox
PEND=$(squeue -h -u $USER -t PENDING -o %i)
DEPS=$(for j in $PEND; do scontrol show job $j | grep -o -E 'Dependency=afterany:[0-9]+' | cut -d: -f2; done)
TAIL=""
for j in $PEND; do echo "$DEPS" | grep -qx "$j" || TAIL=$j; done
[ -z "$TAIL" ] && TAIL=$(squeue -h -u $USER -t RUNNING -o %i | head -1)
if [ -n "$TAIL" ]; then J=$(sbatch --parsable -J "$NAME" --dependency=afterany:$TAIL slurm/run.sh "$@"); else J=$(sbatch --parsable -J "$NAME" slurm/run.sh "$@"); fi
sleep 3
N=$(squeue -h -u $USER -t RUNNING | wc -l)
echo "submitted $J after ${TAIL:-nothing}; running jobs: $N"
[ "$N" -le 1 ] || { echo "WARNING: $N jobs running; requeue+hold $J"; scontrol requeuehold $J; }
