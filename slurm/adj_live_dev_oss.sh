#!/bin/bash
set -uo pipefail
P=/data/models/cape_ws/envs/agentdojo/bin/python
$P scripts/165_agentdojo_live.py actor=gptoss20b split=dev trials=1 arms=spot,spotaudit cut=1.0 tag=devpilot || echo "ADJ OSS DEVPILOT FAILED"
