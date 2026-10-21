#!/bin/bash
set -uo pipefail
P=/data/models/cape_ws/envs/agentdojo/bin/python
$P scripts/165_agentdojo_live.py actor=gemma4 split=conf trials=2 arms=spot,spotaudit cut=1.0 tag=g13 || echo "G13 GEMMA FAILED"
$P scripts/165_agentdojo_live.py actor=gptoss20b split=conf trials=2 arms=spot,spotaudit cut=1.0 tag=g13 || echo "G13 GPTOSS FAILED"
