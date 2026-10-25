#!/bin/bash
# G68 GPU job: per agent, a 4-unit smoke (separate file, never analysed), then all units. Resumable.
set -uo pipefail
cd /home/arya/graybox
for a in gemma4 gptoss20b qwen38 museglimmer; do
  python scripts/260_g68_open_says_does.py stage=draws actor=$a limit=4 out=g68smoke_$a.jsonl
  python scripts/260_g68_open_says_does.py stage=draws actor=$a
done
python scripts/260_g68_open_says_does.py stage=analyze
