#!/usr/bin/env python
"""Pilot (development split only): which AgentDojo attack makes Claude Haiku 5.5 carry out injected instructions often
enough to test a monitor? For one attack (env G17_ATTACK), ~60 attack episodes stratified over the four suites (15 per
suite, seed 1830), no defense. Reports the share of episodes with a call carrying an attacker-only value.
  G17_ATTACK=tool_knowledge /data/models/cape_ws/envs/agentdojo/bin/python -I scripts/183_claude_attack_pilot.py <cap>
"""
import json
import os
import random
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor

cap = sys.argv[1]
sys.argv = ["x", "none", "dev", "attack", cap]
g = runpy.run_path("/home/arya/graybox/scripts/182_claude_agentdojo.py", run_name="pilot")
m, LED, ATTACK = g["m"], g["LED"], os.environ.get("G17_ATTACK", "important_instructions")
NPER = int(os.environ.get("G17_NPER", "15"))
rng = random.Random(1830)
jobs = []
for sn in m.SUITES:
    S = m.get_suites(m.VERSION)[sn]
    pool = [(sn, ut, it, 0) for ut in S.user_tasks if m.split_of(sn, ut) == "dev" for it in S.injection_tasks]
    jobs += rng.sample(pool, min(NPER, len(pool)))
out = f"/data/models/graybox_ws/outputs/agentdojo_claude/pilot_{os.environ.get('G17_MODEL', 'claude-haiku-5-5')}_{ATTACK}.jsonl"
with ThreadPoolExecutor(6) as ex:
    rows = list(ex.map(lambda j: m.episode("none", *j), jobs))
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, default=str) + "\n")
ok = [r for r in rows if r["error"] is None]
inj = sum(any(c.get("injected") for c in r["calls"]) for r in ok)
print(json.dumps({"attack": ATTACK, "episodes": len(rows), "errors": len(rows) - len(ok), "attack_success": sum(bool(r["security"]) for r in ok),
                  "episodes_with_injected_call": inj, "share": round(inj / max(1, len(ok)), 3), "ledger": round(LED.total(), 3)}))
