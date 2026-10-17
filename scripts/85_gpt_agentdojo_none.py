#!/usr/bin/env python
"""GPT-6-luna on AgentDojo WITHOUT the spotlighting defense (arm 'none': AgentDojo's undefended pipeline), with the
same harness, model call, labels and ledger as scripts/83 (imported unchanged; scripts/83 stays locked for T13).
  pilot  40 attack episodes of the dev split, a fixed random sample (rng 85), never analysed with monitors
  run    both splits, attack + benign  -> outputs/agentdojo_gpt/{dev,conf}_gpt6luna_none_{attack,benign}.jsonl
Runs in the AgentDojo env: /data/models/cape_ws/envs/agentdojo/bin/python -I scripts/85_gpt_agentdojo_none.py pilot|run
"""
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("gb83", Path(__file__).with_name("83_gpt_agentdojo.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
m = g.m

if __name__ == "__main__":
    g.OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "pilot":
        jobs = []
        for sn in m.SUITES:
            S = m.get_suites(m.VERSION)[sn]
            for ut in S.user_tasks:
                if m.split_of(sn, ut) == "dev":
                    jobs += [(sn, ut, it, 0) for it in S.injection_tasks]
        jobs = random.Random(85).sample(sorted(jobs), 40)
        rows = []
        with open(g.OUT / "pilot_none.jsonl", "w") as f:
            for j in jobs:
                r = m.episode("none", *j)
                rows.append(r)
                f.write(json.dumps(r, default=str) + "\n")
        ok = [r for r in rows if r["error"] is None]
        inj = sum(bool(c.get("injected")) for r in ok for c in r["calls"])
        print(f"pilot none: {len(ok)}/{len(rows)} ok; attack success {sum(bool(r['security']) for r in ok)}/{len(ok)}; injected calls {inj}; spend ${g.LED.total():.3f}")
    else:
        os.environ.setdefault("V74_CONC", "8")
        for split in ("dev", "conf"):
            m.run("none", split, "attack", f"{split}_gpt6luna_none_attack", 1)
            m.run("none", split, "benign", f"{split}_gpt6luna_none_benign", 1)
