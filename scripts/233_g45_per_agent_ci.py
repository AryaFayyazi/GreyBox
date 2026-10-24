#!/usr/bin/env python
"""Per-agent 95% intervals for G45's own within-position AUROC (the registered endpoint S2 is pooled; this reports each
agent's value with the same cluster bootstrap over scenarios, 2000 resamples, seed 4500). -> results/t1/g45_per_agent_ci.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("g45", ROOT / "scripts/222_g45_closed_self_forecast.py")
G45 = importlib.util.module_from_spec(s)
s.loader.exec_module(G45)
from graybox.stats import within_auroc  # noqa: E402

out = {}
for t in ("gpt", "claude"):
    U = pd.read_parquet(G45.OUT / f"{t}_by_gemma4.parquet")
    own = np.array([G45.G44.risk(json.loads(c)) for c in U.own])
    y, cl = U.y.values, U.scenario_id.values
    st = np.array([f"{a}|{b}" for a, b in zip(U.stratum, U.position)])
    rng, us, B = np.random.default_rng(4500), np.unique(cl), []
    for _ in range(2000):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        B.append(within_auroc(y[i], own[i], st[i]))
    out[t] = [round(float(within_auroc(y, own, st)), 4)] + [round(float(x), 4) for x in np.nanpercentile(B, [2.5, 97.5])]
(ROOT / "results/t1/g45_per_agent_ci.json").write_text(json.dumps({"own_within_position": out}, indent=1))
print(out)
