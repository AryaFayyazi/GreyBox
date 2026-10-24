#!/usr/bin/env python
"""Exploratory (not preregistered): G47's random arm held fewer calls than the self-gate (0.177 vs 0.227 per call), so
its difference in violating episodes is not at a matched dose. Here: the relative reduction per unit of hold rate
(reduction / holds per call), self vs random, cluster bootstrap over scenarios (2000, seed 2441). Also the episodes'
violations removed per hold. -> results/t1/g47_dose_exploratory.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("p225", ROOT / "scripts/225_selfgate_pilot_exploratory.py")
P = importlib.util.module_from_spec(s)
s.loader.exec_module(P)
E = pd.read_json(P.OUT / "g47.jsonl", orient="records", lines=True)
E["unit"] = E.instance_id + "|" + E.seed.astype(str)
full = E.groupby("unit").arm.nunique()
E = E[E.unit.isin(full.index[full == 3])]
piv = lambda c: E.pivot_table(index="unit", columns="arm", values=c, aggfunc="first").astype(float)  # noqa: E731
v, h, n = piv("unsafe_episode"), piv("n_holds"), piv("n_calls")
sid = E.groupby("unit").scenario_id.first().loc[v.index].values


def stats(i):
    out = {}
    for a in ("self", "random"):
        red = 1 - v[a].values[i].mean() / v["none"].values[i].mean()
        rate = h[a].values[i].sum() / n[a].values[i].sum()
        out[a] = (red / rate, (v["none"].values[i].sum() - v[a].values[i].sum()) / h[a].values[i].sum())
    return out["self"][0] - out["random"][0], out["self"][1] - out["random"][1], out


rng, us = np.random.default_rng(2441), np.unique(sid)
B = [stats(np.concatenate([np.flatnonzero(sid == u) for u in rng.choice(us, len(us))]))[:2] for _ in range(2000)]
d1, d2, o = stats(np.arange(len(v)))
ci = lambda x: [round(float(np.nanpercentile(x, 2.5)), 4), round(float(np.nanpercentile(x, 97.5)), 4)]  # noqa: E731
res = {"units": int(len(v)), "reduction_per_hold_rate": {a: round(float(o[a][0]), 3) for a in o},
       "self_minus_random_reduction_per_hold_rate": [round(float(d1), 3)] + ci([b[0] for b in B]),
       "violating_episodes_removed_per_hold": {a: round(float(o[a][1]), 4) for a in o},
       "self_minus_random_removed_per_hold": [round(float(d2), 4)] + ci([b[1] for b in B])}
(ROOT / "results/t1/g47_dose_exploratory.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
