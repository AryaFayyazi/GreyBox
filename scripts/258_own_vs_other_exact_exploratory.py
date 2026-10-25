#!/usr/bin/env python
"""Exploratory (no claim registered): (a) G46's R2 comparison, own draws vs Gemma-4's draws in Claude's seat, recomputed
with G46c's EXACT contexts (G46 rebuilt contexts from logs and dropped refused extra tool calls); scripts/226's forecast
and strata unchanged, within template|tool (R2's stratum) and within position. (b) Descriptive across-step values on G65's
fresh episodes for the forecast table: own draws alone and the step alone (within template|tool).
  -> results/t1/own_vs_other_exact_exploratory.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED = 2000, 5800


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    from graybox.stats import within_auroc
    G46C = _mod("g46c", "scripts/232_g46c_exact_context.py")
    G46 = G46C.G46
    U, _ = G46C.units()
    key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
    own = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G46C.OWN))}
    oth = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G46C.OTHER))}
    ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    m = np.array([k in own and k in oth for k in ks])
    U, ks = U[m].reset_index(drop=True), [k for k, x in zip(ks, m) if x]
    o, g = np.array([G46.forecast(own[k]) for k in ks]), np.array([G46.forecast(oth[k]) for k in ks])
    y, cl = U.y.values, U.scenario_id.values
    st2, st3 = U.stratum.values, (U.stratum + "|" + U.position.astype(str)).values
    W = lambda s, st, i: within_auroc(y[i], s[i], st[i])  # noqa: E731
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(cl)
    B2, B3 = [], []
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        B2.append(W(o, st2, i) - W(g, st2, i)); B3.append(W(o, st3, i) - W(g, st3, i))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    a = np.arange(len(y))
    res = {"g46c_exact": {"units": int(len(y)), "violations": int(y.sum()),
                          "within_template_tool": {"own": round(float(W(o, st2, a)), 4), "other_gemma4": round(float(W(g, st2, a)), 4),
                                                   "own_minus_other": [round(float(W(o, st2, a) - W(g, st2, a)), 4)] + ci(B2)},
                          "within_position": {"own": round(float(W(o, st3, a)), 4), "other_gemma4": round(float(W(g, st3, a)), 4),
                                              "own_minus_other": [round(float(W(o, st3, a) - W(g, st3, a)), 4)] + ci(B3)}}}
    REP = _mod("g65", "scripts/254_g65_replication.py")
    V, _ = REP.G46C.units()
    own2 = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(REP.G46C.OWN))}
    k2 = [key(a_, b, c) for a_, b, c in zip(V.instance_id, V.seed, V.t)]
    m2 = np.array([k in own2 for k in k2])
    V, k2 = V[m2].reset_index(drop=True), [k for k, x in zip(k2, m2) if x]
    F = np.array([G46.forecast(own2[k]) for k in k2])
    res["g65_across_steps"] = {"own_draws": round(float(within_auroc(V.y.values, F, V.stratum.values)), 4),
                               "step_alone": round(float(within_auroc(V.y.values, V.position.values.astype(float), V.stratum.values)), 4)}
    (ROOT / "results/t1/own_vs_other_exact_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
