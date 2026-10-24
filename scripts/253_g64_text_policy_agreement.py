#!/usr/bin/env python
"""G64: a label-free, rank-based test of the two regimes (docs/prereg_G64_text_policy_agreement.md). Within each
template|tool|position stratum, Spearman correlation between the text monitor's probability q and the agent's own
sampled-policy forecast (G46's forecast of its own draws); averaged over strata weighted by stratum size (strata with >= 5
units). No label is used. Claude Haiku 5.5: G46c units x G52's judge; GPT-6-luna: G48 units x G54's judge.
  -> results/t1/g64_text_policy_agreement.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED = 2000, 6400


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def agreement(q, f, st, i):
    num = den = 0.0
    for s in np.unique(st[i]):
        j = i[st[i] == s]
        if len(j) >= 5 and len(np.unique(q[j])) > 1 and len(np.unique(f[j])) > 1:   # deviation: was np.std(.) > 0 (float noise)
            num += len(j) * spearmanr(q[j], f[j])[0]
            den += len(j)
    return num / den if den else np.nan


def main():
    T = _mod("t249", "scripts/249_forecasters_both_strata_exploratory.py")
    key = T.key
    D = {}
    for agent, (units, ownf, jf) in {"claude": (T.G46C.units, T.G46C.OWN, T.G52.JF), "gpt": (T.G48.units, T.G48.OWN, T.G54.JF)}.items():
        U = units()[0]
        own, jd = T.load(ownf), T.load(jf)
        ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
        m = np.array([k in own and k in jd and jd[k].get("p") is not None for k in ks])
        U, ks = U[m].reset_index(drop=True), [k for k, x in zip(ks, m) if x]
        D[agent] = {"q": np.array([jd[k]["p"] for k in ks], float), "f": np.array([T.G46.forecast(own[k]) for k in ks]),
                    "st": (U.stratum + "|" + U.position.astype(str)).values, "cl": U.scenario_id.values}
    rng = np.random.default_rng(BOOT_SEED)
    B = {a: [] for a in D}
    for _ in range(B_N):
        for a, d in D.items():
            us = np.unique(d["cl"])
            i = np.concatenate([np.flatnonzero(d["cl"] == u) for u in rng.choice(us, len(us))])
            B[a].append(agreement(d["q"], d["f"], d["st"], i))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    full = {a: agreement(d["q"], d["f"], d["st"], np.arange(len(d["q"]))) for a, d in D.items()}
    diff = np.array(B["gpt"]) - np.array(B["claude"])
    res = {a: {"units": int(len(D[a]["q"])), "agreement": [round(float(full[a]), 4)] + ci(B[a])} for a in D}
    res["A1_gpt_minus_claude"] = [round(float(full["gpt"] - full["claude"]), 4)] + ci(diff)
    res["endpoints"] = {"A1_lower_gt_0": res["A1_gpt_minus_claude"][1] > 0}
    (ROOT / "results/t1/g64_text_policy_agreement.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
