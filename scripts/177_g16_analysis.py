#!/usr/bin/env python
"""G16 analysis (docs/prereg_G16_own_vs_other.md): the agent's own answers vs another model's answers to the same 8
questions about the same held-out calls, both read with G15's fixed read-out (no training). Within-template AUROC;
scenarios resampled within each agent (2,000, seed 1780). Calls without a stored decision state are dropped (as G15).
  python scripts/177_g16_analysis.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from graybox.stats import cluster_boot, within_auroc  # noqa: E402


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OR = _mod("obsref", "web/build_observable_ref.py")
M = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
OUT = M.OUT
OTHER = {"gemma4": "museglimmer", "gptoss20b": "gemma4", "qwen38": "gemma4", "museglimmer": "gemma4", "claude": "gemma4"}


def mean8(B):
    C = np.array([json.loads(c) for c in B.counts])
    T = OR.theta(C[:, :, 0], C[:, :, 1])
    return np.c_[-T[:, 0], T[:, 1:]].mean(1)


def pair(own, oth):
    k = ["episode_id", "event_id"]
    J = own.merge(oth[k + ["counts"]].rename(columns={"counts": "counts_other"}), on=k)
    return J, mean8(J), mean8(J.assign(counts=J.counts_other))


def main():
    rng = np.random.default_rng(1780)
    res, per = {}, {}
    for a in ["gemma4", "gptoss20b", "qwen38", "museglimmer"]:
        split = "fresh" if a == "gemma4" else "heldout"
        own = pd.read_parquet(OUT / f"{a}_{split}.parquet")
        U, _ = M.states(a)
        bad = set(zip(U.episode_id[~U.ok.values], U.event_id[~U.ok.values].astype(int)))
        own = own[[(e, int(v)) not in bad for e, v in zip(own.episode_id, own.event_id)]]
        J, so, sx = pair(own, pd.read_parquet(OUT / f"{a}_{split}_by_{OTHER[a]}.parquet"))
        y, tm, sc = J.y.values.astype(int), J.template.values, J.scenario_id.values
        per[a] = (y, tm, sc, so, sx)
        res[a] = {"answerer": OTHER[a], "calls": int(len(J)), "own": round(float(within_auroc(y, so, tm)), 4), "other": round(float(within_auroc(y, sx, tm)), 4)}
    W = lambda y, tm, s, i: within_auroc(y[i], s[i], tm[i])  # noqa: E731
    pt = np.mean([W(y, tm, so, np.arange(len(y))) - W(y, tm, sx, np.arange(len(y))) for y, tm, sc, so, sx in per.values()])
    bs = []
    for _ in range(2000):
        v = []
        for y, tm, sc, so, sx in per.values():
            us = np.unique(sc)
            idx = np.concatenate([np.flatnonzero(sc == u) for u in rng.choice(us, len(us))])
            v.append(W(y, tm, so, idx) - W(y, tm, sx, idx))
        bs.append(np.nanmean(v))
    res["O1_own_minus_other_mean"] = [round(float(pt), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]
    f = OUT / "claude_t8b_by_gemma4.parquet"
    if f.exists():
        own = pd.DataFrame([json.loads(x) for x in open(OUT / "claude_t8b.jsonl")])
        own = own[~own.set_index(["episode_id", "event_id"]).index.isin(M.PILOT)]
        J, so, sx = pair(own, pd.read_parquet(f))
        y, tm = J.y.values.astype(int), J.template.values
        rs = J.template.str.startswith("cons_service_restart").values
        d = cluster_boot(lambda i: within_auroc(y[rs][i], so[rs][i], tm[rs][i]) - within_auroc(y[rs][i], sx[rs][i], tm[rs][i]), J.scenario_id.values[rs], 2000, seed=1781)
        d = d[np.isfinite(d)]
        res["claude"] = {"answerer": "gemma4", "restart_own": round(float(within_auroc(y[rs], so[rs], tm[rs])), 4),
                         "restart_other": round(float(within_auroc(y[rs], sx[rs], tm[rs])), 4),
                         "O2_restart_own_minus_other": [round(float(within_auroc(y[rs], so[rs], tm[rs]) - within_auroc(y[rs], sx[rs], tm[rs])), 4),
                                                         round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    lo, hi = res["O1_own_minus_other_mean"][1:]
    res["reading"] = "own answers matter" if lo > 0 else ("another model's answers do better" if hi < 0 else "no detectable difference")
    (ROOT / "results/t1/g16_own_vs_other.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
