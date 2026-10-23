#!/usr/bin/env python
"""G44b: is the agent's advantage over another reader specific to forecasting (before the call), as the theory predicts,
rather than a weaker reader? (docs/prereg_G44b_did.md). For each open agent (gpt-oss-20b, Qwen3.8, Muse-Glimmer) the same
other model (Gemma-4) answered in the agent's seat:
  at the call     G16's data: G15's held-out calls, G15's 8 questions with the call shown; within-template AUROC
  before the call G44's data: steps in template x tool x step strata holding both classes, G27's 8 forward questions
DiD_a = (own - other)_before - (own - other)_at. Pooled: mean over agents; independent cluster bootstraps over scenarios in
the two data sets (2000, seed 4450). CPU only.  python scripts/221_g44b_did.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from graybox.stats import within_auroc  # noqa: E402

AGENTS = ["gptoss20b", "qwen38", "museglimmer"]
B_N, SEED = 2000, 4450


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def at_call(G16, a):
    own = pd.read_parquet(G16.OUT / f"{a}_heldout.parquet")
    U, _ = G16.M.states(a)
    bad = set(zip(U.episode_id[~U.ok.values], U.event_id[~U.ok.values].astype(int)))
    own = own[[(e, int(v)) not in bad for e, v in zip(own.episode_id, own.event_id)]]
    J, so, sx = G16.pair(own, pd.read_parquet(G16.OUT / f"{a}_heldout_by_gemma4.parquet"))
    return J.y.values.astype(int), J.template.values, J.scenario_id.values, so, sx


def before_call(G44, a):
    S = pd.read_parquet(G44.OUT / f"{a}_by_gemma4.parquet")
    so = np.array([G44.risk(json.loads(c)) for c in S.counts])
    sx = np.array([G44.risk(json.loads(c)) for c in S.other])
    st = np.array([f"{x}|{z}|{t}" for x, z, t in zip(S.template, S.tool, S.t)])
    return S.y.values.astype(int), st, S.scenario_id.values, so, sx


def gap(d, i):
    y, st, sc, so, sx = d
    return within_auroc(y[i], so[i], st[i]) - within_auroc(y[i], sx[i], st[i])


def boot(d, rng):
    sc = d[2]
    us = np.unique(sc)
    return np.concatenate([np.flatnonzero(sc == u) for u in rng.choice(us, len(us))])


def main():
    sys.argv = [sys.argv[0]]
    G16 = _mod("g16", "scripts/177_g16_analysis.py")
    G44 = _mod("g44", "scripts/220_g44_self_vs_other_forecast.py")
    A = {a: at_call(G16, a) for a in AGENTS}
    B = {a: before_call(G44, a) for a in AGENTS}
    res = {}
    for a in AGENTS:
        ga, gb = gap(A[a], np.arange(len(A[a][0]))), gap(B[a], np.arange(len(B[a][0])))
        res[a] = {"own_minus_other_at_call": round(float(ga), 4), "own_minus_other_before_call": round(float(gb), 4), "did": round(float(gb - ga), 4)}
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(B_N):
        bs.append(np.nanmean([gap(B[a], boot(B[a], rng)) - gap(A[a], boot(A[a], rng)) for a in AGENTS]))
    res["D1_pooled_did"] = [round(float(np.mean([res[a]["did"] for a in AGENTS])), 4),
                            round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]
    res["endpoints"] = {"D1_lower_gt_0": res["D1_pooled_did"][1] > 0}
    (ROOT / "results/t1/g44b_did.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
