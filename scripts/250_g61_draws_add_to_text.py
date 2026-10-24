#!/usr/bin/env python
"""G61: across steps, do the agent's sampled actions add to what a text monitor (or the step) already gives?
(docs/prereg_G61_draws_add_to_text.md). Claude Haiku 5.5, G46c's units (exact contexts; 1,587 calls). Combination with
no fit and no label: the mean of the two scores' ranks over all units. AUROC within template|tool (across steps).
  -> results/t1/g61_draws_add_to_text.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED = 2000, 6100


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    from graybox.stats import within_auroc
    T = _mod("t249", "scripts/249_forecasters_both_strata_exploratory.py")
    G46C, G46 = T.G46C, T.G46
    U, _ = G46C.units()
    key = T.key
    ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    own, cj, gj = T.load(G46C.OWN), T.load(T.G52.JF), T.load(T.G55.JF)
    fill = lambda v: np.where(np.isnan(v), np.nanmedian(v), v)  # noqa: E731
    D = fill(np.array([G46.forecast(own[k]) if k in own else np.nan for k in ks]))
    C = fill(np.array([cj[k]["p"] if k in cj and cj[k].get("p") is not None else np.nan for k in ks]))
    G = fill(np.array([gj[k]["p"] if k in gj and gj[k].get("p") is not None else np.nan for k in ks]))
    P = U.position.values.astype(float)
    r = lambda v: rankdata(v) / len(v)  # noqa: E731
    S = {"draws": D, "gpt_judge": G, "claude_judge": C, "position": P,
         "gpt_judge+draws": (r(G) + r(D)) / 2, "claude_judge+draws": (r(C) + r(D)) / 2, "position+draws": (r(P) + r(D)) / 2}
    y, st, cl = U.y.values, U.stratum.values, U.scenario_id.values
    pairs = {"A1_gpt_judge+draws_minus_gpt_judge": ("gpt_judge+draws", "gpt_judge"),
             "A2_claude_judge+draws_minus_claude_judge": ("claude_judge+draws", "claude_judge"),
             "A3_position+draws_minus_position": ("position+draws", "position")}
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(cl)
    B = {p: [] for p in pairs}
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        for p, (a, b) in pairs.items():
            B[p].append(within_auroc(y[i], S[a][i], st[i]) - within_auroc(y[i], S[b][i], st[i]))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    W = {k: round(float(within_auroc(y, v, st)), 4) for k, v in S.items()}
    res = {"units": int(len(y)), "violations": int(y.sum()), "across_steps_auroc": W}
    for p, (a, b) in pairs.items():
        res[p] = [round(W[a] - W[b], 4)] + ci(B[p])
    res["endpoints"] = {p.split("_")[0] + "_lower_gt_0": res[p][1] > 0 for p in pairs}
    (ROOT / "results/t1/g61_draws_add_to_text.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
