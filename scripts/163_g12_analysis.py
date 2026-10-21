#!/usr/bin/env python
"""G12 analysis (docs/prereg_G12_composition_law.md): the composition-weighted prevention law.

For each agent: tau_k = offline per-template flag rate (results/t1/g12_offline_tau.json, step A, development);
w_k = share of the PLAIN arm's violating episodes that fall on template k (same live test; the plain arm carries no
information about the gate's effect); predicted reduction = sum_k w_k tau_k. Observed = 1 - gate/plain violating-
episode rate over the agent's scenarios. Bootstrap over scenario instances (all seeds together), 2,000, seed 1620; the
prediction is recomputed inside each resample (its w_k change with the resampled plain arm).
  python scripts/163_g12_analysis.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
A = importlib.import_module("152_live_selfaudit_analysis")
AGENTS, TOL, NB, SEED = ("gemma4", "gptoss20b", "qwen38", "museglimmer"), 0.15, 2000, 1620


def analyse(a, tau):
    E = A.per_episode(A.load([f"{a}_core24_g12_c100"]))
    d = A.load([f"{a}_core24_g12_c100"]).drop_duplicates(["arm", "instance_id", "seed"])
    tmpl = d.groupby("instance_id").template.first()
    P = E.pivot_table(index="scen", columns="arm", values="viol")
    P = P.dropna()
    P["tm"] = tmpl.reindex(P.index).values
    T = sorted(set(P.tm) & set(tau))
    P = P[P.tm.isin(T)]
    t = np.array([tau[k]["tau"] for k in T])

    def stats(Q):
        n, g = Q["none"].values, Q["gate@1.00"].values
        w = np.array([n[Q.tm.values == k].sum() for k in T])
        w = w / w.sum() if w.sum() > 0 else w
        return 1 - g.mean() / max(n.mean(), 1e-9), float((w * t).sum()), w
    obs, pred, w = stats(P)
    rng = np.random.default_rng(SEED)
    bo, bp = [], []
    for _ in range(NB):
        Q = P.iloc[rng.integers(0, len(P), len(P))]
        o, p, _ = stats(Q)
        bo.append(o)
        bp.append(p)
    err = np.array(bo) - np.array(bp)
    return {"scenarios": int(len(P)), "templates": T, "weights": [round(float(x), 4) for x in w], "tau": [round(float(x), 4) for x in t],
            "plain_violation": round(float(P["none"].mean()), 4), "gate_violation": round(float(P["gate@1.00"].mean()), 4),
            "observed": [round(float(obs), 4), round(float(np.percentile(bo, 2.5)), 4), round(float(np.percentile(bo, 97.5)), 4)],
            "predicted": round(pred, 4), "observed_minus_predicted": [round(float(obs - pred), 4), round(float(np.percentile(err, 2.5)), 4),
                                                                       round(float(np.percentile(err, 97.5)), 4)]}


def main():
    tau = json.loads((ROOT / "results/t1/g12_offline_tau.json").read_text())
    res = {a: analyse(a, {k: v for k, v in tau[a].items() if v["n"] >= 10}) for a in AGENTS}
    for a in AGENTS:
        print(a, json.dumps(res[a]), flush=True)
    E = {f"Q1_{a}": abs(res[a]["observed"][0] - res[a]["predicted"]) <= TOL for a in AGENTS}
    E["Q1_all_four"] = all(E[f"Q1_{a}"] for a in AGENTS)
    E["Q2_mean_abs_error_le_0.10"] = float(np.mean([abs(res[a]["observed"][0] - res[a]["predicted"]) for a in AGENTS])) <= 0.10
    res["mean_abs_error"] = round(float(np.mean([abs(res[a]["observed"][0] - res[a]["predicted"]) for a in AGENTS])), 4)
    res["endpoints"] = {k: bool(v) for k, v in E.items()}
    print(json.dumps(res["endpoints"], indent=1), "mean |error|", res["mean_abs_error"], flush=True)
    (ROOT / "results/t1/g12_composition_law.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
