#!/usr/bin/env python
"""G30: confirm, on three more agents, that the agent's answers before a call rank that call's violation beyond its
position, and track its true state within a step (docs/prereg_G30_precall_within_step.md). Data: G27b's step batteries
(gpt-oss-20b, Qwen3.8, Muse-Glimmer; scripts/197). Scores (G27's, nothing fit): the raw pre-call read-out, the signed
sum of the step's 8 answer angles. Strata: template x tool x step index. Truth: the stored decision states (layer
index 3), same-step pairs only. CPU only.  python scripts/200_g30_precall_within_step.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from graybox.stats import within_auroc  # noqa: E402

AGENTS = ["gptoss20b", "qwen38", "museglimmer"]
B_N, SEED, LAYER = 2000, 3000, 3


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def load(G27, M171, actor):
    S = pd.read_parquet(G27.OUT / f"{actor}_steps.parquet")
    C = np.array([json.loads(c) for c in S.counts], float)
    Y, _ = G27.angles(C)
    sg = np.array([q[1] for q in G27.QUESTIONS])
    s = Y @ sg
    y = S.y.values.astype(int)
    strata = np.array([f"{a}|{b}|{t}" for a, b, t in zip(S.template, S.tool, S.t)])
    U, H = M171.states(actor)
    key = {(e, int(v)): i for i, (e, v) in enumerate(zip(U.episode_id, U.event_id))}
    idx = np.array([key.get((e, int(v)), -1) for e, v in zip(S.episode_id, S.event_id)])
    ok = (idx >= 0) & np.array([bool(U.ok.values[j]) if j >= 0 else False for j in idx])
    return {"s": s, "y": y, "strata": strata, "scen": S.scenario_id.values, "Y": Y, "t": S.t.values, "ok": ok,
            "H": H[idx[ok], LAYER].astype(np.float32) if ok.sum() > 2 else None,
            "parsed_share": float(C[:, :, 1].sum() / (C.shape[0] * C.shape[1] * 4))}


def same_step_rsa(D, rows=None):
    ok = np.flatnonzero(D["ok"])
    if rows is not None:
        ok = np.intersect1d(ok, rows)
    if len(ok) < 10:
        return np.nan
    hi = {j: i for i, j in enumerate(np.flatnonzero(D["ok"]))}
    Hs = D["H"][[hi[j] for j in ok]]
    same = pdist(D["t"][ok].astype(float)[:, None]) == 0
    return spearmanr(pdist(D["Y"][ok])[same], pdist(Hs)[same]).correlation


def main():
    G27 = _mod("g27", "scripts/194_g27_state_tomography.py")
    M171 = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    data = {a: load(G27, M171, a) for a in AGENTS if (G27.OUT / f"{a}_steps.parquet").exists()}
    rng = np.random.default_rng(SEED)
    res = {}
    for a, D in data.items():
        both = [k for k in np.unique(D["strata"]) if 0 < D["y"][D["strata"] == k].sum() < (D["strata"] == k).sum()]
        res[a] = {"steps": int(len(D["y"])), "parsed_share": round(D["parsed_share"], 4),
                  "within_template_tool_step": round(float(within_auroc(D["y"], D["s"], D["strata"])), 4),
                  "strata_with_both_classes": len(both), "violations_in_them": int(sum(D["y"][D["strata"] == k].sum() for k in both)),
                  "same_step_rsa_with_truth": round(float(same_step_rsa(D)), 4), "steps_with_true_state": int(D["ok"].sum())}
    a1, a2 = [], []
    for _ in range(B_N):
        v1, v2 = [], []
        for D in data.values():
            us = np.unique(D["scen"])
            idx = np.concatenate([np.flatnonzero(D["scen"] == u) for u in rng.choice(us, len(us))])
            v1.append(within_auroc(D["y"][idx], D["s"][idx], D["strata"][idx]))
            v2.append(same_step_rsa(D, np.unique(idx)))
        a1.append(np.nanmean(v1)); a2.append(np.nanmean(v2))
    p1 = float(np.mean([res[a]["within_template_tool_step"] for a in data]))
    p2 = float(np.nanmean([res[a]["same_step_rsa_with_truth"] for a in data]))
    res["W1_pooled_within_step_auroc"] = [round(p1, 4), round(float(np.nanpercentile(a1, 2.5)), 4), round(float(np.nanpercentile(a1, 97.5)), 4)]
    res["W2_pooled_same_step_rsa_with_truth"] = [round(p2, 4), round(float(np.nanpercentile(a2, 2.5)), 4), round(float(np.nanpercentile(a2, 97.5)), 4)]
    res["endpoints"] = {"W1_lower_gt_0.5": res["W1_pooled_within_step_auroc"][1] > 0.5, "W2_lower_gt_0": res["W2_pooled_same_step_rsa_with_truth"][1] > 0}
    (ROOT / "results/t1/g30_precall_within_step.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
