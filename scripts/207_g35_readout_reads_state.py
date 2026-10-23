#!/usr/bin/env python
"""G35: does the monitor come from the reconstructed state? (docs/prereg_G35_readout_reads_state.md)
On G15's held-out calls of the 4 open agents: the read-out s of the agent's own 8 answers (G15, nothing fit), the
agent's true decision state H (layer index 3), and the hidden state's violation coordinate P = the G14/G15 probe
(trained on development calls; a measuring instrument for the hidden state here, never part of the method).
  L1 (primary)  Spearman(s, P) within template AND within class (same label): the read-out follows the hidden state's
                violation coordinate beyond the label.
  L2            Spearman between the answer geometry (Euclidean, 8 angles) and the decision-state geometry (cosine)
                over pairs of calls with the same template and label.
  L3            the share of the read-out's within-template AUROC above 0.5 that is lost when the part of s that P
                predicts (rank regression within template) is removed.
Pooled: mean over the 4 agents; cluster bootstrap over scenarios within each agent (2000, seed 3500).  CPU only.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from graybox.stats import within_auroc  # noqa: E402

B_N, SEED, LAYER = 2000, 3500, 3


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def load():
    sys.argv = [sys.argv[0], "mode=heldout"]
    G15 = _mod("g15", "scripts/174_g15_training_free.py")
    out = {}
    for a in G15.OPEN:
        f = G15.OUT / f"{a}_{'fresh' if a == 'gemma4' else 'heldout'}.parquet"
        B = pd.read_parquet(f)
        s, _ = G15.score(B)
        P = G15.white_box(a, B)
        C = np.array([json.loads(c) for c in B.counts])
        Ang = G15.OR.theta(C[:, :, 0], C[:, :, 1])
        U, H = G15.M.states(a)
        key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
        ids = np.array([key[k] for k in zip(B.episode_id, B.event_id.astype(int))])
        out[a] = {"s": s, "P": P, "A": Ang, "H": H[ids, LAYER].astype(np.float32), "y": B.y.values.astype(int),
                  "tm": B.template.values, "sc": B.scenario_id.values}
    return out


def groups(D, idx):
    g = {}
    for i in idx:
        g.setdefault((D["tm"][i], D["y"][i]), []).append(i)
    return [np.array(v) for v in g.values() if len(v) >= 5]


def l1(D, idx):
    """pooled within template x class Spearman: rank-transform within each cell, then correlate"""
    a, b = [], []
    for g in groups(D, idx):
        if np.ptp(D["s"][g]) == 0 or np.ptp(D["P"][g]) == 0:
            continue
        a.append((rankdata(D["s"][g]) - 0.5) / len(g) - 0.5)
        b.append((rankdata(D["P"][g]) - 0.5) / len(g) - 0.5)
    if not a:
        return np.nan
    return float(np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1])


def l2(D, idx, cap=200):
    da, dh = [], []
    rng = np.random.default_rng(0)
    for g in groups(D, np.unique(idx)):
        g = g if len(g) <= cap else rng.choice(g, cap, replace=False)
        da.append(pdist(D["A"][g]))
        Hn = D["H"][g] / np.linalg.norm(D["H"][g], axis=1, keepdims=True)
        dh.append(pdist(Hn, "cosine"))
    return float(spearmanr(np.concatenate(da), np.concatenate(dh)).correlation)


def l3(D, idx):
    """share of the read-out's AUROC above 0.5 lost after removing the part of s that P predicts within template"""
    s, P, y, tm = D["s"][idx], D["P"][idx], D["y"][idx], D["tm"][idx]
    r = np.zeros(len(idx))
    for u in np.unique(tm):
        i = tm == u
        rs, rp = rankdata(s[i]), rankdata(P[i])
        if i.sum() < 3 or np.ptp(rp) == 0:
            r[i] = rs
            continue
        beta = np.polyfit(rp, rs, 1)
        r[i] = rs - np.polyval(beta, rp)
    a0, a1 = within_auroc(y, s, tm), within_auroc(y, r, tm)
    return float((a0 - a1) / (a0 - 0.5)) if a0 > 0.5 else np.nan


def main():
    data = load()
    res = {}
    for a, D in data.items():
        i = np.arange(len(D["y"]))
        res[a] = {"calls": int(len(i)), "L1": round(l1(D, i), 4), "L2": round(l2(D, i), 4), "L3": round(l3(D, i), 4),
                  "auroc_readout": round(float(within_auroc(D["y"], D["s"], D["tm"])), 4),
                  "auroc_probe": round(float(within_auroc(D["y"], D["P"], D["tm"])), 4),
                  "cells_template_x_class": len(groups(D, i))}
    rng = np.random.default_rng(SEED)
    B = {k: [] for k in ("L1", "L2", "L3")}
    for _ in range(B_N):
        v = {k: [] for k in B}
        for D in data.values():
            us = np.unique(D["sc"])
            idx = np.concatenate([np.flatnonzero(D["sc"] == u) for u in rng.choice(us, len(us))])
            v["L1"].append(l1(D, idx)); v["L3"].append(l3(D, idx))
            if _ % 10 == 0:
                v["L2"].append(l2(D, idx))
        for k in B:
            if v[k]:
                B[k].append(np.nanmean(v[k]))
    for k in B:
        res[f"{k}_pooled"] = [round(float(np.mean([res[a][k] for a in data])), 4),
                              round(float(np.nanpercentile(B[k], 2.5)), 4), round(float(np.nanpercentile(B[k], 97.5)), 4)]
    res["note"] = "L2's bootstrap uses every 10th replicate (200) to bound the cost"
    res["endpoints"] = {"L1_lower_gt_0": res["L1_pooled"][1] > 0, "L2_lower_gt_0": res["L2_pooled"][1] > 0,
                        "L3_lower_gt_0": res["L3_pooled"][1] > 0}
    (ROOT / "results/t1/g35_readout_reads_state.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
