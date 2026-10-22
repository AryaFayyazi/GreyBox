#!/usr/bin/env python
"""G28: does the estimated state track the TRUE hidden state? (docs/prereg_G28_tomography_vs_truth.md)
On the G27 steps of Gemma-4 whose true decision state is stored (G14/T17 decision states, layer index 3, the layer of
G15's white-box probe), compare the geometry (pairwise Euclidean distances, Spearman over all pairs; no fitted map) of
  filtered  E[z_t | y_<=t]   (G27's Kalman estimate from the agent's answers),
  raw       y_t              (the step's own answer angles),
  forecast  E[z_t | y_<t]    (made before the step),
with the geometry of the true states. The true states are used only to validate; the estimate never sees them.
python scripts/195_g28_tomography_vs_truth.py   (CPU; after G27's battery)
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
LAYER, B_N, SEED = 3, 1000, 2800


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def estimates():
    """G27's cross-fit Kalman filter, as in scripts/194 stage_analyze (same folds and seed)."""
    g = _mod("g27", "scripts/194_g27_state_tomography.py")
    S = pd.read_parquet(g.OUT / "gemma4_steps.parquet")
    C = np.array([json.loads(c) for c in S.counts], float)
    Y, R = g.angles(C)
    ep, scen = S.episode_id.values, S.scenario_id.values
    us = np.unique(scen)
    rng = np.random.default_rng(2701)
    fold = dict(zip(us, rng.permutation(len(us)) % 5))
    fc, fl = np.zeros_like(Y), np.zeros_like(Y)
    for f in range(5):
        tr = [e for e in np.unique(ep) if fold[scen[ep == e][0]] != f]
        M = g.sysid([(Y[ep == e], R[ep == e]) for e in tr])
        for e in np.unique(ep):
            i = np.flatnonzero(ep == e)
            if fold[scen[i[0]]] == f:
                a, b, _ = g.kalman(Y[i], R[i], M)
                fc[i], fl[i] = a, b
    return S, Y, fc, fl


def rsa(A, B):
    return spearmanr(pdist(A), pdist(B)).correlation


def main():
    M = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    S, Y, fc, fl = estimates()
    U, H = M.states("gemma4")
    key = {(e, int(v)): i for i, (e, v) in enumerate(zip(U.episode_id, U.event_id))}
    idx = np.array([key.get((e, int(v)), -1) for e, v in zip(S.episode_id, S.event_id)])
    ok = idx >= 0
    ok &= np.array([bool(U.ok.values[j]) if j >= 0 else False for j in idx])
    later = ok & (S.t.values > 0)
    Ht = H[idx[ok], LAYER].astype(np.float32)
    scen = S.scenario_id.values[ok]
    est = {"filtered": fl[ok], "raw": Y[ok], "forecast": fc[ok]}
    pt = {k: rsa(v, Ht) for k, v in est.items()}
    rng = np.random.default_rng(SEED)
    us = np.unique(scen)
    bd, bfc = [], []
    for _ in range(B_N):
        pick = rng.choice(us, len(us))
        ii = np.concatenate([np.flatnonzero(scen == u) for u in pick])
        ii = np.unique(ii)                                  # distances need distinct units; a cluster bootstrap over scenarios
        bd.append(rsa(est["filtered"][ii], Ht[ii]) - rsa(est["raw"][ii], Ht[ii]))
        bfc.append(rsa(est["forecast"][ii], Ht[ii]))
    res = {"steps_with_true_state": int(ok.sum()), "layer_index": LAYER, "rsa_with_true_state": {k: round(float(v), 4) for k, v in pt.items()},
           "T1_filtered_minus_raw": [round(float(pt["filtered"] - pt["raw"]), 4), round(float(np.nanpercentile(bd, 2.5)), 4), round(float(np.nanpercentile(bd, 97.5)), 4)],
           "T2_forecast_ci": [round(float(np.nanpercentile(bfc, 2.5)), 4), round(float(np.nanpercentile(bfc, 97.5)), 4)],
           "reported_steps_after_first": {k: round(float(rsa(v[later[ok]], Ht[later[ok]])), 4) for k, v in est.items()}}
    res["endpoints"] = {"T1_lower_gt_0": res["T1_filtered_minus_raw"][1] > 0, "T2_lower_gt_0": res["T2_forecast_ci"][0] > 0}
    (ROOT / "results/t1/g28_tomography_vs_truth.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
