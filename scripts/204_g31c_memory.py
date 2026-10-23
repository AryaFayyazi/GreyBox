#!/usr/bin/env python
"""G31c: how long does the state read through the API remember, and is the measurement noise that G27's filter
assumed right? Label-free (docs/prereg_G31c_memory.md). Data: G27b's step batteries (gpt-oss-20b, Qwen3.8,
Muse-Glimmer; pooled) and G27's Gemma-4 battery (reported; seen in an exploratory analysis).
Per agent, with no label: each answer angle minus its template x step mean (removes position and task);
  rho_h   = within-episode autocorrelation at lag h (h = 1..5), mean over the 8 questions
  ratio   = unbiased within-stratum variance of the angles / the nominal noise 1/(m + 1/2) that G27's filter used
  CPU only.  python scripts/204_g31c_memory.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY, REPORTED = ["gptoss20b", "qwen38", "museglimmer"], ["gemma4"]
B_N, SEED, H = 2000, 3150, 5


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def load(G27, actor):
    S = pd.read_parquet(G27.OUT / f"{actor}_steps.parquet").sort_values(["episode_id", "t"]).reset_index(drop=True)
    C = np.array([json.loads(c) for c in S.counts], float)
    Y, R = G27.angles(C)
    key = (S.template + "|" + S.t.astype(str)).values
    w = np.zeros(len(S))   # n_c / (n_c - 1): undoes the shrinkage of demeaning; 0 for single-step cells
    for u in np.unique(key):
        i = key == u
        Y[i] -= Y[i].mean(0)
        w[i] = i.sum() / (i.sum() - 1) if i.sum() > 1 else 0.0
    pos = {(e, t): i for i, (e, t) in enumerate(zip(S.episode_id, S.t))}
    lag = {h: np.array([pos.get((e, t - h), -1) for e, t in zip(S.episode_id, S.t)]) for h in range(1, H + 1)}
    return {"Y": Y, "R": R, "w": w, "lag": lag, "scen": S.scenario_id.values, "parsed": C[..., 1] > 0}


def rho(D, idx, h):
    a = idx[D["lag"][h][idx] >= 0]
    b = D["lag"][h][a]
    num = (D["Y"][a] * D["Y"][b]).mean(0)
    den = np.sqrt((D["Y"][a] ** 2).mean(0) * (D["Y"][b] ** 2).mean(0))
    return float(np.nanmean(num / den))


def ratio(D, idx):
    """unbiased within-cell variance of the angles over the nominal noise, mean over questions (multi-step cells)"""
    i = idx[D["w"][idx] > 0]
    return float(np.mean((D["w"][i, None] * D["Y"][i] ** 2).mean(0) / D["R"][i].mean(0)))


def stats(D, idx):
    out = {f"rho_lag{h}": rho(D, idx, h) for h in range(1, H + 1)}
    out["A2_rho1_minus_rho3"] = out["rho_lag1"] - out["rho_lag3"]
    out["A3_var_over_nominal_noise"] = ratio(D, idx)
    return out


def main():
    G27 = _mod("g27", "scripts/194_g27_state_tomography.py")
    data = {a: load(G27, a) for a in PRIMARY + REPORTED if (G27.OUT / f"{a}_steps.parquet").exists()}
    prim = [a for a in PRIMARY if a in data]
    res = {a: {k: round(v, 4) for k, v in stats(D, np.arange(len(D["Y"]))).items()} for a, D in data.items()}
    rng = np.random.default_rng(SEED)
    keys = ["rho_lag1", "A2_rho1_minus_rho3", "A3_var_over_nominal_noise"]
    B = {k: [] for k in keys}
    for _ in range(B_N):
        v = {k: [] for k in keys}
        for a in prim:
            D = data[a]
            us = np.unique(D["scen"])
            idx = np.concatenate([np.flatnonzero(D["scen"] == u) for u in rng.choice(us, len(us))])
            st = stats(D, idx)
            for k in keys:
                v[k].append(st[k])
        for k in keys:
            B[k].append(np.nanmean(v[k]))
    for k in keys:
        res[f"pooled_{k}"] = [round(float(np.mean([res[a][k] for a in prim])), 4),
                              round(float(np.nanpercentile(B[k], 2.5)), 4), round(float(np.nanpercentile(B[k], 97.5)), 4)]
    res["pooled_agents"] = prim
    res["pooled_rho_by_lag"] = [round(float(np.mean([res[a][f"rho_lag{h}"] for a in prim])), 4) for h in range(1, H + 1)]
    res["endpoints"] = {"A1_rho1_lower_gt_0": res["pooled_rho_lag1"][1] > 0,
                        "A2_decay_lower_gt_0": res["pooled_A2_rho1_minus_rho3"][1] > 0,
                        "A3_ratio_upper_lt_1": res["pooled_A3_var_over_nominal_noise"][2] < 1}
    (ROOT / "results/t1/g31c_memory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
