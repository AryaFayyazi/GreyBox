#!/usr/bin/env python
"""G31b: what the state read before the FIRST call says about a violation that comes LATER (early warning), and how
fast the pre-call information decays with the lag between the answers and the call (docs/prereg_G31b_horizon.md).
Data: G27b's step batteries (gpt-oss-20b, Qwen3.8, Muse-Glimmer; primary) and G27's Gemma-4 battery (reported only).
Score: G27's raw read-out, the signed sum of a step's 8 answer angles (nothing fit; no label used). CPU only.
  E1  episodes whose first call is not a violation; score: the read-out before the first call; label: the episode
      violates at a later step; strata: template x first tool.
  H1  calls at step t >= 1; AUROC of the read-out before call t minus that of the read-out before call t-1;
      strata: template x tool x step.
  python scripts/201_g31b_horizon.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from graybox.stats import within_auroc  # noqa: E402

PRIMARY, REPORTED = ["gptoss20b", "qwen38", "museglimmer"], ["gemma4"]
B_N, SEED, H_MAX = 2000, 3100, 3


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def load(G27, actor):
    import pandas as pd
    S = pd.read_parquet(G27.OUT / f"{actor}_steps.parquet").sort_values(["episode_id", "t"]).reset_index(drop=True)
    C = np.array([json.loads(c) for c in S.counts], float)
    Y, _ = G27.angles(C)
    s = Y @ np.array([q[1] for q in G27.QUESTIONS])
    pos = {(e, t): i for i, (e, t) in enumerate(zip(S.episode_id, S.t))}
    lag = {h: np.array([pos.get((e, t - h), -1) for e, t in zip(S.episode_id, S.t)]) for h in range(H_MAX + 1)}
    return {"s": s, "y": S.y.values.astype(int), "t": S.t.values, "ev": S.episode_violates.values.astype(int),
            "ep": S.episode_id.values, "tool": S.tool.values, "template": S.template.values, "scen": S.scenario_id.values, "lag": lag}


def _all(D):
    return np.arange(len(D["y"]))


def e1(D, idx):
    """early warning: first-step read-out vs 'violates later', among episodes whose first call is safe.
    idx: row indices (repeats allowed, for the bootstrap)"""
    r = idx[(D["t"][idx] == 0) & (D["y"][idx] == 0)]
    st = np.array([f"{a}|{b}" for a, b in zip(D["template"][r], D["tool"][r])])
    return within_auroc(D["ev"][r], D["s"][r], st), r


def lag_auc(D, idx, h, t_min):
    """AUROC of the read-out before call t-h for the label of call t, on calls with t >= t_min"""
    r = idx[(D["t"][idx] >= t_min) & (D["lag"][h][idx] >= 0)]
    st = np.array([f"{a}|{b}|{t}" for a, b, t in zip(D["template"][r], D["tool"][r], D["t"][r])])
    return within_auroc(D["y"][r], D["s"][D["lag"][h][r]], st)


def h1(D, idx):
    return lag_auc(D, idx, 0, 1) - lag_auc(D, idx, 1, 1)


def dprime2(a):
    """squared separation of two equal-variance Gaussians with this AUROC: additive over independent components"""
    a = float(np.clip(a, 0.5, 0.999))
    return 2 * norm.ppf(a) ** 2


def boot_rows(D, rng):
    us = np.unique(D["scen"])
    return np.concatenate([np.flatnonzero(D["scen"] == u) for u in rng.choice(us, len(us))])


def stats(D):
    a = _all(D)
    a_e1, r = e1(D, a)
    out = {"E1_early_warning_auroc": round(float(a_e1), 4), "E1_episodes": int(len(r)),
           "E1_episodes_violating_later": int(D["ev"][r].sum())}
    fv = {e: t for e, t, y in zip(D["ep"], D["t"], D["y"]) if y}
    late = [fv[e] for e in D["ep"][r] if e in fv]
    out["E1_median_step_of_later_violation"] = float(np.median(late)) if late else None
    for h in range(H_MAX + 1):
        out[f"auroc_lag{h}_own_set"] = round(float(lag_auc(D, a, h, h)), 4)
        out[f"auroc_lag{h}_common_set_t_ge_{H_MAX}"] = round(float(lag_auc(D, a, h, H_MAX)), 4)
    out["H1_lag0_minus_lag1_t_ge_1"] = round(float(h1(D, a)), 4)
    out["persistent_share_dprime2_lag1_over_lag0"] = round(dprime2(lag_auc(D, a, 1, 1)) / max(dprime2(lag_auc(D, a, 0, 1)), 1e-9), 4)
    return out


def main():
    G27 = _mod("g27", "scripts/194_g27_state_tomography.py")
    data = {a: load(G27, a) for a in PRIMARY + REPORTED if (G27.OUT / f"{a}_steps.parquet").exists()}
    prim = [a for a in PRIMARY if a in data]
    res = {a: stats(D) for a, D in data.items()}
    rng = np.random.default_rng(SEED)
    be, bh = [], []
    for _ in range(B_N):
        ve, vh = [], []
        for a in prim:
            idx = boot_rows(data[a], rng)
            ve.append(e1(data[a], idx)[0]); vh.append(h1(data[a], idx))
        be.append(np.nanmean(ve)); bh.append(np.nanmean(vh))
    pe = float(np.mean([res[a]["E1_early_warning_auroc"] for a in prim]))
    ph = float(np.mean([res[a]["H1_lag0_minus_lag1_t_ge_1"] for a in prim]))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    res["pooled_agents"] = prim
    res["E1_pooled_early_warning_auroc"] = [round(pe, 4)] + ci(be)
    res["H1_pooled_lag0_minus_lag1"] = [round(ph, 4)] + ci(bh)
    res["endpoints"] = {"E1_lower_gt_0.5": res["E1_pooled_early_warning_auroc"][1] > 0.5,
                        "H1_lower_gt_0": res["H1_pooled_lag0_minus_lag1"][1] > 0}
    (ROOT / "results/t1/g31b_horizon.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
