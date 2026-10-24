#!/usr/bin/env python
"""G46b (addendum to G46, docs/prereg_G46b_policy_gap.md): does the proposition "the forecast needs the agent" predict,
in size, how much another model loses? On G46's units, with r known (CAPE's checker on each drawn call):
  f_own = share of unsafe calls among the agent's non-read draws (of K = 8); f_oth = the same for Gemma-4's draws in its seat.
  observed D = mean (Y - f_oth)^2 - mean (Y - f_own)^2                      (Brier excess of the other reader)
  predicted D = mean (f_own - f_oth)^2 - 2 mean f_own (1 - f_own) / (k_own - 1)  (k_own: non-read own draws)
In expectation observed = predicted when the agent's draws and its actual action come from the same policy at the same
state and the other draws are independent of Y given the transcript.
  stage=analyze -> results/t1/g46b_policy_gap.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, SEED = 2000, 4610


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G46 = _mod("g46", "scripts/226_g46_rollout_forecast.py")


def shares(rec):
    """the unit is an executed NON-READ call, so Y is conditioned on the action being non-read: the policy's violation rate
    is the share of unsafe calls among the draws that are non-read calls; returns (share, number of non-read draws)"""
    reads = G46.G45.T11.READS
    v = [rec["reads"][d]["unsafe"] for d in rec["draws"] if d and d in rec["reads"] and json.loads(d)[0] not in reads]
    return (float(np.mean(v)) if v else np.nan), len(v)


def main():
    U, _ = G46.units()
    key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
    own = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G46.OWN))}
    oth = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G46.OTHER))}
    ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    keep = [i for i, k in enumerate(ks) if k in own and k in oth and shares(own[k])[1] >= 2 and shares(oth[k])[1] >= 2]
    U = U.iloc[keep].reset_index(drop=True)
    ks = [ks[i] for i in keep]
    fo = np.array([shares(own[k])[0] for k in ks])
    ko = np.array([shares(own[k])[1] for k in ks])
    fx = np.array([shares(oth[k])[0] for k in ks])
    kx = np.array([shares(oth[k])[1] for k in ks])
    y = U.y.values.astype(float)
    cl = U.scenario_id.values

    def stats(i):
        obs = np.mean((y[i] - fx[i]) ** 2) - np.mean((y[i] - fo[i]) ** 2)
        pred = np.mean((fo[i] - fx[i]) ** 2) - 2 * np.mean(fo[i] * (1 - fo[i]) / (ko[i] - 1))
        gap = np.mean((fo[i] - fx[i]) ** 2) - np.mean(fo[i] * (1 - fo[i]) / (ko[i] - 1)) - np.mean(fx[i] * (1 - fx[i]) / (kx[i] - 1))
        return obs, pred, obs - pred, gap
    o, p, d, g = stats(np.arange(len(y)))
    rng = np.random.default_rng(SEED)
    us = np.unique(cl)
    B = np.array([stats(np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])) for _ in range(B_N)])
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    res = {"units": int(len(y)), "violations": int(y.sum()),
           "mean_f_own": round(float(fo.mean()), 4), "mean_f_other": round(float(fx.mean()), 4), "mean_y": round(float(y.mean()), 4),
           "B1_observed_excess": [round(float(o), 4)] + ci(B[:, 0]),
           "predicted_excess": [round(float(p), 4)] + ci(B[:, 1]),
           "B3_observed_minus_predicted": [round(float(d), 4)] + ci(B[:, 2]),
           "B2_policy_gap": [round(float(g), 4)] + ci(B[:, 3])}
    res["endpoints"] = {"B3_interval_inside_pm0.02": res["B3_observed_minus_predicted"][1] >= -0.02 and res["B3_observed_minus_predicted"][2] <= 0.02,
                        "B2_policy_gap_lower_gt_0": res["B2_policy_gap"][1] > 0}
    (ROOT / "results/t1/g46b_policy_gap.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
