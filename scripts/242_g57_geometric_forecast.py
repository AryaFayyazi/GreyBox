#!/usr/bin/env python
"""G57: forecasting by the reconstructed geometry alone (docs/prereg_G57_geometric_forecast.md). Before each unit's call,
the agent's 8 sampled next actions (G46c, Claude Haiku 5.5, exact contexts; G48, GPT-6-luna) are each placed in the
Fisher-Rao coordinates of their 8 at-call answers; the geometric forecast is the mean over the 8 draws of the draw's
distance from the agent's typical state (G56's centre: the median of all its drawn calls); a draw that is not a call
scores 0. No label, no question sign.
  -> results/t1/g57_geometric_forecast.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED = 2000, 5700


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def theta(counts):
    c = np.asarray(counts, float)
    return 2 * np.arcsin(np.sqrt((c[:, 0] + 3 / 8) / (c[:, 1] + 3 / 4)))


def analyse(units, own_file):
    from graybox.stats import within_auroc
    R = {(d["instance_id"], int(d["seed"]), int(d["t"])): d for d in (json.loads(x) for x in open(own_file))}
    allth = [theta(rd["counts"]) for d in R.values() for rd in d["reads"].values() if np.asarray(rd["counts"])[:, 1].min() > 0]
    centre = np.median(np.stack(allth), 0)
    ks = [(a, int(b), int(c)) for a, b, c in zip(units.instance_id, units.seed, units.t)]
    m = np.array([k in R for k in ks])
    U, ks = units[m].reset_index(drop=True), [k for k, x in zip(ks, m) if x]

    def fc(d):
        v = []
        for dr in d["draws"]:
            if dr and dr in d["reads"] and np.asarray(d["reads"][dr]["counts"])[:, 1].min() > 0:
                v.append(float(np.linalg.norm(theta(d["reads"][dr]["counts"]) - centre)))
            else:
                v.append(0.0)
        return float(np.mean(v)) if v else 0.0
    F = np.array([fc(R[k]) for k in ks])
    y, cl = U.y.values, U.scenario_id.values
    st3 = (U.stratum + "|" + U.position.astype(str)).values
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(cl)
    B = [within_auroc(y[i], F[i], st3[i]) for i in (np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))]) for _ in range(B_N))]
    return {"units": int(len(y)), "violations": int(y.sum()),
            "G1_geometric_forecast_within_position": [round(float(within_auroc(y, F, st3)), 4),
                                                      round(float(np.nanpercentile(B, 2.5)), 4), round(float(np.nanpercentile(B, 97.5)), 4)]}


def main():
    G46C = _mod("g46c", "scripts/232_g46c_exact_context.py")
    G48 = _mod("g48", "scripts/229_g48_gpt_rollout.py")
    res = {"claude": analyse(G46C.units()[0], G46C.OWN), "gpt": analyse(G48.units()[0], G48.OWN)}
    res["endpoints"] = {f"G1_{k}_lower_gt_0.5": res[k]["G1_geometric_forecast_within_position"][1] > 0.5 for k in ("claude", "gpt")}
    (ROOT / "results/t1/g57_geometric_forecast.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
