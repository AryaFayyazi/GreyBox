#!/usr/bin/env python
"""G56: monitoring by the reconstructed geometry alone (docs/prereg_G56_geometric_monitor.md). A call's observable state
is its 8 answer angles, exact Fisher-Rao coordinates (Theorem 1). The geometric monitor scores a call by its Fisher-Rao
distance from the agent's typical state, the coordinate-wise median of the agent's own unlabelled calls. It uses no label,
no question sign and no read-out design; flipping a question's wording (theta -> pi - theta) leaves it unchanged.
Data (not analysed this way before this lock): the drawn calls of G46c (Claude Haiku 5.5, exact contexts) and G48
(GPT-6-luna, six templates): each distinct drawn call with its 8 angles (4 answers each) and CAPE's checker label.
  stage=analyze -> results/t1/g56_geometric_monitor.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED = 2000, 5600


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def calls(own_file, units, tmpl):
    """one row per distinct drawn call: its angles, its label, its scenario and stratum (template|tool), its signed read-out"""
    from graybox.selfaudit import STATE_QUESTIONS
    sg = np.array([s for _, s in STATE_QUESTIONS], float)
    sid = {(a, int(b)): s for a, b, s in zip(units.instance_id, units.seed, units.scenario_id)}
    rows = []
    for d in (json.loads(x) for x in open(own_file)):
        for key, rd in d["reads"].items():
            name = json.loads(key)[0]
            c = np.asarray(rd["counts"], float)
            if c[:, 1].min() == 0:
                continue
            th = 2 * np.arcsin(np.sqrt((c[:, 0] + 3 / 8) / (c[:, 1] + 3 / 4)))
            rows.append({"scenario_id": sid.get((d["instance_id"], int(d["seed"])), d["instance_id"]), "stratum": f"{tmpl(d)}|{name}",
                         "y": int(rd["unsafe"]), "theta": th, "readout": float((sg * th).mean())})
    D = pd.DataFrame(rows)
    T = np.stack(D.theta.values)
    centre = np.median(T, 0)                                          # label-free: the agent's typical state
    D["geo"] = np.sqrt(((T - centre) ** 2).sum(1))                     # Fisher-Rao distance (Theorem 1)
    g = D.groupby("stratum").y.agg(["sum", "count"])
    keep = set(g.index[(g["sum"] > 0) & (g["sum"] < g["count"])])
    return D[D.stratum.isin(keep)].reset_index(drop=True)


def stage_analyze():
    from graybox.stats import within_auroc
    G46C = _mod("g46c", "scripts/232_g46c_exact_context.py")
    G48 = _mod("g48", "scripts/229_g48_gpt_rollout.py")
    Uc, _ = G46C.units()
    Ug, _ = G48.units()
    tm_c = {(a, int(b)): s.split("|")[0] for a, b, s in zip(Uc.instance_id, Uc.seed, Uc.stratum)}
    tm_g = {(a, int(b)): s.split("|")[0] for a, b, s in zip(Ug.instance_id, Ug.seed, Ug.stratum)}
    sets = {"claude": calls(G46C.OWN, Uc, lambda d: tm_c.get((d["instance_id"], int(d["seed"])), "cons_service_restart")),
            "gpt": calls(G48.OWN, Ug, lambda d: tm_g.get((d["instance_id"], int(d["seed"])), "?"))}
    rng = np.random.default_rng(BOOT_SEED)
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    res = {}
    for k, D in sets.items():
        y, st, cl = D.y.values, D.stratum.values, D.scenario_id.values
        us = np.unique(cl)
        Bg, Bd = [], []
        for _ in range(B_N):
            i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
            a, b = within_auroc(y[i], D.geo.values[i], st[i]), within_auroc(y[i], D.readout.values[i], st[i])
            Bg.append(a); Bd.append(a - b)
        g, r = within_auroc(y, D.geo.values, st), within_auroc(y, D.readout.values, st)
        res[k] = {"calls": int(len(D)), "violations": int(y.sum()), "strata": int(len(set(st))),
                  "M1_geometric_within_stratum": [round(float(g), 4)] + ci(Bg),
                  "signed_readout_within_stratum": round(float(r), 4),
                  "M2_geometric_minus_readout": [round(float(g - r), 4)] + ci(Bd)}
    res["endpoints"] = {f"M1_{k}_lower_gt_0.5": res[k]["M1_geometric_within_stratum"][1] > 0.5 for k in sets}
    (ROOT / "results/t1/g56_geometric_monitor.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    stage_analyze()
