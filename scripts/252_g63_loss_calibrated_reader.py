#!/usr/bin/env python
"""G63: the text monitor's loss predicted from a reader recalibrated on a few labelled draws from OTHER scenarios
(docs/prereg_G63_loss_calibrated_reader.md). G62 failed because GPT-6-luna's probability for each drawn call overstates
violations (calibration -0.195). Here the scenarios are split in two folds (seed 6300); in each fold, Platt scaling
(logistic regression on logit r, C=1e6) is fit on 200 distinct drawn calls sampled (seed 6301) from the OTHER fold, with
CAPE's checker label of each drawn call, and applied to this fold's reads. No label from the scored fold is used, and no
label of any executed call. Then G62's quantities. Pairs: Claude Haiku 5.5 (G46c units) x G52's judge, x G55's judge.
  -> results/t1/g63_loss_calibrated_reader.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED, BAND, N_CAL = 2000, 6302, 0.02, 200


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    from sklearn.linear_model import LogisticRegression
    G62 = _mod("g62", "scripts/251_g62_loss_no_labels.py")
    T = _mod("t249", "scripts/249_forecasters_both_strata_exploratory.py")
    G46C, key = T.G46C, T.key
    reads = T.G46.G45.T11.READS
    U, _ = G46C.units()
    rd, own = T.load(T.G58.JF), T.load(G46C.OWN)
    sid = {key(a, b, c): s for a, b, c, s in zip(U.instance_id, U.seed, U.t, U.scenario_id)}
    scen = np.array(sorted(set(U.scenario_id)))
    rng = np.random.default_rng(6300)
    fold = dict(zip(scen, rng.permutation(len(scen)) % 2))
    # distinct drawn calls with the reader's probability and the checker's label, per fold
    pool = {0: [], 1: []}
    for k, d in rd.items():
        if k not in own or k not in sid:
            continue
        for x, p in d["p"].items():
            if p is not None and x in own[k]["reads"] and json.loads(x)[0] not in reads:
                pool[fold[sid[k]]].append((float(p), int(own[k]["reads"][x]["unsafe"])))
    lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))  # noqa: E731
    cal = {}
    rs = np.random.default_rng(6301)
    for f in (0, 1):
        src = pool[1 - f]
        idx = rs.choice(len(src), min(N_CAL, len(src)), replace=False)
        P, L = np.array([src[i][0] for i in idx]), np.array([src[i][1] for i in idx])
        m = LogisticRegression(C=1e6).fit(lg(P)[:, None], L)
        cal[f] = m
    D = {}
    for pair, jf in (("claude_self", T.G52.JF), ("claude_gpt", T.G55.JF)):
        jd = T.load(jf)
        rows = []
        for a, b, c, y, s in zip(U.instance_id, U.seed, U.t, U.y, U.scenario_id):
            k = key(a, b, c)
            if k not in rd or k not in jd or jd[k].get("p") is None:
                continue
            r = [rd[k]["p"][x] for x in rd[k]["draws"] if x and json.loads(x)[0] not in reads and rd[k]["p"].get(x) is not None]
            if len(r) >= 2:
                rc = cal[fold[s]].predict_proba(lg(np.array(r))[:, None])[:, 1]
                rows.append((float(y), float(np.mean(rc)), float(np.var(rc, ddof=1) / len(rc)), float(jd[k]["p"]), s))
        y, f_, v, q, cl = (np.array(x) for x in zip(*rows))
        D[pair] = {"y": y, "f": f_, "v": v, "q": q, "cl": cl}
    rng = np.random.default_rng(BOOT_SEED)
    boots = {p: [] for p in D}
    for _ in range(B_N):
        for p, d in D.items():
            us = np.unique(d["cl"])
            i = np.concatenate([np.flatnonzero(d["cl"] == u) for u in rng.choice(us, len(us))])
            boots[p].append(G62.quantities(d, i))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    names = ("observed_excess", "predicted_excess", "observed_minus_predicted", "calibration_y_minus_f")
    res = {"labelled_draws_per_fold": N_CAL, "pairs": {}}
    for p, d in D.items():
        full, B = G62.quantities(d, np.arange(len(d["y"]))), np.array(boots[p])
        res["pairs"][p] = {"units": int(len(d["y"])), "violations": int(d["y"].sum())}
        res["pairs"][p].update({nm: [round(float(full[j]), 4)] + ci(B[:, j]) for j, nm in enumerate(names)})
    res["endpoints"] = {f"L1_{p}_inside_pm0.02": (lambda v: v[1] >= -BAND and v[2] <= BAND)(res["pairs"][p]["observed_minus_predicted"]) for p in D}
    (ROOT / "results/t1/g63_loss_calibrated_reader.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
