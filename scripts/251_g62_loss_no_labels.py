#!/usr/bin/env python
"""G62: the text monitor's loss predicted with NO label at all (docs/prereg_G62_loss_no_labels.md). As G60, but each
sampled action's violation probability r comes from a reader, GPT-6-luna reasoning over the drawn call (G58's reads),
instead of CAPE's checker. f = mean r over the unit's non-read draws (k of them), s2 = their sample variance (ddof 1).
predicted = mean (f - q)^2 - 2 mean s2/k (no label); observed = mean (Y - q)^2 - mean (Y - f)^2 (Y only to score it).
Pairs: Claude Haiku 5.5 (G46c units) x its own text judge (G52), x GPT-6-luna's text judge (G55). Units with k >= 2.
  -> results/t1/g62_loss_no_labels.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED, BAND = 2000, 6200, 0.02


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def quantities(d, i):
    y, f, v, q = d["y"][i], d["f"][i], d["v"][i], d["q"][i]
    obs = np.mean((y - q) ** 2) - np.mean((y - f) ** 2)
    pred = np.mean((f - q) ** 2) - 2 * np.mean(v)
    return obs, pred, obs - pred, np.mean(y) - np.mean(f)


def main():
    T = _mod("t249", "scripts/249_forecasters_both_strata_exploratory.py")
    G46C, key = T.G46C, T.key
    reads = T.G46.G45.T11.READS
    U, _ = G46C.units()
    rd = T.load(T.G58.JF)
    D = {}
    for pair, jf in (("claude_self", T.G52.JF), ("claude_gpt", T.G55.JF)):
        jd = T.load(jf)
        rows = []
        for a, b, c, y, sid in zip(U.instance_id, U.seed, U.t, U.y, U.scenario_id):
            k = key(a, b, c)
            if k not in rd or k not in jd or jd[k].get("p") is None:
                continue
            r = [rd[k]["p"][x] for x in rd[k]["draws"] if x and json.loads(x)[0] not in reads and rd[k]["p"].get(x) is not None]
            if len(r) >= 2:
                rows.append((float(y), float(np.mean(r)), float(np.var(r, ddof=1) / len(r)), float(jd[k]["p"]), sid))
        y, f, v, q, cl = (np.array(x) for x in zip(*rows))
        D[pair] = {"y": y, "f": f, "v": v, "q": q, "cl": cl}
    rng = np.random.default_rng(BOOT_SEED)
    boots = {p: [] for p in D}
    for _ in range(B_N):
        for p, d in D.items():
            us = np.unique(d["cl"])
            i = np.concatenate([np.flatnonzero(d["cl"] == u) for u in rng.choice(us, len(us))])
            boots[p].append(quantities(d, i))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    names = ("observed_excess", "predicted_excess", "observed_minus_predicted", "calibration_y_minus_f")
    res = {"pairs": {}}
    for p, d in D.items():
        full, B = quantities(d, np.arange(len(d["y"]))), np.array(boots[p])
        res["pairs"][p] = {"units": int(len(d["y"])), "violations": int(d["y"].sum())}
        res["pairs"][p].update({nm: [round(float(full[j]), 4)] + ci(B[:, j]) for j, nm in enumerate(names)})
    res["endpoints"] = {f"L1_{p}_inside_pm0.02": (lambda v: v[1] >= -BAND and v[2] <= BAND)(res["pairs"][p]["observed_minus_predicted"]) for p in D}
    (ROOT / "results/t1/g62_loss_no_labels.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
