#!/usr/bin/env python
"""Exploratory (asked by the user): is comparing cosine (hidden, text) with Euclidean (answers) fair? On G6's 1,000 fresh
contexts, per open model, Spearman with the hidden geometry under every combination of distance: hidden cosine or
Euclidean; answers Euclidean on angles (ours), cosine on angles, or Euclidean on raw P(yes); input text (all-MiniLM-L6-v2)
cosine or Euclidean. -> results/t1/metric_robustness_exploratory.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("g83m", ROOT / "scripts/282_g83_newkind_battery.py")
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)
g, g4, G6 = M.g, M.g4, M.G6


def main():
    Cc = g.contexts()
    dev = np.where(Cc.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    E = np.load(G6.OUT / "run/contexts_minilm.npy").astype(np.float64)
    D = lambda X, m: squareform(pdist(X, m))  # noqa: E731
    text = {"cosine": D(E, "cosine"), "euclidean": D(E, "euclidean")}
    out = {}
    for m in g.MODELS:
        run = G6.OUT / "run" / m
        Hm = np.load(run / "hidden_none.npz")["h"][:, g.select_layer(m, dev_sel)[0]].astype(np.float64)
        p = g4.p_of(np.load(run / "psi_none.npz", allow_pickle=True))
        th = g.theta(p)
        hid = {"cosine": D(Hm, "cosine"), "euclidean": D(Hm, "euclidean")}
        ans = {"angles_euclidean (ours)": D(th, "euclidean"), "angles_cosine": D(th, "cosine"), "raw_p_euclidean": D(p, "euclidean")}
        out[m] = {f"hidden_{hk}": {**{a: round(g.rho(A, H), 4) for a, A in ans.items()},
                                    **{f"text_{tk}": round(g.rho(T, H), 4) for tk, T in text.items()}} for hk, H in hid.items()}
        print(m, json.dumps(out[m]))
    mean = {hk: {k: round(float(np.mean([out[m][hk][k] for m in out])), 4) for k in next(iter(out.values()))[hk]} for hk in ("hidden_cosine", "hidden_euclidean")}
    (ROOT / "results/t1/metric_robustness_exploratory.json").write_text(json.dumps({"contexts": 1000, "mean": mean, "per_model": out}, indent=1))
    print(json.dumps(mean, indent=1))


if __name__ == "__main__":
    main()
