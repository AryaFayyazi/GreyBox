#!/usr/bin/env python
"""Battery scaling (development split): hidden-state recovery as a function of the number of questions m.

For each model with the 512-item pool (psi_v3.npz) and hidden states (hidden.npz):
  random-m     mean over 5 random item subsets of size m
  designed-m   black-box design: column-pivoted QR on the θ answer matrix of the DESIGN contexts (no hidden states,
               no labels): picks the items least predictable from those already chosen
  designed-x   items designed on a different model, applied here (does a design transfer across models?)
Target: variance-weighted R² of the top-64 residual PCs at mid depth (and at the last saved layer), ridge, 5-fold CV.
Baseline: text embedding (MiniLM), and text + Ψ.
Design contexts and evaluation folds are disjoint: the design uses the first half of dev contexts (answers only);
R² is cross-validated on the second half.

python scripts/16_scaling.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr

sys.path.insert(0, str(Path(__file__).resolve().parent))
an = importlib.import_module("13_l2_analysis")
L2 = an.L2
RES = an.RES
MS = (8, 16, 32, 64, 128, 256, 512)


def theta_items(z):
    """One column per item (θ only), so item selection and subset sizes count questions, not features."""
    q = (z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6)
    return 2 * np.arcsin(np.sqrt(q))


def design(theta_design, m):
    X = theta_design - theta_design.mean(0)
    _, _, piv = qr(X, pivoting=True, mode="economic")
    return piv[:m]


def main():
    C = pd.read_parquet(L2 / "contexts.parquet")
    dev = np.flatnonzero((C.split == "dev").values)
    rng = np.random.default_rng(0)
    perm = rng.permutation(dev)
    des, ev = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    E = an.text_embeddings(C.text)
    models = [d.name for d in sorted(L2.iterdir()) if (d / "psi_v3.npz").exists() and (d / "hidden.npz").exists()]
    TH = {m: theta_items(np.load(L2 / m / "psi_v3.npz")) for m in models}
    designs = {m: design(TH[m][des], 512) for m in models}
    res = {"models": models, "n_design": len(des), "n_eval": len(ev), "rows": {}}
    for m in models:
        Z = np.load(L2 / m / "hidden.npz")
        out = {}
        for j in (len(Z["layers"]) // 2, len(Z["layers"]) - 1):
            Hl = Z["h"][ev, j].astype(np.float32)
            mu = Hl.mean(0)
            _, S, Vt = np.linalg.svd(Hl - mu, full_matrices=False)
            Y, w = (Hl - mu) @ Vt[:64].T, S[:64] ** 2 / (S ** 2).sum()
            score = lambda X: float((an.cv_r2(X, Y) * w).sum() / w.sum())  # noqa: E731
            row = {"text": round(score(E[ev]), 4)}
            for mm in MS:
                rnd = [score(TH[m][ev][:, rng.choice(512, mm, replace=False)]) for _ in range(5 if mm < 512 else 1)]
                row[f"random_{mm}"] = round(float(np.mean(rnd)), 4)
                row[f"designed_{mm}"] = round(score(TH[m][ev][:, designs[m][:mm]]), 4)
                others = [o for o in models if o != m]
                if others:
                    row[f"designed_on_other_{mm}"] = round(float(np.mean([score(TH[m][ev][:, designs[o][:mm]])
                                                                           for o in others])), 4)
                row[f"text+designed_{mm}"] = round(score(np.hstack([E[ev], TH[m][ev][:, designs[m][:mm]]])), 4)
            out[f"L{int(Z['layers'][j])}"] = row
            print(m, f"L{int(Z['layers'][j])}", row, flush=True)
        res["rows"][m] = out
        res.setdefault("designed_items", {})[m] = [str(x) for x in np.load(L2 / m / "psi_v3.npz")["items"][designs[m][:32]]]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "scaling_dev.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
