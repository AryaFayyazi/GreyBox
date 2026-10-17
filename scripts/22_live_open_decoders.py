#!/usr/bin/env python
"""Per-open-model decoders for the Live truth overlay: θ(Ψ on the live battery) -> true hidden state (top PCs).

For each L2 model with psi.npz + hidden.npz, at the mid-depth saved layer:
  frame   PCA of the true hidden state on the dev-train contexts (top 32 PCs) and a 3D display rotation (top 3)
  decoder ridge from θ of the live battery items (readouts bundle) to the 32 PCs, fit on dev-train
  check   held-out dev contexts: variance-weighted R² of the 32 PCs, and per-axis correlation of the 3D display
Saved to /data/models/graybox_ws/models/live/open_<model>.npz (used by graybox.live when the model is open).

python scripts/22_live_open_decoders.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

L2 = Path("/data/models/graybox_ws/outputs/l2")
MOD = Path("/data/models/graybox_ws/models/live")
RES = Path("/home/arya/graybox/results/live")


def main():
    R = np.load(MOD / "readouts.npz", allow_pickle=True)
    items = list(R["items"])
    C = pd.read_parquet(L2 / "contexts.parquet")
    dev = np.flatnonzero((C.split == "dev").values)
    rng = np.random.default_rng(0)
    perm = rng.permutation(dev)
    tr, te = np.sort(perm[: int(0.8 * len(perm))]), np.sort(perm[int(0.8 * len(perm)):])
    report = {}
    for d in sorted(L2.iterdir()):
        if not ((d / "psi.npz").exists() and (d / "hidden.npz").exists()):
            continue
        z = np.load(d / "psi.npz")
        col = [list(z["items"]).index(k) for k in items]
        q = (z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6)
        X = 2 * np.arcsin(np.sqrt(q[:, col]))
        Hz = np.load(d / "hidden.npz")
        j = len(Hz["layers"]) // 2
        H = Hz["h"][:, j].astype(np.float32)
        mu = H[tr].mean(0)
        _, S, Vt = np.linalg.svd(H[tr] - mu, full_matrices=False)
        comps, w = Vt[:32], S[:32] ** 2 / (S ** 2).sum()
        P = (H - mu) @ comps.T
        xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
        reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((X[tr] - xm) / xs, P[tr])
        Ph = reg.predict((X[te] - xm) / xs)
        r2 = 1 - ((P[te] - Ph) ** 2).sum(0) / ((P[te] - P[te].mean(0)) ** 2).sum(0)
        scale = 1.0 / np.percentile(np.linalg.norm(P[tr][:, :3], axis=1), 97)
        corr = [round(float(np.corrcoef(P[te][:, k], Ph[:, k])[0, 1]), 4) for k in range(3)]
        report[d.name] = {"layer": int(Hz["layers"][j]), "R2_top32_weighted": round(float((r2 * w).sum() / w.sum()), 4),
                          "display_axis_corr": corr, "n_test": int(len(te))}
        np.savez(MOD / f"open_{d.name}.npz", items=np.array(items), xm=xm, xs=xs, coef=reg.coef_, intercept=reg.intercept_,
                 mu=mu, comps=comps, scale=scale, layer=int(Hz["layers"][j]), check=json.dumps(report[d.name]))
        print(d.name, report[d.name], flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "open_decoders.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
