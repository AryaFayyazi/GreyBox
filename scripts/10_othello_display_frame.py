#!/usr/bin/env python
"""Add a display frame to the Othello demo bundle: PCA of the TRUE top-64 PCs over positions t >= 1 (the opening
position's state is an outlier that otherwise takes the first axis). Estimated and true paths are both mapped by the
same 64->3 rotation, so the comparison stays exact; only the viewing angle changes."""
from pathlib import Path

import numpy as np
import torch

OUT = Path("/data/models/graybox_ws/outputs/othello")
MOD = Path("/data/models/graybox_ws/models/othello/demo.pt")
ck = torch.load(MOD, weights_only=False)
Z = np.load(OUT / "l1_g3000_p10_s0.npz")
games = np.unique(Z["game"])
te = np.isin(Z["game"], np.random.default_rng(1).choice(games, int(len(games) * 0.2), replace=False))
H = Z[f"resid{ck['layer']}"].astype(np.float32)[~te]
t = Z["t"][~te]
P = (H - ck["hmu"]) @ ck["comps"].T                      # train-state true PCs (64)
keep = t >= 1
mu = P[keep].mean(0)
_, S, Vt = np.linalg.svd(P[keep] - mu, full_matrices=False)
D = Vt[:3].T                                              # (64, 3)
cloud = (P[keep] - mu) @ D
scale = 1.0 / np.percentile(np.linalg.norm(cloud, axis=1), 97)
rng = np.random.default_rng(0)
idx = rng.choice(np.flatnonzero(keep), 3000, replace=False)
ck.update({"disp_mu": mu.astype(np.float32), "disp_dirs": D.astype(np.float32), "disp_scale": float(scale),
           "disp_cloud": (((P[idx] - mu) @ D) * scale).astype(np.float32), "disp_cloud_t": t[idx].astype(np.int16),
           "disp_var_explained": float((S[:3] ** 2).sum() / (S ** 2).sum())})
torch.save(ck, MOD)
print("display frame saved; var explained by 3 axes (t>=1):", round(ck["disp_var_explained"], 3))
