#!/usr/bin/env python
"""Figure: the agent's true hidden state vs the grey-box replica from the transcript (T15's map, READER, alpha 1000).
(a) true fresh decision states on dev PCs 1-2, (b) READER replicas on the same axes, (c) per-PC R^2 on fresh steps.
python scripts/92_fig_replica.py"""
import importlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
F = importlib.import_module("90_figures")
I, T4 = Path("/data/models/graybox_ws/outputs/interrogate"), Path("/data/models/graybox_ws/outputs/t4")
Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
Dd, Df = np.load(I / "gemma4_clean_decision_hidden.npz"), np.load(I / "gemma4_fresh_decision_hidden.npz")
okd, okf = Dd["ok"].astype(bool), Df["ok"].astype(bool)
Yd_full, Yf_full = Dd["h"][okd].reshape(okd.sum(), -1).astype(np.float32), Df["h"][okf].reshape(okf.sum(), -1).astype(np.float32)
sc = StandardScaler().fit(Yd_full); pca = PCA(128, random_state=0).fit(sc.transform(Yd_full))
Yd, Yf = pca.transform(sc.transform(Yd_full)), pca.transform(sc.transform(Yf_full))
Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz"); j = [int(x) for x in Hd["layers"]].index(32)
xs = StandardScaler().fit(Hd["h"][okd, j].astype(np.float32))
rec = Ridge(alpha=1000).fit(xs.transform(Hd["h"][okd, j].astype(np.float32)), Yd).predict(xs.transform(Hf["h"][okf, j].astype(np.float32)))
y = Uf.y.values[okf].astype(bool)
t15 = json.loads((ROOT / "results/t1/t15_latent_replication.json").read_text())
assert abs(r2_score(Yf, rec, multioutput="variance_weighted") - t15["r2_fresh"]["READER"]) < 1e-3
per = [r2_score(Yf[:, k], rec[:, k]) for k in range(128)]
rng = np.random.default_rng(0); safe = np.flatnonzero(~y); sel = np.r_[rng.choice(safe, 2500, replace=False), np.flatnonzero(y)]
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
lim = np.percentile(np.abs(Yf[sel, :2]), 99.5) * 1.05
for ax, Z, title in ((axs[0], Yf, "(a) The agent's true hidden state"), (axs[1], rec, "(b) Grey-box replica, from the transcript")):
    ax.scatter(Z[sel][~y[sel], 0], Z[sel][~y[sel], 1], s=3, c="#9aa5b1", alpha=0.45, lw=0, label="safe action")
    ax.scatter(Z[sel][y[sel], 0], Z[sel][y[sel], 1], s=7, c=F.C[1], alpha=0.85, lw=0, label="violation")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("PC 1 of the agent's state"); ax.set_ylabel("PC 2")
    ax.set_title(title, loc="left", fontsize=9, color=F.INK)
axs[0].legend(fontsize=7, frameon=False, loc="lower left", markerscale=2.5)
ax = axs[2]; ax.grid(color=F.GRID, lw=0.6); ax.set_axisbelow(True)
ax.bar(np.arange(1, 129), per, color=F.C[0], width=0.9)
ax.set_xlabel("Principal component of the agent's state (by variance)"); ax.set_ylabel("$R^2$ of the replica (fresh steps)")
ax.set_ylim(0, 1); ax.set_xlim(0, 129)
ax.set_title(f"(c) Variance replicated: {t15['r2_fresh']['READER']:.0%} (text only: {t15['r2_fresh']['TXT']:.0%})", loc="left", fontsize=9, color=F.INK)
fig.tight_layout(); F.save(fig, "replica")
print("pc1-2 R2", round(per[0], 3), round(per[1], 3))
