#!/usr/bin/env python
"""First-page figure: a model's latent geometry, read from its answers (nothing is fit except a display rotation).
  (a) the TRUE hidden geometry of one open model (Llama-3.1 8B, G1's dev-chosen layer), 1,440 held-out contexts:
      classical scaling of chord distances sqrt(2 (1 - cos)), coloured by the kind of context
  (b) the same contexts from the model's yes/no answers only (Fisher-Rao, 88 questions): classical scaling, then an
      orthogonal Procrustes rotation and scale onto (a) for display (the geometry is defined up to a rigid motion)
  (c) every pair of contexts: rank of the hidden distance vs rank of the answer distance (Spearman = their correlation)
  (d) the same for the model's own replies (G5: mean embedding of 8 replies), or, before G5 exists, the input text
  (e) Claude Haiku 5.5 (closed): classical scaling of its cross-fit Fisher-Rao distances (24 questions, 4 + 4 samples),
      rotated onto the open model's hidden geometry of the same 192 contexts for display only
python scripts/134_fig_teaser_geometry.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from graybox.geometry import mds  # noqa: E402


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = _mod("g", "129_infogeom.py")
v2 = _mod("v2", "118_paper_figures_v2.py")
plt = v2.plt
INK, MUTE = v2.INK, v2.MUTE
SRC = {"harmful": ("harmful request", "#eb6834"), "benign": ("benign request", "#2a78d6"), "sst2": ("review", "#1baf7a"),
       "boolq": ("yes/no question", "#eda100"), "gsm8k": ("maths problem", "#8a5cd1"), "wiki": ("encyclopedia", "#5d6873")}
MODEL, NAME = "llama31_8b", "Llama-3.1 8B"


def procrustes(X, Y):
    """Rotate (orthogonal, reflections allowed) and scale Y onto X, both centred."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(Yc.T @ Xc)
    return Yc @ (U @ Vt) * (s.sum() / (Yc ** 2).sum())


def chord(H):
    return np.sqrt(np.maximum(2 * g.cos_dist(H), 0))


def scatter(ax, X, src, s=2.2):
    for k, (lab, col) in SRC.items():
        i = src == k
        if i.any():
            ax.scatter(X[i, 0], X[i, 1], s=s, color=col, alpha=0.65, lw=0, rasterized=True, label=lab)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c9ced3"); sp.set_linewidth(0.5)


def copula(ax, dh, do, rho):
    n = len(dh)
    ax.hist2d(dh / n, do / n, bins=48, cmap="Blues", norm=LogNorm(vmin=25), rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 1], ["near", "far"]); ax.set_yticks([0, 1], ["near", "far"])
    ax.tick_params(length=0, pad=1.5)
    ax.text(0.04, 0.95, r"$\rho$ = %.2f" % rho, transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def main():
    C = g.contexts()
    test = np.where(C.split.values == "confirm")[0]
    src = C.source.values[test]
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    k, layer, _ = g.select_layer(MODEL, dev_sel)
    H, _ = g.hidden(MODEL)
    Hh = H[test, k]
    p, _ = g.p_yes(MODEL)
    Dh, Da = chord(Hh), np.sqrt(g.sq_dist(g.theta(p[test])))
    Xh10, _ = mds(Dh, 10)                                   # rotate in 10 dimensions, show the hidden frame's top 2
    Xa = procrustes(Xh10, mds(Da, 10)[0])[:, :2]
    Xh = Xh10[:, :2]
    iu = np.triu_indices(len(test), 1)
    rh, ra = rankdata(Dh[iu]), rankdata(Da[iu])
    g5 = Path("/data/models/graybox_ws/outputs/g5/confirm") / MODEL / "replies_none.minilm.npy"
    if g5.exists():
        E = np.load(g5).reshape(len(test), 8, -1).astype(np.float64).mean(1)
        Do, other = chord(E), "its own replies"
    else:
        Do, other = chord(np.load(g.L2 / "minilm.npy").astype(np.float64)[test]), "the input text"
    ro = rankdata(Do[iu])
    rho_a, rho_o = float(np.corrcoef(rh, ra)[0, 1]), float(np.corrcoef(rh, ro)[0, 1])
    mz = np.load("/data/models/graybox_ws/outputs/infogeom/claude_mds_crossfit.npz", allow_pickle=True)
    cid = list(mz["ids"])
    pos = C.reset_index().set_index("id").loc[cid, "index"].values
    Xc_h, _ = mds(chord(H[pos, k]), 2)
    Xc = procrustes(Xc_h, mz["X"])                            # Claude's 2-D coordinates (cross-fit), rotated for display

    fig = plt.figure(figsize=(7.0, 1.95))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.78, 0.78, 1], wspace=0.34)
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    scatter(ax[0], Xh, src)
    ax[0].set_title("(a) Hidden geometry", loc="left", fontsize=7, fontweight="bold")
    ax[0].text(0, 1.0, f"{NAME}, weights open", transform=ax[0].transAxes, fontsize=5.4, color=MUTE, va="bottom")
    scatter(ax[1], Xa, src)
    ax[1].set_title("(b) From its answers", loc="left", fontsize=7, fontweight="bold")
    ax[1].text(0, 1.0, "no weights, no training", transform=ax[1].transAxes, fontsize=5.4, color=MUTE, va="bottom")
    copula(ax[2], rh, ra, rho_a)
    ax[2].set_title("(c) Answers", loc="left", fontsize=7, fontweight="bold")
    ax[2].set_xlabel("hidden distance (rank)", fontsize=5.6, labelpad=1); ax[2].set_ylabel("answer distance (rank)", fontsize=5.6, labelpad=0)
    copula(ax[3], rh, ro, rho_o)
    ax[3].set_title("(d) " + ("Replies" if g5.exists() else "Input text"), loc="left", fontsize=7, fontweight="bold")
    ax[3].set_xlabel("hidden distance (rank)", fontsize=5.6, labelpad=1); ax[3].set_yticklabels([])   # same axes as (c)
    scatter(ax[4], Xc, mz["source"], s=5)
    ax[4].set_title("(e) Claude (closed)", loc="left", fontsize=7, fontweight="bold")
    ax[4].text(0, 1.0, "from 8 sampled answers", transform=ax[4].transAxes, fontsize=5.4, color=MUTE, va="bottom")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=5.8, markerscale=3, handletextpad=0.1, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.07))
    v2.save(fig, "teaser_geometry")
    print("saved; rho answers %.3f, %s %.3f" % (rho_a, other, rho_o))


if __name__ == "__main__":
    main()
