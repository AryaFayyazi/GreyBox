#!/usr/bin/env python
"""First-page figure for the grey-box method (one row, four panels):
  (a) an open model's TRUE hidden geometry and (b) the same contexts placed from its yes/no answers only (scripts/134,
      unchanged computation);
  (c) recreating an agent's decision state through its API (G14): within-template AUROC of a probe trained on TRUE
      decision states, applied to the true state (white box) and to the state recreated from the API (grey box), per
      agent; for agents other than Gemma-4 also Gemma-4's grey box (another agent's coordinates);
  (d) the prevention law: predicted vs observed live reduction (scripts/160's points: G12, G13; G11b failures).
  python scripts/173_fig_teaser_greybox.py [dry=1]   (dry=1 reads the G14 dry run and stamps the panel DRY RUN)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _mod("t134", "134_fig_teaser_geometry.py")
L = _mod("t160", "160_fig_law.py")
g, v2, plt, INK, MUTE = T.g, T.v2, T.plt, T.INK, T.MUTE
DRY = "dry=1" in sys.argv[1:]
NAMES = {"gemma4": "Gemma-4", "gptoss20b": "gpt-oss", "qwen38": "Qwen3.8", "museglimmer": "Muse-Glimmer", "claude": "Claude Haiku 5.5"}


def geometry():
    C = g.contexts()
    test = np.where(C.split.values == "confirm")[0]
    dev = np.where(C.split.values == "dev")[0]
    k, _, _ = g.select_layer(T.MODEL, np.random.default_rng(0).choice(dev, 1000, replace=False))
    H, _ = g.hidden(T.MODEL)
    p, _ = g.p_yes(T.MODEL)
    Dh, Da = T.chord(H[test, k]), np.sqrt(g.sq_dist(g.theta(p[test])))
    Xh10, _ = T.mds(Dh, 10)
    return Xh10[:, :2], T.procrustes(Xh10, T.mds(Da, 10)[0])[:, :2], C.source.values[test]


def greybox_panel(ax):
    f = ROOT / "results/t1" / ("g14_DRYRUN_dev.json" if DRY else "g14_decision_grey_box.json")
    R = json.loads(f.read_text())
    rows = [a for a in ("gemma4", "gptoss20b", "qwen38", "museglimmer", "claude") if a in R]
    for i, a in enumerate(rows[::-1]):
        w = R[a]["within"]
        if "white_box" in w:
            ax.scatter([w["white_box"]], [i], s=26, facecolor="white", edgecolor=INK, lw=1.0, zorder=3)
        if "grey_box" in w:
            ax.scatter([w["grey_box"]], [i], s=26, color=L.BLUE if a != "claude" else L.ORANGE, zorder=4)
        if "others_grey_box" in w:
            ax.scatter([w["others_grey_box"]], [i], s=22, marker="s", color=L.AQUA, zorder=4)
        ax.text(0.49, i, NAMES[a], ha="right", va="center", fontsize=5.8, color=INK)
    ax.set_yticks([]); ax.set_xlim(0.5, 1.0); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("violations ranked (AUROC, within template)", fontsize=5.6, labelpad=1)
    ax.grid(axis="x", color=L.GRID, lw=0.5); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    h = [ax.scatter([], [], s=26, facecolor="white", edgecolor=INK, lw=1.0, label="true state (weights)"),
         ax.scatter([], [], s=26, color=L.BLUE, label="recreated via API"),
         ax.scatter([], [], s=22, marker="s", color=L.AQUA, label="in other agents' coordinates")]
    ax.legend(handles=h, loc="lower left", fontsize=5.0, handletextpad=0.2, borderaxespad=0.2)
    if DRY:
        ax.text(0.5, 0.5, "DRY RUN", transform=ax.transAxes, fontsize=14, color="#d0d0d0", ha="center", va="center", rotation=20, zorder=0)


def main():
    Xh, Xa, src = geometry()
    fig = plt.figure(figsize=(7.0, 2.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 0.85, 1.25, 1.3], wspace=0.42)
    ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
    T.scatter(ax[0], Xh, src)
    ax[0].set_title("(a) Hidden geometry", loc="left", fontsize=7, fontweight="bold")
    ax[0].text(0, 1.0, f"{T.NAME}, weights open", transform=ax[0].transAxes, fontsize=5.2, color=MUTE, va="bottom")
    T.scatter(ax[1], Xa, src)
    ax[1].set_title("(b) From its answers", loc="left", fontsize=7, fontweight="bold")
    ax[1].text(0, 1.0, "no weights, no training", transform=ax[1].transAxes, fontsize=5.2, color=MUTE, va="bottom")
    greybox_panel(ax[2])
    ax[2].set_title("(c) Recreated decision state", loc="left", fontsize=7, fontweight="bold")
    L.law_panel(ax[3], L.points())
    ax[3].set_title("(d) Self-policing, predicted", loc="left", fontsize=7, fontweight="bold")
    ax[3].legend_.remove()
    for t in ax[3].texts:
        t.set_fontsize(4.6)
    ax[3].set_xlabel("predicted reduction", fontsize=5.6, labelpad=1); ax[3].set_ylabel("observed", fontsize=5.6, labelpad=0)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=3, frameon=False, fontsize=5.4, markerscale=3, handletextpad=0.1, columnspacing=0.8,
               bbox_to_anchor=(0.01, -0.13))
    v2.save(fig, "teaser_greybox" + ("_DRY" if DRY else ""))
    print("saved")


if __name__ == "__main__":
    main()
