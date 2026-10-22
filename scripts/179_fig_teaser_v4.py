#!/usr/bin/env python
"""First-page figure (v4). (a) An open model's hidden geometry and (b) the same contexts placed only by its yes/no
answers (scripts/134's computation, unchanged). (c) Monitoring a decision (G15, held-out): per agent, the fixed read-out
of its own 8 answers vs a probe trained on its hidden states. (d) Prevention, predicted (G12, G13): observed vs predicted
live reduction. Palette: the dataviz skill's validated reference slots.
  python scripts/179_fig_teaser_v4.py  -> paper/figures/teaser_v4.pdf, results/figures/teaser_v4.png
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _mod("t173", "scripts/173_fig_teaser_greybox.py")           # geometry() from scripts/134
L160 = _mod("t160", "scripts/160_fig_law.py")
plt = T.plt
BLUE, ORANGE, INK, INK2, MUTE, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def style(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color(MUTE); ax.spines["bottom"].set_color(MUTE)
    ax.tick_params(labelsize=5.9, colors=INK2, length=2.2, width=0.6)


def panel_decision(ax):
    R = json.loads((ROOT / "results/t1/g15_training_free.json").read_text())
    CI = json.loads((ROOT / "results/t1/g15_per_agent_ci_exploratory.json").read_text())
    rows = [("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss"), ("qwen38", "Qwen3.8"), ("museglimmer", "Muse-Glimmer"), ("claude", "Claude Haiku 5.5")]
    style(ax)
    ax.grid(axis="y", visible=False)
    for i, (a, name) in enumerate(rows[::-1]):
        if a == "claude":
            eight, wb = R["claude"]["restart_mean8"], None
        else:
            eight, wb = R[a]["within"]["mean8"], R[a]["within"]["white"]
        if wb is not None:
            ax.scatter([wb], [i], s=30, marker="D", facecolor="white", edgecolor=INK, lw=1.0, zorder=3)
        lo_, hi_ = (R["claude"]["restart_mean8_ci"] if a == "claude" else CI[a][1:])
        ax.plot([lo_, hi_], [i, i], color=ORANGE if a == "claude" else BLUE, lw=1.0, alpha=0.55, zorder=2, solid_capstyle="round")
        ax.scatter([eight], [i], s=46, color=ORANGE if a == "claude" else BLUE, edgecolor="white", lw=0.9, zorder=4)
        ax.text(0.635, i, name, fontsize=6.4, color=INK, ha="right", va="center")
    ax.axhline(0.5, color=GRID, lw=0.8)
    ax.text(0.645, -0.36, "closed: no hidden states to train on", fontsize=5.3, color=INK2, va="center")
    ax.set_xlim(0.64, 1.005); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("violations ranked above allowed calls (AUROC)", fontsize=6.2, color=INK, labelpad=2)
    h = [ax.scatter([], [], s=46, color=BLUE, edgecolor="white", lw=0.9, label="its own answers, no training (95% CI)"),
         ax.scatter([], [], s=30, marker="D", facecolor="white", edgecolor=INK, lw=1.0, label="probe trained on its hidden states")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=5.7, handletextpad=0.3, borderaxespad=0.2)


def panel_law(ax):
    style(ax)
    pts = [p for p in L160.points() if p[6] in ("g12", "g13")]
    xs = np.array([-0.05, 1.05])
    ax.fill_between(xs, xs - 0.15, xs + 0.15, color="#efeeea", lw=0, zorder=0)
    ax.plot(xs, xs, color=MUTE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for name, x, y, lo, hi, c, kind in pts:
        col = ORANGE if "Claude" in name else BLUE
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=col, elinewidth=0.8, capsize=0, alpha=0.55, zorder=2)
        ax.scatter([x], [y], s=34, marker="s" if kind == "g13" else "o", color=col, edgecolor="white", lw=0.8, zorder=3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.2, 1.05)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("predicted from offline flag rates", fontsize=6.2, color=INK, labelpad=2)
    ax.set_ylabel("observed live reduction", fontsize=6.2, color=INK, labelpad=1)
    ax.text(0.03, 0.97, "mean |error| 0.035", transform=ax.transAxes, fontsize=5.9, color=INK, fontweight="bold", va="top")
    h = [ax.scatter([], [], s=34, marker="o", color=BLUE, edgecolor="white", lw=0.8, label="violations, 5 agents"),
         ax.scatter([], [], s=34, marker="s", color=BLUE, edgecolor="white", lw=0.8, label="injections, 2 agents"),
         ax.scatter([], [], s=34, marker="o", color=ORANGE, edgecolor="white", lw=0.8, label="Claude (closed)")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.03, -0.02), ncol=1, frameon=True, framealpha=0.95, edgecolor="none", fontsize=5.2, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)


def main():
    Xh, Xa, src = T.geometry()
    fig = plt.figure(figsize=(7.0, 2.15), facecolor="white")
    gs = fig.add_gridspec(1, 6, width_ratios=[0.84, 0.84, 0.22, 1.2, 0.10, 0.86], wspace=0.30, left=0.005, right=0.995, top=0.74, bottom=0.17)
    ax = [fig.add_subplot(gs[0, i]) for i in (0, 1, 3, 5)]
    T.T.scatter(ax[0], Xh, src)
    T.T.scatter(ax[1], Xa, src)
    panel_decision(ax[2]); panel_law(ax[3])
    heads = [("a", "Hidden geometry", f"{T.T.NAME}, hidden states"), ("b", "From its answers", "API only, no training; ρ = 0.74"),
             ("c", "Monitoring a decision", "held-out calls; preregistered"), ("d", "Prevention, predicted", "live; preregistered")]
    fig.canvas.draw()
    ytitle = max(a_.get_position().y1 for a_ in ax) + 0.20
    for a_, (k, title, sub) in zip(ax, heads):
        pos = a_.get_position()
        x0 = pos.x0 - (0.085 if k == "c" else (0.04 if k == "d" else 0.0))
        fig.text(x0, ytitle, f"({k}) {title}", fontsize=7.9, fontweight="bold", color=INK, va="bottom")
        fig.text(x0, ytitle - 0.058, sub, fontsize=5.6, color=INK2, va="bottom")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=3, frameon=False, fontsize=5.9, markerscale=3, handletextpad=0.1, columnspacing=0.9, bbox_to_anchor=(0.005, -0.09))
    for out in (ROOT / "paper/figures/teaser_v4.pdf", ROOT / "results/figures/teaser_v4.png"):
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("saved")


if __name__ == "__main__":
    main()
