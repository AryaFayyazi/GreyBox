#!/usr/bin/env python
"""First-page figure (v6: the chain; panel c adds the second closed agent at its call and before it, G38). Based on v5.
v5: panel c adds the second benchmarks. (a) An open model's hidden geometry and (b) the same contexts placed only by its yes/no
answers (scripts/134's computation, unchanged). (c) Monitoring a decision (G15, held-out): per agent, the fixed read-out
of its own 8 answers vs a probe trained on its hidden states. (d) Prevention, predicted (G12, G13): observed vs predicted
live reduction. Palette: the dataviz skill's validated reference slots.
  python scripts/214_fig_teaser_v6.py  -> paper/figures/teaser_v6.pdf, results/figures/teaser_v6.png
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
    """G15 rows (CAPE-Bench, held-out) and, below a divider, the second benchmarks: G18 (Claude, tau2-bench) and, when
    its result exists, G17 (open agents, AgentDojo)."""
    R = json.loads((ROOT / "results/t1/g15_training_free.json").read_text())
    CI = json.loads((ROOT / "results/t1/g15_per_agent_ci_exploratory.json").read_text())
    rows = [(N, R[a]["within"]["mean8"], CI[a][1:], R[a]["within"]["white"], BLUE) for a, N in
            (("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss"), ("qwen38", "Qwen3.8"), ("museglimmer", "Muse-Glimmer"))]
    rows.append(("Claude Haiku 5.5, restarts", R["claude"]["restart_mean8"], R["claude"]["restart_mean8_ci"], None, ORANGE))
    f38 = ROOT / "results/t1/g38_gpt_readout.json"
    if f38.exists():
        g38 = json.loads(f38.read_text())
        rows.append(("GPT-6-luna (closed)", g38["A1_at_call"][0], g38["A1_at_call"][1:], None, ORANGE))
        rows.append(("GPT-6-luna, before call", g38["P1_pre_call"][0], g38["P1_pre_call"][1:], None, "hollow"))
    second = []
    f17 = ROOT / "results/t1/g17_agentdojo_readout.json"
    if f17.exists():
        g = json.loads(f17.read_text())
        second += [(f"{N}, AgentDojo", g[a]["within_suite"]["read_out_8"], g[a]["read_out_8_ci"], None, BLUE)
                   for a, N in (("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss")) if a in g]
    g = json.loads((ROOT / "results/t1/g18_tau2_claude_readout.json").read_text())
    second.append(("Claude Haiku 5.5, τ²-bench", g["R1_within_tool_read_out_8"][0], g["R1_within_tool_read_out_8"][1:], None, ORANGE))
    # (G21, another model answering, is a comparison outside the method: not shown)
    style(ax)
    ax.grid(axis="y", visible=False)
    X0 = 0.5
    ys, y = [], 0.0
    for k in range(len(second)):                      # bottom block first
        ys.append(y); y += 1
    ydiv = y - 0.35; y += 0.3
    ys2 = []
    for k in range(len(rows)):
        ys2.append(y); y += 1
    allrows = list(zip(second[::-1], ys)) + list(zip(rows[::-1], ys2))
    for (name, eight, ci_, wb, col), yy in allrows:
        if wb is not None:
            ax.scatter([wb], [yy], s=30, marker="D", facecolor="white", edgecolor=INK, lw=1.0, zorder=3)
        c_ = ORANGE if col == "hollow" else col
        ax.plot(list(ci_), [yy, yy], color=c_, lw=1.0, alpha=0.55, zorder=2, solid_capstyle="round")
        if col == "hollow":
            ax.scatter([eight], [yy], s=40, facecolor="white", edgecolor=ORANGE, lw=1.4, zorder=4)
        else:
            ax.scatter([eight], [yy], s=46, color=c_, edgecolor="white", lw=0.9, zorder=4)
        ax.text(X0 - 0.006, yy, name, fontsize=6.2, color=INK, ha="right", va="center")
    ax.axhline(ydiv, color=MUTE, lw=0.6, ls=(0, (2, 2)))
    ax.text(X0 + 0.006, ydiv - 0.45, "second benchmarks", fontsize=5.3, color=INK2, va="center", ha="left")
    ax.axvline(0.5, color=GRID, lw=0.8)
    ax.set_xlim(X0, 1.005); ax.set_ylim(-0.6, y - 0.4); ax.set_yticks([])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("violations ranked above allowed calls (AUROC)", fontsize=6.2, color=INK, labelpad=2)
    h = [ax.scatter([], [], s=46, color=BLUE, edgecolor="white", lw=0.9, label="open agent: its answers (95% CI)"),
         ax.scatter([], [], s=46, color=ORANGE, edgecolor="white", lw=0.9, label="closed agent, API only: its answers"),
         ax.scatter([], [], s=40, facecolor="white", edgecolor=ORANGE, lw=1.4, label="closed agent, before the call exists"),
         ax.scatter([], [], s=30, marker="D", facecolor="white", edgecolor=INK, lw=1.0, label="probe trained on its hidden states")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=5.5, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.25)


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
    e12 = np.mean([abs(y - x) for _, x, y, *_r, kind in pts if kind == "g12"])
    eall = np.mean([abs(y - x) for _, x, y, *_r in pts])
    ax.text(0.03, 0.97, f"mean |error| {e12:.3f} (G12)", transform=ax.transAxes, fontsize=5.6, color=INK, fontweight="bold", va="top")
    ax.text(0.03, 0.89, f"all 7 points: {eall:.3f}", transform=ax.transAxes, fontsize=5.3, color=INK2, va="top")
    h = [ax.scatter([], [], s=34, marker="o", color=BLUE, edgecolor="white", lw=0.8, label="violations, 5 agents"),
         ax.scatter([], [], s=34, marker="s", color=BLUE, edgecolor="white", lw=0.8, label="injections, 2 agents"),
         ax.scatter([], [], s=34, marker="o", color=ORANGE, edgecolor="white", lw=0.8, label="Claude Haiku 5.5 (closed)")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.03, -0.02), ncol=1, frameon=True, framealpha=0.95, edgecolor="none", fontsize=5.2, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)


def main():
    Xh, Xa, src = T.geometry()
    fig = plt.figure(figsize=(7.0, 2.35), facecolor="white")
    gs = fig.add_gridspec(1, 6, width_ratios=[0.84, 0.84, 0.22, 1.2, 0.10, 0.86], wspace=0.30, left=0.005, right=0.995, top=0.66, bottom=0.16)
    ax = [fig.add_subplot(gs[0, i]) for i in (0, 1, 3, 5)]
    T.T.scatter(ax[0], Xh, src)
    T.T.scatter(ax[1], Xa, src)
    panel_decision(ax[2]); panel_law(ax[3])
    heads = [("a", "Hidden geometry", f"{T.T.NAME}, hidden states"), ("b", "Reconstructed", "from its answers; ρ = 0.74"),
             ("c", "Monitor and forecast", "held-out calls; preregistered"), ("d", "Prevent, predicted", "live; preregistered")]
    fig.canvas.draw()
    ytitle = max(a_.get_position().y1 for a_ in ax) + 0.255
    for a_, (k, title, sub) in zip(ax, heads):
        pos = a_.get_position()
        x0 = pos.x0 - (0.085 if k == "c" else (0.04 if k == "d" else 0.0))
        fig.text(x0, ytitle, f"({k}) {title}", fontsize=7.9, fontweight="bold", color=INK, va="bottom")
        fig.text(x0, ytitle - 0.058, sub, fontsize=5.6, color=INK2, va="bottom")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=3, frameon=False, fontsize=5.9, markerscale=3, handletextpad=0.1, columnspacing=0.9, bbox_to_anchor=(0.005, -0.09))
    for out in (ROOT / "paper/figures/teaser_v6.pdf", ROOT / "results/figures/teaser_v6.png"):
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("saved")


if __name__ == "__main__":
    main()
