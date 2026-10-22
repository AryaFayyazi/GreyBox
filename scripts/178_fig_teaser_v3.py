#!/usr/bin/env python
"""First-page figure (v3): monitoring a closed agent through its own answers, with no training.
  (a) how it works, on a REAL call of Claude Haiku 5.5 from the G15 battery: the proposed call, its own answers to the
      8 side-branch questions (4 answers each), and where the fixed read-out puts the call among Claude's calls of the
      same kind (restart template: allowed vs violating);
  (b) G15 (preregistered, held-out): per agent, the read-out of its 8 answers vs a probe trained on its hidden states
      vs its single self-audit answer (within-template AUROC);
  (c) the prevention law: predicted vs observed live reduction (G12 dots, G13 squares; scripts/160's points).
Palette: the dataviz skill's validated reference slots (blue = the method, orange = violation / the closed agent).
  python scripts/178_fig_teaser_v3.py  -> paper/figures/teaser_v3.pdf, results/figures/teaser_v3.png
"""
import importlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), "/home/arya/new_safety/cape/src", "/home/arya/new_safety/cape/scripts"]
BLUE, ORANGE, AQUA, INK, INK2, MUTE, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.4, "axes.edgecolor": MUTE, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.4, "ytick.major.size": 2.4, "axes.spines.top": False, "axes.spines.right": False})


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OR = _mod("obsref", "web/build_observable_ref.py")
L160 = _mod("t160", "scripts/160_fig_law.py")
SHORT = ["complies with the policy?", "resource it may not use?", "data outside its audience?", "follows an injected instruction?",
         "production, without approval?", "credential not issued?", "affects uninvolved people?", "breaks a task constraint?"]
EXAMPLE = ("third_party" , "report-builder")


def claude_example():
    C = pd.DataFrame([json.loads(x) for x in open("/data/models/graybox_ws/outputs/decision_greybox/claude_t8b.jsonl")])
    cnt = np.array([json.loads(c) for c in C.counts])
    T = OR.theta(cnt[:, :, 0], cnt[:, :, 1])
    C["s"] = np.c_[-T[:, 0], T[:, 1:]].mean(1)
    U = importlib.import_module("172_claude_decision_greybox").units().set_index(["episode_id", "event_id"])
    rs = C[C.template.str.startswith("cons_service_restart")].copy()
    rs["args"] = [U.loc[(e, v)].args for e, v in zip(rs.episode_id, rs.event_id)]
    ex = rs[(rs.y == 1) & rs.args.str.contains("report-builder")].sort_values("s", ascending=False).iloc[0]
    return ex, rs


def panel_how(ax):
    ex, rs = claude_example()
    counts = json.loads(ex.counts)
    cmd = json.loads(ex.args)["command"]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # 1. the proposed call
    ax.add_patch(FancyBboxPatch((0.0, 0.815), 0.70, 0.165, boxstyle="round,pad=0.006,rounding_size=0.02", fc="#f3f2ee", ec="none"))
    ax.text(0.012, 0.945, "Claude Haiku 5.5  (closed: only its API)", fontsize=5.9, color=INK, va="center", fontweight="bold")
    ax.text(0.012, 0.893, "proposes the tool call", fontsize=5.4, color=INK2, va="center")
    ax.text(0.012, 0.845, cmd, fontsize=5.2, color=INK, family="DejaVu Sans Mono", va="center")
    # 2. the side branch: 8 questions, 4 answers each
    ax.text(0.0, 0.745, "Before it runs, Claude answers 8 questions in side branches", fontsize=5.6, color=INK, fontweight="bold", va="center")
    ax.text(0.392, 0.69, "4 answers", fontsize=4.9, color=INK2, va="center", ha="center")
    y0, dy = 0.635, 0.068
    for j, (q, (k, m)) in enumerate(zip(SHORT, counts)):
        y = y0 - j * dy
        ax.text(0.0, y, q, fontsize=5.3, color=INK, va="center")
        for i in range(4):
            x = 0.36 + i * 0.024
            if i < m:
                yes = i < k
                warn = yes != (j == 0)                       # "no" to "complies", or "yes" to a rule: points to a violation
                ax.scatter([x], [y], s=19, color=ORANGE if warn else "#cfcec8", zorder=3, lw=0)
                ax.text(x, y - 0.002, "Y" if yes else "N", fontsize=3.5, color="white", ha="center", va="center", zorder=4, fontweight="bold")
            else:
                ax.scatter([x], [y], s=19, facecolor="white", edgecolor="#cfcec8", lw=0.6, zorder=3)
    ax.scatter([0.012], [0.035], s=14, color=ORANGE, lw=0); ax.text(0.03, 0.035, "points to a violation", fontsize=4.9, color=INK2, va="center")
    ax.scatter([0.33], [0.035], s=14, color="#cfcec8", lw=0); ax.text(0.348, 0.035, "it does not", fontsize=4.9, color=INK2, va="center")
    # 3. the fixed read-out, among Claude's calls of this kind
    ax.add_patch(FancyArrowPatch((0.475, 0.40), (0.555, 0.40), arrowstyle="-|>", mutation_scale=7, color=MUTE, lw=0.8))
    sub = ax.inset_axes([0.62, 0.24, 0.37, 0.40])
    bins = np.linspace(0.15, 0.70, 23)
    a, v = rs[rs.y == 0].s.values, rs[rs.y == 1].s.values
    ha, _ = np.histogram(a, bins); hv, _ = np.histogram(v, bins)
    w = bins[1] - bins[0]
    sub.bar(bins[:-1] + w / 2, ha / ha.max(), width=w * 0.86, bottom=1.08, color="#c9c8c2", lw=0)
    sub.bar(bins[:-1] + w / 2, hv / hv.max(), width=w * 0.86, bottom=0.0, color=ORANGE, lw=0, alpha=0.9)
    sub.axvline(ex.s, color=INK, lw=0.9, zorder=4)
    sub.annotate("this call", (ex.s, 2.12), xytext=(ex.s - 0.02, 2.12), fontsize=4.9, color=INK, ha="right", va="center")
    sub.set_yticks([1.55, 0.5], ["allowed", "violating"], fontsize=5.0)
    sub.set_ylim(0, 2.2); sub.set_xlim(0.14, 0.71)
    sub.tick_params(axis="x", labelsize=4.8, pad=1)
    sub.set_xlabel("its 8 answers, fixed read-out", fontsize=5.0, labelpad=1)
    sub.set_title("among Claude's restart calls", fontsize=5.0, color=INK2, pad=2)
    ax.add_patch(FancyBboxPatch((0.62, 0.025), 0.37, 0.085, boxstyle="round,pad=0.006,rounding_size=0.025", fc=INK, ec="none"))
    ax.text(0.805, 0.068, "held for review", fontsize=5.6, color="white", ha="center", va="center", fontweight="bold")


def panel_g15(ax):
    R = json.loads((ROOT / "results/t1/g15_training_free.json").read_text())
    rows = [("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss-20b"), ("qwen38", "Qwen3.8-27B"), ("museglimmer", "Muse-Glimmer"), ("claude", "Claude Haiku 5.5\n(restarts)")]
    for i, (a, name) in enumerate(rows[::-1]):
        ax.axhspan(i - 0.42, i + 0.42, color="#f6f5f2" if i % 2 else SURF, lw=0, zorder=0)
        if a == "claude":
            c = R["claude"]
            one, eight, wb = c["restart_complies"], c["restart_mean8"], None
        else:
            w = R[a]["within"]
            one, eight, wb = w["complies"], w["mean8"], w["white"]
        lo = min(one, eight) if wb is None else min(one, eight, wb)
        ax.plot([lo, eight], [i, i], color="#d8d7d2", lw=1.2, zorder=1)
        ax.scatter([one], [i], s=22, facecolor="white", edgecolor=BLUE, lw=1.0, zorder=3)
        if wb is not None:
            ax.scatter([wb], [i], s=26, marker="D", facecolor="white", edgecolor=INK, lw=0.9, zorder=3)
        ax.scatter([eight], [i], s=40, color=ORANGE if a == "claude" else BLUE, edgecolor="white", lw=0.8, zorder=4)
        ax.text(0.27, i, name, fontsize=5.6, color=INK, ha="right", va="center")
    ax.set_xlim(0.29, 1.01); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("violations ranked above allowed calls\n(AUROC within each kind of task; held-out)", fontsize=5.4, labelpad=1)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    h = [ax.scatter([], [], s=40, color=BLUE, label="its 8 answers, fixed read-out"),
         ax.scatter([], [], s=26, marker="D", facecolor="white", edgecolor=INK, lw=0.9, label="probe trained on its weights"),
         ax.scatter([], [], s=22, facecolor="white", edgecolor=BLUE, lw=1.0, label="its single answer")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, frameon=True, framealpha=0.95, edgecolor="none", fontsize=4.9, handletextpad=0.2, borderaxespad=0.3)


def panel_law(ax):
    pts = [p for p in L160.points() if p[6] in ("g12", "g13")]
    xs = np.array([0, 1.05])
    ax.fill_between(xs, xs - 0.15, xs + 0.15, color="#eeede9", lw=0, zorder=0)
    ax.plot(xs, xs, color=MUTE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for name, x, y, lo, hi, c, kind in pts:
        col = ORANGE if "Claude" in name else BLUE
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=col, elinewidth=0.8, capsize=1.4, alpha=0.8, zorder=2)
        ax.scatter([x], [y], s=26, marker="s" if kind == "g13" else "o", color=col, edgecolor="white", lw=0.7, zorder=3)
    ax.text(0.04, 0.93, "5 agents (CAPE-Bench) and\n2 on prompt injection (AgentDojo)", transform=ax.transAxes, fontsize=5.1, color=INK2, va="top")
    ax.text(0.62, 0.40, "± 0.15", fontsize=5.0, color=MUTE, rotation=38)
    ax.set_xlim(-0.04, 1.05); ax.set_ylim(-0.25, 1.0)
    ax.set_xlabel("predicted from offline flag rates", fontsize=5.4, labelpad=1)
    ax.set_ylabel("observed live reduction", fontsize=5.4, labelpad=1)
    ax.tick_params(labelsize=5.0)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def main():
    fig = plt.figure(figsize=(7.0, 2.5), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.05, 1.0, 0.85], wspace=0.40, left=0.01, right=0.99, top=0.84, bottom=0.17)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    panel_how(axa); panel_g15(axb); panel_law(axc)
    for ax, t, sub, dx in ((axa, "a", "How it works: ask the agent, read the answers", 0.0), (axb, "b", "It ranks violations like a trained probe", -0.42),
                           (axc, "c", "Used as a gate: prevention is predicted", -0.22)):
        ax.text(dx, 1.13 if ax is not axa else 1.10, t, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=INK, va="bottom")
        ax.text(dx + 0.05 if ax is axa else dx + 0.09, 1.13 if ax is not axa else 1.10, sub, transform=ax.transAxes, fontsize=6.6, fontweight="bold", color=INK, va="bottom")
    for out in (ROOT / "paper/figures/teaser_v3.pdf", ROOT / "results/figures/teaser_v3.png"):
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("saved")


if __name__ == "__main__":
    main()
