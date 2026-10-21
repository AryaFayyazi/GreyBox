#!/usr/bin/env python
"""Figure: the prevention law. x = the share of its own violating calls that the agent flags in an OFFLINE development
audit (cut 1.0; Claude: the checklist on non-read calls); y = the live gate's relative reduction in episodes with an
executed violation, with 95% intervals. Filled: preregistered tests (G9 Gemma-4 overall; G11b Qwen3.8, Muse-Glimmer).
Hollow: exploratory (G9 per template, >= 10 plain violating episodes; G10 Claude's two sets, with the prediction
computed after the test from development data). Diagonal: the law's prediction for a gate.
  python scripts/160_fig_law.py   -> paper/figures/law.pdf and results/figures/law.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTE, GRID = "#1b2126", "#5d6873", "#e3e6e9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.6, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "axes.linewidth": 0.6,
                     "xtick.color": MUTE, "ytick.color": MUTE, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.fontsize": 5.8, "axes.labelsize": 6.6, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2})
SA, LS, CS = Path("/data/models/graybox_ws/outputs/self_audit"), Path("/data/models/graybox_ws/outputs/live_selfaudit"), Path("/data/models/graybox_ws/outputs/claude_selfaudit")
T1 = ROOT / "results/t1"


def boot_red(none, gate, groups, nb=2000, seed=0):
    """Relative reduction with a bootstrap over groups (scenario instances)."""
    df = pd.DataFrame({"g": groups, "n": none, "a": gate})
    P = df.groupby("g")[["n", "a"]].mean()
    rng = np.random.default_rng(seed)
    b = [1 - P.a.values[i].mean() / max(P.n.values[i].mean(), 1e-9) for i in (rng.integers(0, len(P), len(P)) for _ in range(nb))]
    return 1 - P.a.mean() / P.n.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def points():
    """(label, predicted, observed, lo, hi, color, kind): kind 'g12' / 'g13' preregistered predictions; 'g11b' registered, failed."""
    pts = []
    g12 = json.loads((T1 / "g12_composition_law.json").read_text())
    for a, name in (("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss"), ("qwen38", "Qwen3.8"), ("museglimmer", "Muse-Glimmer"), ("claude", "Claude Haiku 5.5")):
        r = g12[a]
        pts.append((name, r["predicted"], *r["observed"], ORANGE if a == "claude" else BLUE, "g12"))
    r = json.loads((T1 / "g11b_selfpolicing_law.json").read_text())
    for a, name in (("museglimmer", "Muse-Glimmer"), ("qwen38", "Qwen3.8")):
        pts.append((name, r[a]["predicted"], *r[a]["reduction"], MUTE, "g11b"))
    g13 = T1 / "g13_agentdojo_selfpolicing.json"
    if g13.exists():
        r = json.loads(g13.read_text())
        for a, name in (("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss")):
            if a in r and r[a].get("predicted_reduction") is not None:
                pts.append((name, r[a]["predicted_reduction"], *r[a]["reduction"], BLUE, "g13"))
    return pts


RECOG = [  # (agent, benchmark, prompt, violations flagged, safe flagged), development audits, flag = majority "no"
    ("Gemma-4", "CAPE", "plain", 0.966, 0.111), ("Muse-Glimmer", "CAPE", "plain", 0.874, 0.212), ("Qwen3.8", "CAPE", "plain", 0.473, 0.062),
    ("gpt-oss", "CAPE", "plain", 0.370, 0.116), ("Claude", "CAPE", "plain", 0.051, 0.290), ("Claude", "CAPE", "rule by rule", 0.684, 0.245),
    ("Gemma-4", "AgentDojo", "plain", 1.000, 0.020), ("gpt-oss", "AgentDojo", "plain", 0.963, 0.184)]


def recognition(ax):
    rows = RECOG[::-1]
    for i, (a, b, pr, v, s) in enumerate(rows):
        c = ORANGE if a.startswith("Claude") else BLUE
        ax.plot([s, v], [i, i], color=c, lw=1.0, alpha=0.6)
        ax.scatter([v], [i], s=22, color=c, zorder=3)
        ax.scatter([s], [i], s=22, facecolor="white", edgecolor=c, linewidth=1.0, zorder=3)
        ax.text(-0.03, i, f"{a}" + (f" ({pr})" if a.startswith("Claude") else "") + f", {b}", ha="right", va="center", fontsize=5.6, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("share of the agent's own calls it flags when asked")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    h1 = ax.scatter([], [], s=22, color=INK, label="its violations")
    h2 = ax.scatter([], [], s=22, facecolor="white", edgecolor=INK, label="its allowed calls")
    ax.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.2)
    ax.set_title("a  Does the agent recognize its own violations?", fontsize=6.8, loc="left", fontweight="bold", color=INK, pad=14)


def main():
    pts = points()
    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(6.85, 2.7), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.42})
    recognition(axa)
    law_panel(axb, pts)
    axb.set_title("b  The law predicts live prevention", fontsize=6.8, loc="left", fontweight="bold", color=INK, pad=14)
    fig2.subplots_adjust(left=0.215, right=0.985, top=0.86, bottom=0.16)
    fig2.savefig(ROOT / "paper/figures/selfpolice.pdf")
    fig2.savefig(ROOT / "results/figures/selfpolice.png", dpi=220)
    fig, ax = plt.subplots(figsize=(3.3, 1.95))
    law_panel(ax, pts)
    fig.tight_layout(pad=0.3)
    (ROOT / "paper/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "paper/figures/law.pdf")
    fig.savefig(ROOT / "results/figures/law.png", dpi=220)
    print(json.dumps([(p[0], p[6], round(float(p[1]), 3), [round(float(v), 3) for v in p[2:5]]) for p in pts], indent=0))


def law_panel(ax, pts):
    xs = np.array([0, 1.05])
    ax.fill_between(xs, xs - 0.15, xs + 0.15, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.plot(xs, xs, color=MUTE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    off = {("Gemma-4", "g12"): (0.03, -0.06, "left"), ("gpt-oss", "g12"): (0.03, -0.055, "left"), ("Qwen3.8", "g12"): (0.0, -0.17, "center"),
           ("Muse-Glimmer", "g12"): (-0.03, 0.075, "right"), ("Claude Haiku 5.5", "g12"): (0.03, 0.05, "left"),
           ("Muse-Glimmer", "g11b"): (0.0, -0.16, "center"), ("Qwen3.8", "g11b"): (0.03, -0.03, "left"),
           ("Gemma-4", "g13"): (-0.03, 0.0, "right"), ("gpt-oss", "g13"): (0.03, -0.04, "left")}
    for name, x, y, lo, hi, c, kind in pts:
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, elinewidth=0.8, capsize=1.5, alpha=0.9, zorder=2)
        if kind == "g11b":
            ax.scatter([x], [y], s=22, marker="x", color=c, linewidth=1.1, zorder=3)
        else:
            ax.scatter([x], [y], s=24, marker="s" if kind == "g13" else "o", facecolor=c, edgecolor="white", linewidth=0.6, zorder=3)
        dx, dy, ha = off.get((name, kind), (0.03, 0.0, "left"))
        tag = {"g12": "", "g13": " (AgentDojo)", "g11b": " (G11b)"}[kind]
        ax.text(x + dx, y + dy, name + tag, fontsize=5.3, color=MUTE if kind == "g11b" else INK, ha=ha, va="center")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlabel("predicted: $\\sum_k w_k\\tau_k$ ($\\tau_k$: offline flag rates)")
    ax.set_ylabel("observed live reduction")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    hs = [ax.scatter([], [], s=24, marker="o", facecolor=INK, edgecolor="white", label="G12, CAPE: preregistered, passed")]
    if any(p[6] == "g13" for p in pts):
        hs.append(ax.scatter([], [], s=24, marker="s", facecolor=INK, edgecolor="white", label="G13, AgentDojo: preregistered"))
    hs.append(ax.scatter([], [], s=22, marker="x", color=MUTE, linewidth=1.1, label="G11b: registered, failed"))
    from matplotlib.patches import Patch
    hs.append(Patch(facecolor=GRID, alpha=0.7, label="tolerance: predicted $\\pm$ 0.15"))
    ax.legend(handles=hs, loc="upper left", handletextpad=0.3, fontsize=5.2, borderaxespad=0.3)


if __name__ == "__main__":
    main()
