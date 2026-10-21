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
    pts = []
    A = pd.read_parquet(SA / "gemma4_clean_sample.parquet")
    A = A[np.isfinite(A.s)]
    A["tm"] = A.template.str.split("|").str[0]
    g9 = json.loads((T1 / "g9_live_self_policing.json").read_text())["arms"]["gate@1.00"]["relative_violation_reduction"]
    pts.append(("Gemma-4 (G9)", (A.s[A.y == 1] >= 1 - 1e-9).mean(), *g9, BLUE, True))
    d = pd.read_parquet(LS / "gemma4_v4fresh_g9_c100.parquet")
    tau = A[A.y == 1].groupby("tm").s.apply(lambda x: (x >= 1 - 1e-9).mean())
    for tm, e in d.groupby("template"):
        none, gate = e[e.arm == "none"].set_index(["instance_id", "seed"]).unsafe_episode, e[e.arm == "gate"].set_index(["instance_id", "seed"]).unsafe_episode
        if none.sum() < 10 or tm not in tau.index:
            continue
        j = pd.concat([none.rename("n"), gate.rename("a")], axis=1).dropna().reset_index()
        pts.append((None, tau[tm], *boot_red(j.n.astype(float), j.a.astype(float), j.instance_id), BLUE, False))
    g11 = T1 / "g11b_selfpolicing_law.json"
    if g11.exists():
        r = json.loads(g11.read_text())
        for a, name, c in (("museglimmer", "Muse-Glimmer (G11b)", AQUA), ("qwen38", "Qwen3.8 (G11b)", YELLOW)):
            pts.append((name, r[a]["predicted"], *r[a]["reduction"], c, True))
    g10 = json.loads((T1 / "g10_claude_self_policing.json").read_text())
    D = pd.DataFrame([json.loads(x) for x in open(CS / "offline_dev_checklist.jsonl")])
    D = D[D.s.notna() & ~D.tool.isin({"list_files", "read_file", "open_mock_url"}) & (D.y == 1)]
    for key, m, name in (("decidable", D.template != "tp_campaign_test", "Claude, transcript decides (G10)"),
                         ("unseen_fact", D.template == "tp_campaign_test", "Claude, unchecked fact (G10)")):
        pts.append((name, (D.s[m] >= 0.67).mean(), *g10[key]["reduction"], ORANGE, False))
    return pts


def main():
    pts = points()
    fig, ax = plt.subplots(figsize=(3.3, 2.55))
    ax.plot([0, 1], [0, 1], color=MUTE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.30, 0.335, "law: reduction = flag rate", ha="left", fontsize=5.6, color=MUTE, rotation=37, rotation_mode="anchor")
    LAB = {"Gemma-4 (G9)": (0.985, 0.655, "right"), "Claude, transcript decides (G10)": (0.64, 0.775, "right"),
           "Claude, unchecked fact (G10)": (0.175, 0.085, "left"), "Muse-Glimmer (G11b)": (None, None, "right"),
           "Qwen3.8 (G11b)": (None, None, "left")}
    for name, x, y, lo, hi, c, conf in pts:
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, elinewidth=0.8, capsize=1.5, alpha=0.9 if conf else 0.55, zorder=2)
        ax.scatter([x], [y], s=26 if conf else 14, facecolor=c if conf else "white", edgecolor=c, linewidth=1.0, zorder=3)
        if name:
            lx, ly, ha = LAB.get(name, (None, None, "left"))
            lx = x + (0.025 if ha == "left" else -0.025) if lx is None else lx
            ly = y + 0.02 if ly is None else ly
            ax.text(lx, ly, name, fontsize=5.4, color=INK, ha=ha)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("own violating calls flagged in an offline audit (development)")
    ax.set_ylabel("live reduction in violating episodes")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    h1 = ax.scatter([], [], s=26, facecolor=INK, edgecolor=INK, label="preregistered")
    h2 = ax.scatter([], [], s=14, facecolor="white", edgecolor=INK, label="exploratory (Gemma-4 per template; Claude)")
    ax.legend(handles=[h1, h2], loc="upper left", handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    (ROOT / "paper/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "paper/figures/law.pdf")
    fig.savefig(ROOT / "results/figures/law.png", dpi=220)
    print(json.dumps([(p[0], round(float(p[1]), 3), [round(float(v), 3) for v in p[2:5]]) for p in pts], indent=0))


if __name__ == "__main__":
    main()
