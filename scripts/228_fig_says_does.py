#!/usr/bin/env python
"""Figure: what a closed agent says about its next action vs what its policy does (G46; GPT-6-luna's G40 for reference).
Within template|tool|position AUROC with 95% cluster-bootstrap intervals (over scenarios), for each forecaster read before
the call exists. Writes paper/figures/says_does.pdf (one column, 3.25 in)."""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, INK, MUTE, GRID = "#2A78D6", "#EB6834", "#1BAF7A", "#1B2126", "#5D6873", "#E6E9ED"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": MUTE, "axes.linewidth": 0.6,
                     "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def g46_rows():
    G = _mod("g46", "scripts/226_g46_rollout_forecast.py")          # sets sys.path for graybox
    from graybox.stats import within_auroc
    U, _ = G.units()
    key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
    own = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G.OWN))}
    oth = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G.OTHER))} if G.OTHER.exists() else {}
    if len(oth) < len(U):                       # the other model's draws are drawn only when complete (no partial look)
        oth = {}
    ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    m = np.array([k in own and (not oth or k in oth) for k in ks])
    U, ks = U[m].reset_index(drop=True), [k for k, x in zip(ks, m) if x]
    S = {"report": U.report.values, "own": np.array([G.forecast(own[k]) for k in ks]),
         "ceiling": np.array([G.ceiling(own[k]) for k in ks])}
    if oth:
        S["other"] = np.array([G.forecast(oth[k]) for k in ks])
    y, st = U.y.values, (U.stratum + "|" + U.position.astype(str)).values
    cl = U.scenario_id.values
    us = np.unique(cl)
    rng = np.random.default_rng(4600)
    out = {}
    for name, s in S.items():
        b = []
        for _ in range(1000):
            i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
            b.append(within_auroc(y[i], s[i], st[i]))
        out[name] = (float(within_auroc(y, s, st)), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5)))
    return out, int(len(U)), int(y.sum())


def main():
    r, n, v = g46_rows()
    g40 = json.loads((ROOT / "results/t1/g40_gpt_precall_confirm.json").read_text())
    c1 = g40["C1_within_template_tool_position"]
    rows = [("GPT-6-luna:\nits self-report", c1[0], c1[1], c1[2], BLUE, "o"),
            ("Claude Haiku 5.5:\nits self-report", *r["report"], ORANGE, "o")]
    if "other" in r:
        rows.append(("Gemma-4 sampling\nin Claude's seat", *r["other"], MUTE, "s"))
    rows += [("Claude's own sampled\nactions, own answers", *r["own"], ORANGE, "D"),
             ("ceiling: its sampled\nactions, checker", *r["ceiling"], INK, "|")]
    fig, ax = plt.subplots(figsize=(3.25, 0.33 * len(rows) + 0.5))
    ys = np.arange(len(rows))[::-1]
    for yy, (lab, m, lo, hi, c, mk) in zip(ys, rows):
        ax.plot([lo, hi], [yy, yy], color=c, lw=1.4, alpha=0.55, solid_capstyle="round")
        face = "white" if mk == "o" and "self-report" in lab and "Claude" in lab else c
        ax.scatter([m], [yy], s=34 if mk != "|" else 60, marker=mk, facecolor=face, edgecolor=c, lw=1.1, zorder=3)
        ax.text(m, yy + 0.22, f"{m:.2f}", ha="center", va="bottom", fontsize=6.0, color=INK)
    ax.axvline(0.5, color=MUTE, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.505, ys[-1] - 0.55, "chance", fontsize=6.0, color=MUTE, va="center")
    ax.set_yticks(ys, [r_[0] for r_ in rows], fontsize=6.2)
    ax.set_xlim(0.25, 1.0); ax.set_ylim(ys[-1] - 0.8, ys[0] + 0.7)
    ax.set_xlabel("AUROC within position (95% CI)", fontsize=6.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.text(0.02, 0.985, f"Claude Haiku 5.5: {n:,} calls, {v} violations, 492 fresh episodes", fontsize=6.0, color=MUTE, va="top")
    fig.subplots_adjust(left=0.36, right=0.96, top=0.90, bottom=0.2)
    fig.savefig(ROOT / "paper/figures/says_does.pdf")
    fig.savefig(ROOT / "results/figures/says_does.png", dpi=220)
    print(json.dumps({k: [round(x, 4) for x in v_] for k, v_ in r.items()}))


if __name__ == "__main__":
    main()
