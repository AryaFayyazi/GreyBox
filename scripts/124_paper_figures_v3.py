#!/usr/bin/env python
"""Paper figures, version 3 (drawn at print size; from saved results and the public per-unit release; nothing refit).
  transfer   (a) the monitor under increasing shift as a heat map (reader, TF-IDF, 70B judge, the agent's own internals);
             (b) labels of a new policy collected on open agents, applied to a closed agent (AgentDojo);
             (c) a closed agent told that it is monitored: score laws of both arms, the frozen cut, and the shift
                 bound of Proposition (shift)                                                          full width
python scripts/124_paper_figures_v3.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
v2 = importlib.import_module("118_paper_figures_v2")
plt, R, U, OUT = v2.plt, v2.R, v2.U, v2.OUT
BLUE, ORANGE, AQUA, INK, MUTE, GRID, LIGHT, PCT = v2.BLUE, v2.ORANGE, v2.AQUA, v2.INK, v2.MUTE, v2.GRID, v2.LIGHT, v2.PCT
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def shift_rows():
    J = lambda p: json.loads((R / p).read_text())  # noqa: E731
    t4, t5, t6, t7 = J("t1/t4_reader.json")["auroc_fresh"], J("t1/t5_family.json")["auroc"], J("t1/t6_cross_agent.json")["auroc_gptoss"], J("t1/t7_agentdojo.json")
    t8, t9, t11, t13b = J("t1/t8b_claude_all_actions.json")["auroc"], J("t1/t9_text_baseline.json"), J("t1/t11_gpt_agent.json"), J("t1/t13b_gpt_agentdojo_none.json")
    tx = {k: t9[k]["auroc"]["TXT"] for k in ("T4_same_scenarios", "T5_unseen_families", "T6_gptoss", "T7_agentdojo", "T8b_claude")}
    return [("same scenarios", t4["READER_act"], tx["T4_same_scenarios"], t4["J_act"], t4["I_dec"]),
            ("unseen task families", t5["READER_family"], tx["T5_unseen_families"], t5["J_act"], t5["AGENT_family"]),
            ("another open agent", t6["READER_from_gemma"], tx["T6_gptoss"], t6["J_act"], None),
            ("closed: Claude", t8["READER"], tx["T8b_claude"], t8["J_act"], None),
            ("closed: GPT-6-luna", t11["auroc_within_template"]["READER"], t11["auroc_within_template"]["TXT"], t11["auroc_within_template"]["J_act"], None),
            ("new benchmark (AgentDojo)", t7["per_actor"]["pooled"]["READER"], tx["T7_agentdojo"], t7["per_actor"]["pooled"]["J_act"], None),
            ("closed GPT on AgentDojo", t13b["auroc_within_suite"]["READER"], t13b["auroc_within_suite"]["TXT"], t13b["auroc_within_suite"]["J_act"], None)]


def fig_transfer():
    fig = plt.figure(figsize=(v2.FULL, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.8, 1.05], wspace=0.62)
    # (a) heat map
    ax = fig.add_subplot(gs[0])
    rows = shift_rows()
    cols = ["reader", "TF-IDF", "70B\njudge", "agent's\ninternals"]
    M = np.array([[np.nan if v is None else v for v in r[1:]] for r in rows], float)
    cmap = LinearSegmentedColormap.from_list("gb", ["#fbe3d9", "#ffffff", "#d6e6f8", "#7fb0e8", BLUE])
    ax.imshow(np.where(np.isnan(M), 0.5, M), cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        best = np.nanargmax(M[i])
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f3f4f5", edgecolor="white", hatch="////", lw=0))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.2, color=MUTE)
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}".lstrip("0") if v < 1 else "1.00", ha="center", va="center", fontsize=6.0,
                    color="white" if v > 0.9 else INK, fontweight="bold" if j == best else "normal")
            if j == best:
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, lw=0.9))
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=5.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=5.9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_title("(a) AUROC under increasing shift", pad=4)
    ax.axhline(2.5, color=INK, lw=0.7); ax.axhline(4.5, color=INK, lw=0.7)
    ax.text(3.62, 1.0, "open", rotation=270, va="center", fontsize=5.4, color=MUTE)
    ax.text(3.62, 3.5, "closed", rotation=270, va="center", fontsize=5.4, color=MUTE)
    ax.text(3.62, 5.5, "new", rotation=270, va="center", fontsize=5.4, color=MUTE)
    # (b) labels from open agents applied to a closed agent
    ax = fig.add_subplot(gs[1]); v2.grid(ax, "both")
    t14 = json.loads((R / "t1/t14_agentdojo_labels.json").read_text())
    ks = [int(k) for k in t14["curve"]]
    xp = np.array([max(k, 30) for k in ks], float)
    yc = np.array([t14["curve"][str(k)]["caught_benign5"] for k in ks])
    au = [t14["curve"][str(k)]["auroc"] for k in ks]
    ax.fill_between(xp, 0, yc, color=BLUE, alpha=0.12, lw=0)
    ax.plot(xp, yc, color=BLUE, lw=1.4, marker="o", ms=4.2, mec="white", mew=0.6, zorder=4)
    for x, y, a in zip(xp, yc, au):
        ax.text(x, y + 0.06, f"{a:.2f}", ha="center", fontsize=5.6, color=BLUE)
    ax.axhline(0.05, color=MUTE, lw=0.6, ls=":")
    ax.text(xp[0], 0.085, "5% cut", fontsize=5.2, color=MUTE)
    ax.set_xscale("log"); ax.set_xticks(xp); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
    ax.set(xlabel="labelled AgentDojo calls\nof open agents", ylabel="injections caught", ylim=(0, 1.13))
    ax.yaxis.set_major_formatter(PCT)
    ax.text(0.97, 0.03, f"labels: AUROC\n(judge {t14['judge_auroc']:.2f})", transform=ax.transAxes, fontsize=5.3, color=MUTE, va="bottom", ha="right")
    ax.set_title("(b) Labels of a new policy")
    # (c) a closed agent told it is monitored
    ax = fig.add_subplot(gs[2]); v2.grid(ax, "y")
    d = pd.read_csv(U / "t24_monitor_aware.csv.gz")
    t24 = json.loads((R / "t1/t24_monitor_aware.json").read_text())
    lg = lambda p: np.log10(np.clip(p, 1e-13, 1))  # noqa: E731
    cut = lg(t24["cut_frozen_from_t8b"])
    bins = np.linspace(-13, 0, 66)
    for arm, col, lab in (("plain", MUTE, "not told"), ("aware", BLUE, "told it is monitored")):
        s = d[(d.arm == arm) & (d.y == 0)].p
        h, e = np.histogram(lg(s), bins=bins, density=True)
        ax.stairs(h, e, color=col, lw=1.0, fill=arm == "aware", alpha=0.25 if arm == "aware" else 1.0)
        ax.stairs(h, e, color=col, lw=1.0, label=f"safe actions, {lab}")
    for arm, y0, mk in (("plain", -0.027, "o"), ("aware", -0.06, "o")):
        v = lg(d[(d.arm == arm) & (d.y == 1)].p)
        jit = np.random.default_rng(0).uniform(-0.009, 0.009, len(v))
        ax.scatter(v, np.full(len(v), y0) + jit, s=5, marker=mk, facecolor=ORANGE if arm == "aware" else "white", edgecolor=ORANGE, lw=0.5, zorder=5,
                   label=f"violations, {'told' if arm == 'aware' else 'not told'}")
    ax.axvline(cut, color=INK, lw=0.9, ls=(0, (3, 1.6)))
    ax.text(cut + 0.15, 0.2, "frozen\ncut", fontsize=5.4, color=INK, va="top")
    p, a = t24["plain"], t24["aware"]
    ax.text(0.02, 0.98, f"caught {p['caught_frozen_cut']:.0%} → {a['caught_frozen_cut']:.0%}\nescalated {p['escalated_frozen_cut']:.1%} → {a['escalated_frozen_cut']:.1%}",
            transform=ax.transAxes, fontsize=5.4, color=INK, va="top")
    ax.set(xlabel="reader score, log$_{10}$ p", ylabel="density", xlim=(-13, 0), ylim=(-0.08, 0.3))
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=5.3, handlelength=1.2, columnspacing=0.8)
    ax.set_title("(c) Agent told it is monitored")
    v2.save(fig, "transfer")


if __name__ == "__main__":
    fig_transfer()
    (ROOT / "paper/figures/transfer.pdf").write_bytes((OUT / "transfer.pdf").read_bytes())
    print("written and copied to paper/figures: transfer")
