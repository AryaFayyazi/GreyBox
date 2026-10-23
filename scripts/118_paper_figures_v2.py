#!/usr/bin/env python
"""Paper figures, version 2, drawn at their PRINT size (ICML: text width 6.75 in, column 3.25 in), so the fonts print at
6.5-7.5 pt. From saved results and the public per-unit release only; nothing is refit.
  replication     per-component R^2 (reader vs TF-IDF), R^2 per agent, the white-box readout on the replica, and a check of
                  the conditional-mean identity Var(replica)/Var(state) = R^2 (Proposition: access and replica, b)   full width
  prevention      live prevention (violation and goal rates, 95% Wilson intervals) and the label-free cut under bursts,
                  with the bound beta/rho (Proposition: bursts)                                           full width
  generalization  monitoring under increasing shift (reader, TF-IDF, judge, agent internals)              column
  policy_labels   labels of a new policy: added back to CAPE-Bench; from open agents to a closed one   0.75 width
Palette: the validated categorical order (blue, orange, aqua, yellow) of scripts/90_figures.py.
python scripts/118_paper_figures_v2.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.stats import within_auroc  # noqa: E402

R, U, OUT = ROOT / "results", ROOT / "results/units", ROOT / "results/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTE, GRID, LIGHT = "#1b2126", "#5d6873", "#e3e6e9", "#c9ced3"
FULL, COL = 6.85, 3.3
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.6, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "axes.linewidth": 0.6,
                     "xtick.color": MUTE, "ytick.color": MUTE, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 7.2,
                     "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titleweight": "bold", "legend.frameon": False,
                     "legend.fontsize": 6.2, "axes.labelsize": 6.6, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, not Type 3, in the PDF
PCT = mtick.PercentFormatter(1.0, decimals=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def r2_per_pc(Y, P):
    Y, P = Y.astype(np.float64), P.astype(np.float64)       # the release stores float16: sums of squares overflow
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def fig_replication():
    g = np.load(U / "t15_gemma_replication.npz")
    rr, rt = r2_per_pc(g["Y_true"], g["Y_reader"]), r2_per_pc(g["Y_true"], g["Y_text"])
    assert abs(np.average(rr, weights=g["Y_true"].astype(np.float64).var(0)) - 0.664) < 0.002
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.72), gridspec_kw={"width_ratios": [1.3, 0.95, 1.05, 0.85], "wspace": 0.7})
    ax = axs[0]; grid(ax, "y")
    # (a) why a white-box probe works on the replica: the true-state probe's score, on the true state and on the replicas,
    # for every fresh step (split violins: safe left, violations right). The replica keeps the class means and shrinks the
    # spread within each class (the conditional mean; Proposition: Rao-Blackwell): separation d' rises.
    from scipy.stats import gaussian_kde
    from sklearn.metrics import roc_auc_score
    yy = g["y"].astype(bool)
    cols = [("true\nstate", "probe_true"), ("reader\nreplica", "probe_READER"), ("TF-IDF\nreplica", "probe_TXT")]
    grid_y = np.linspace(-13, 6, 400)
    for i0, (lab_, key) in enumerate(cols):
        i = 1.3 * i0
        sc_ = g[key].astype(np.float64)
        a_, b_ = sc_[yy], sc_[~yy]
        for part, side, col in ((b_, -1, MUTE), (a_, 1, ORANGE)):
            kd = gaussian_kde(part, bw_method=0.18)(grid_y)
            wdt = 0.42 * kd / kd.max()
            face = (BLUE if key == "probe_READER" else col) if side < 0 else col
            ax.fill_betweenx(grid_y, i, i + side * wdt, color=face, alpha=0.55 if side < 0 else 0.7, lw=0)
            ax.plot(i + side * wdt, grid_y, color=face, lw=0.6)
            ax.plot([i, i + side * 0.3], [np.median(part)] * 2, color=INK, lw=0.8)
        dp = (a_.mean() - b_.mean()) / np.sqrt((a_.var() + b_.var()) / 2)
        ax.text(i, 6.6, f"d$'$ {dp:.2f}", ha="center", va="bottom", fontsize=5.6,
                color=BLUE if key == "probe_READER" else INK, fontweight="bold" if key == "probe_READER" else "normal")
    ax.set(xlim=(-0.6, 3.2), ylim=(-13, 8.4), ylabel="true-state probe score (logit)")
    ax.set_xticks([0, 1.3, 2.6]); ax.set_xticklabels([c[0] for c in cols], fontsize=5.6)
    ax.set_yticks([-10, -5, 0, 5])
    ax.set_xlabel("left half: safe  |  right half: violations", fontsize=5.6, color=INK)
    ax.set_title("(a) Why probes transfer")
    t17 = json.loads((R / "t1/t17_replication_agents.json").read_text())
    names = ["Gemma", "gpt-oss", "Qwen", "Muse"]
    rd = [t17["exploratory_gemma4_same_split"]["r2_fresh"]] + [t17["agents"][a]["r2_fresh"] for a in ("gptoss20b", "qwen38", "museglimmer")]
    ax = axs[1]; grid(ax)
    x = np.arange(4)
    ax.bar(x - 0.2, [d["READER"] for d in rd], 0.38, color=BLUE, label="reader")
    ax.bar(x + 0.2, [d["TXT"] for d in rd], 0.38, color=LIGHT, edgecolor=MUTE, lw=0.4, label="TF-IDF")
    for i, d in enumerate(rd):
        ax.text(i - 0.2, d["READER"] + 0.015, f"{d['READER']:.2f}", ha="center", fontsize=5.6, color=INK)
    ax.set(xticks=x, ylim=(0, 0.92), ylabel="$R^2$, 128 components")
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(names, fontsize=5.6, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_title("(b) Four agents")
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.1)
    ax = axs[2]; grid(ax, "y")
    # (c) every recorded layer of every agent: R^2 over the FULL layer (not only the top components), reader minus text
    t15 = json.loads((R / "t1/t15_latent_replication.json").read_text())
    per = [("Gemma", t15["r2_fresh_per_layer_full_space"])] + [(n, t17["agents"][a]["r2_fresh_per_layer_full_space"])
                                                              for n, a in (("gpt-oss", "gptoss20b"), ("Qwen", "qwen38"), ("Muse", "museglimmer"))]
    shades = ["#bcd5f3", "#8db8ec", "#5c99e2", BLUE, "#1b4f91"]
    n_pos = 0
    for i, (name, d) in enumerate(per):
        Ls = sorted(d["READER"], key=int)
        for k, L in enumerate(Ls):
            gain = d["READER"][L] - d["TXT"][L]
            n_pos += gain > 0
            ax.bar(i + (k - 2) * 0.16, gain, 0.15, color=shades[k], lw=0)
    ax.axhline(0, color=MUTE, lw=0.6)
    ax.set(xticks=range(4), xlim=(-0.6, 3.6), ylim=(0, 0.13), ylabel="$R^2$ gain over TF-IDF, full layer")
    ax.set_yticks([0, 0.03, 0.06, 0.09])
    ax.set_xticklabels([n for n, _ in per], fontsize=5.6, rotation=30, ha="right", rotation_mode="anchor")
    ax.text(0.97, 0.96, f"{n_pos} of {sum(len(d['READER']) for _, d in per)} layers > 0\nbars: shallow to deep", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.7, color=INK)
    ax.set_title("(c) Every layer")
    ax = axs[3]; grid(ax, "both")
    vr = g["Y_reader"].astype(np.float64).var(0) / g["Y_true"].astype(np.float64).var(0)
    ax.plot([0, 1], [0, 1], color=MUTE, lw=0.7, ls=":")
    ax.scatter(rr, vr, s=4, color=BLUE, alpha=0.8, lw=0)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="$R^2$ of the component", ylabel="Var(replica) / Var(state)")
    ax.set_title("(d) Variance identity")
    ax.text(0.96, 0.05, "identity", ha="right", fontsize=5.6, color=MUTE, transform=ax.transAxes)
    save(fig, "replication")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, min(p, max(0.0, c - h)), max(p, c + h)


def fig_prevention():
    """All four preregistered live tests (T16, T16b, T16c, T28): violation and goal rates with 95% Wilson intervals; and
    the label-free cut under attack bursts (T8b replay)."""
    from matplotlib.patches import Patch
    a = json.loads((R / "t1/t16_live_gate.json").read_text())
    b = json.loads((R / "t1/t16b_claude_live_gate.json").read_text())
    c = json.loads((R / "t1/t16c_review_gate.json").read_text())
    d = json.loads((R / "t1/t28_recalibrated_gate.json").read_text())
    nA, nB, nC, nD = a.get("scenarios_complete", 500), b.get("scenarios_complete", 450), c.get("scenarios", 450), d["scenarios_complete"]
    arms = [(a["unsafe_episode_rate"]["ungated"], a["goal_rate"]["ungated"], nA, LIGHT),
            (a["unsafe_episode_rate"]["gated"], a["goal_rate"]["gated"], nA, BLUE),
            (b["unsafe_episode_rate"]["ungated"], b["goal_rate"]["ungated"], nB, LIGHT),
            (b["unsafe_episode_rate"]["gated"], b["goal_rate"]["gated"], nB, BLUE),
            (c["unsafe_episode_rate"]["review"], c["goal_rate"]["review"], nC, AQUA),
            (d["ungated"]["unsafe_rate"], d["ungated"]["goal_rate"], nD, LIGHT),
            (d["gated"]["unsafe_rate"], d["gated"]["goal_rate"], nD, BLUE)]
    xs = [0, 0.9, 2.2, 3.1, 4.0, 5.3, 6.2]
    groups = ((0.45, "GPT-6-luna"), (3.1, "Claude"), (5.75, "same agent,\ntold it is\nmonitored"))
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.95), gridspec_kw={"width_ratios": [1.25, 1.25, 1.0], "wspace": 0.36})
    for j, (ax, idx, lab, top) in enumerate(((axs[0], 0, "episodes with an executed violation", 0.105),
                                             (axs[1], 1, "episodes that reached the goal", 0.62))):
        grid(ax)
        for x, r in zip(xs, arms):
            p, lo, hi = wilson(round(r[idx] * r[2]), r[2])
            ax.bar(x, p, 0.78, color=r[3], edgecolor=MUTE if r[3] == LIGHT else r[3], lw=0.4, zorder=2)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color=INK, lw=0.6, capsize=1.5, capthick=0.6, zorder=3)
            ax.text(x, hi + top * 0.035, f"{100 * p:.1f}", ha="center", va="bottom", fontsize=5.9, color=INK)
        ax.set(xticks=[], xlim=(-0.6, 6.8), ylim=(0, top * 1.12))
        ax.set_ylabel(lab)
        ax.yaxis.set_major_formatter(PCT)
        for x0, name in groups:
            ax.text(x0, -0.035, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.0, color=INK)
        for xb in (1.55, 4.65):
            ax.axvline(xb, color=GRID, lw=0.6, zorder=1)
        ax.set_title("(a) Violations" if j == 0 else "(b) Goals reached")
    leg = [Patch(fc=LIGHT, ec=MUTE, lw=0.4, label="no gate"), Patch(fc=BLUE, label="block (cut re-set when told)"),
           Patch(fc=AQUA, label="hold for review")]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.36, -0.06), ncol=3, fontsize=6.0, handlelength=1.2, columnspacing=1.2)
    g = json.loads((R / "t1/t8b_calibration_exploratory.json").read_text())["gate_eval_76c"]["burst"]
    rho = sorted(float(k.split("_")[1]) for k in g)
    ax = axs[2]; grid(ax, "both")
    rr = np.linspace(0.02, 0.52, 100)
    ax.plot(rr, np.minimum(1, 0.05 / rr), color=MUTE, ls=":", lw=0.9, label=r"$\beta/\rho$ bound, moving cut")
    for key, col, mk, lab in (("rolling500", INK, "^", "moving cut, last 500"), ("rolling5000", ORANGE, "s", "moving cut, last 5,000"),
                              ("frozen_default", BLUE, "o", "frozen cut (ours)")):
        ax.plot(rho, [g[f"rate_{r:g}"][key]["caught"] for r in rho], color=col, lw=1.3 if mk == "o" else 1.0, marker=mk, ms=3.2, mec="white",
                mew=0.4, ls=(0, (3, 1.5)) if key == "rolling500" else "-", label=lab)
    ax.set(xlim=(0, 0.52), ylim=(0, 1.05), xlabel=r"share of violations in a burst, $\rho$", ylabel="violations caught at a 5% budget")
    ax.yaxis.set_major_formatter(PCT); ax.xaxis.set_major_formatter(PCT)
    ax.set_title("(c) Bursts of attacks")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", bbox_to_anchor=(0.84, -0.06), ncol=2, fontsize=6.0, handlelength=1.5, columnspacing=0.8)
    ax.set_xlabel(r"share of violations in a burst, $\rho$", labelpad=1)
    save(fig, "prevention")


def fig_generalization():
    J = lambda p: json.loads((R / p).read_text())  # noqa: E731
    t4, t5, t6, t7 = J("t1/t4_reader.json")["auroc_fresh"], J("t1/t5_family.json")["auroc"], J("t1/t6_cross_agent.json")["auroc_gptoss"], J("t1/t7_agentdojo.json")
    t8, t9, t10, t11, t13b = J("t1/t8b_claude_all_actions.json")["auroc"], J("t1/t9_text_baseline.json"), J("t1/t10_claude_unseen_families.json"), J("t1/t11_gpt_agent.json"), J("t1/t13b_gpt_agentdojo_none.json")
    tx = {k: t9[k]["auroc"]["TXT"] for k in ("T4_same_scenarios", "T5_unseen_families", "T6_gptoss", "T7_agentdojo", "T8b_claude")}
    rows = [("Same scenarios", t4["READER_act"], t4["J_act"], t4["I_dec"], tx["T4_same_scenarios"]),
            ("Unseen task families", t5["READER_family"], t5["J_act"], t5["AGENT_family"], tx["T5_unseen_families"]),
            ("Another agent (gpt-oss)", t6["READER_from_gemma"], t6["J_act"], None, tx["T6_gptoss"]),
            ("New benchmark (AgentDojo)", t7["per_actor"]["pooled"]["READER"], t7["per_actor"]["pooled"]["J_act"], None, tx["T7_agentdojo"]),
            ("Closed: Claude", t8["READER"], t8["J_act"], None, tx["T8b_claude"]),
            ("Claude, unseen policy", t10["auroc_within"]["READER_lofo"], t10["auroc_within"]["J_act"], None, t10["auroc_within"]["TXT_lofo"]),
            ("Closed: GPT-6-luna", t11["auroc_within_template"]["READER"], t11["auroc_within_template"]["J_act"], None, t11["auroc_within_template"]["TXT"]),
            ("GPT, AgentDojo (new policy)", t13b["auroc_within_suite"]["READER"], t13b["auroc_within_suite"]["J_act"], None, t13b["auroc_within_suite"]["TXT"])]
    fig, ax = plt.subplots(figsize=(COL, 2.55))
    grid(ax, "x")
    ax.axvline(0.5, color=MUTE, ls="--", lw=0.6, zorder=0.5)
    ax.text(0.51, -0.75, "chance", fontsize=5.6, color=MUTE, va="center")
    n = len(rows)
    for i, (name, r, j, w, t) in enumerate(rows[::-1]):
        ax.plot([min(j, t), max(r, t)], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
        ax.plot(t, i, "s", mfc="white", mec=MUTE, ms=3.6, mew=0.8, zorder=3)
        ax.plot(j, i, "o", color=AQUA, ms=4.4, mec="white", mew=0.6, zorder=3)
        ax.plot(r, i, "o", color=BLUE, ms=4.8, mec="white", mew=0.6, zorder=4)
        if w is not None:
            ax.plot(w, i, "D", color=ORANGE, ms=3.4, mec="white", mew=0.5, zorder=5)
        ax.text(1.012, i, f"{r:.3f}", va="center", fontsize=5.8, color=INK)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6, color=INK)
    ax.set_xlim(0.3, 1.06); ax.set_ylim(-1.1, n - 0.4)
    ax.set_xlabel("violation AUROC (within template or suite)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=BLUE, ms=4.4, mec="white", label="reader"),
          Line2D([], [], marker="s", ls="", mfc="white", mec=MUTE, ms=3.6, label="TF-IDF"),
          Line2D([], [], marker="o", ls="", color=AQUA, ms=4.2, mec="white", label="70B judge"),
          Line2D([], [], marker="D", ls="", color=ORANGE, ms=3.4, mec="white", label="agent internals")]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=4, fontsize=5.8, handletextpad=0.1, columnspacing=0.7)
    save(fig, "generalization")


def fig_policy_labels():
    J = lambda p: json.loads((R / p).read_text())  # noqa: E731
    t10, t12, t14 = J("t1/t10_new_policy_labels_exploratory.json"), J("t1/t12_new_policy_labels.json"), J("t1/t14_agentdojo_labels.json")
    fig, axs = plt.subplots(1, 2, figsize=(5.1, 1.75), gridspec_kw={"wspace": 0.38})
    ax = axs[0]; grid(ax, "both")
    xs = [int(n) for n in t10["by_n"] if int(n) <= 40]
    ax.plot(xs, [t10["by_n"][str(n)]["cons_service_restart"] for n in xs], color=ORANGE, lw=1.2, ls="--", marker="o", ms=3.2, mec="white", mew=0.4,
            label="Claude, restart policy")
    xs = [int(n) for n in t12["curve"]]
    ax.plot(xs, [t12["curve"][str(n)]["F_caught"] for n in xs], color=BLUE, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.4, label="GPT, credentials policy")
    ax.set(xlabel="labelled steps of the new policy", ylabel="caught (5% reviewed)", ylim=(-0.03, 1.05), xticks=[0, 5, 10, 20, 40])
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="lower right")
    ax.set_title("(a) A policy absent from training")
    ax = axs[1]; grid(ax, "both")
    ks = [int(k) for k in t14["curve"]]
    xp = [max(k, 30) for k in ks]
    ax.plot(xp, [t14["curve"][str(k)]["caught_benign5"] for k in ks], color=BLUE, lw=1.3, marker="o", ms=3.2, mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_xticks(xp); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
    ax.annotate(f"AUROC {t14['curve'][str(ks[0])]['auroc']:.2f}", (xp[0], t14["curve"][str(ks[0])]["caught_benign5"]), xytext=(4, -9), textcoords="offset points", fontsize=5.8, color=INK)
    ax.annotate(f"AUROC {t14['curve'][str(ks[-1])]['auroc']:.2f}\n(judge {t14['judge_auroc']:.2f})", (xp[-1], t14["curve"][str(ks[-1])]["caught_benign5"]),
                xytext=(-44, -26), textcoords="offset points", fontsize=5.8, color=INK)
    ax.set(xlabel="labelled AgentDojo calls of open agents", ylabel="injected calls caught", ylim=(-0.03, 1.05))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("(b) Prompt injection, closed GPT")
    save(fig, "policy_labels")


if __name__ == "__main__":
    fig_replication()
    fig_prevention()
    fig_generalization()
    fig_policy_labels()
    for f in ("replication", "prevention", "generalization", "policy_labels"):
        (ROOT / "paper/figures" / f"{f}.pdf").write_bytes((OUT / f"{f}.pdf").read_bytes())
    print("written and copied to paper/figures: replication, prevention, generalization, policy_labels")
