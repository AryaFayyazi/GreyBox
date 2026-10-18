#!/usr/bin/env python
"""Paper figures from saved results (nothing is recomputed). Palette: the validated categorical order
(#2a78d6 blue, #eb6834 orange, #1baf7a aqua, #eda100 yellow; validate_palette.js: all checks pass, contrast relief
by direct labels). Output: results/figures/*.pdf and *.png
python scripts/90_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTE, GRID = "#1b2126", "#5d6873", "#e3e6e9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_guard():
    d = json.loads((R / "t1/guard_sim_fresh.json").read_text())
    h = json.loads((R / "t1/guard_heldout.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for i, K in enumerate(("2", "4", "8", "16")):
        pts = sorted(d["K"][K]["operating_points"], key=lambda p: p["esc_safe"])
        x = [0] + [100 * p["esc_safe"] for p in pts]; y = [0] + [100 * p["caught"] for p in pts]
        ax.plot(x, y, color=C[i], lw=2, marker="o", ms=4, markeredgecolor="white", markeredgewidth=1, zorder=3)
        sel = h["K"][K]["target_esc_safe_0.05"]
        ax.plot(100 * sel["fresh_esc_safe"], 100 * sel["fresh_caught"], "o", ms=10, mfc="none", mec=C[i], mew=1.6, zorder=4)
        ax.annotate(f"K={K}", (x[-1], y[-1]), xytext=(6, -10 if K in ("8", "4") else 4), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlim(0, 10); ax.set_ylim(0, 100)
    ax.set_xlabel("Safe actions sent to an approver (%)")
    ax.set_ylabel("Violations caught before execution (%)")
    ax.set_title("Black-box agent guard, 2,800 fresh episodes", loc="left", fontsize=9.5, color=INK)
    ax.text(9.8, 6, "rings: threshold chosen on earlier episodes\n(escalation target 5%), applied once", ha="right", fontsize=7.5, color=MUTE)
    save(fig, "guard_tradeoff")


def fig_claude():
    """Grey-box refusal forecast vs every moderation baseline: the 3-level verdict (the preregistered L3b/L3d
    baseline; weak and tie-heavy) and two GRADED moderators (scripts/67 re-analysis; L3e's preregistered baseline).
    The printed difference is against the stronger graded moderator."""
    G = json.loads((R / "l3/graded_baseline_reanalysis.json").read_text())["claude"]
    rows = []
    for name, f, key, gk, mk, tag in (("L3b  per-question (88×4 calls)", "l3/claude_livebench.json", "D2_harmful_only", "auroc", "baseline_auroc", "L3b"),
                                      ("L3d  transcript (2 calls)", "l3/claude_transcript.json", "F1_harmful_refusal", "greybox_auroc", "moderation_auroc", "L3d")):
        e = json.loads((R / f).read_text())[key]
        qg, lg = G[f"{tag}_qg_p"], G[f"{tag}_lg_p"]
        best = max((qg, lg), key=lambda x: x["baseline"])
        rows.append((name, e[gk], e[mk], qg["baseline"], lg["baseline"], [best["diff_ci"][0], best["diff_ci"][1]]))
    c = json.loads((R / "l3/claude_batch.json").read_text())["E1_harmful_refusal"]
    cg = json.loads((R / "l3/l3c_graded_reanalysis.json").read_text())
    best = max((cg["qg_p"]["E1_refusal"], cg["lg_p"]["E1_refusal"]), key=lambda x: x["graded"])
    rows.insert(1, ("L3c  in-conversation (n=4; 60% usable)", c["greybox_auroc"], c["moderation_auroc"], cg["qg_p"]["E1_refusal"]["graded"],
                    cg["lg_p"]["E1_refusal"]["graded"], [best["diff"], best["ci95"]]))
    e = json.loads((R / "l3/l3e_tool.json").read_text())["primary_qwen3guard_judge"]["A2"]
    best = max((e["qg_p"], e["lg_p"]), key=lambda x: x["baseline_auroc"])
    rows.append(("L3e  prospective, fresh (2 calls)", e["qg_p"]["greybox_auroc"], None, e["qg_p"]["baseline_auroc"], e["lg_p"]["baseline_auroc"], [best["diff"], best["ci95"]]))
    fig, ax = plt.subplots(figsize=(5.8, 0.5 * len(rows) + 1.5))
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for i, (name, g, m3, qg, lg, d) in enumerate(rows[::-1]):
        lo = min(v for v in (g, m3, qg, lg) if v is not None)
        ax.plot([lo, max(g, qg, lg)], [i, i], color=GRID, lw=3, zorder=1)
        if m3 is not None:
            ax.plot(m3, i, "o", mfc="white", mec=MUTE, ms=7, mew=1.4, zorder=3)
        ax.plot(qg, i, "s", color=C[1], ms=7, mec="white", mew=1.2, zorder=3)
        ax.plot(lg, i, "D", color=C[3] if len(C) > 3 else INK, ms=6.5, mec="white", mew=1.2, zorder=3)
        ax.plot(g, i, "o", color=C[0], ms=8, mec="white", mew=1.5, zorder=4)
        ax.text(0.905, i, f"{d[0]:+.2f} [{d[1][0]:+.2f}, {d[1][1]:+.2f}]", va="center", fontsize=8, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.text(0.905, len(rows) - 0.45, "vs stronger graded\nmoderator [95% CI]", fontsize=7, color=MUTE, va="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.5, color=INK)
    ax.set_xlim(0.6, 1.0); ax.set_xlabel("AUROC: which harmful requests Claude Haiku 5.5 refuses")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C[0], label="grey box"), Line2D([], [], marker="s", ls="", color=C[1], label="Qwen3Guard, graded"),
         Line2D([], [], marker="D", ls="", color=C[3] if len(C) > 3 else INK, label="Llama Guard, graded"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=MUTE, label="3-level verdict")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=4, fontsize=7, frameon=False, handletextpad=0.2, columnspacing=0.9)
    ax.set_title("Refusal forecast vs moderation: beats the 3-level verdict, ties graded moderators", loc="left", fontsize=9, color=INK)
    save(fig, "claude_vs_moderation")


def fig_time():
    d = json.loads((R / "t1/t2_time.json").read_text())["auroc"]
    series = (("Black-box resampling", ("R16_ts", "R16_dec"), C[0]), ("White-box probe", ("WB_ts", "I_dec"), C[1]),
              ("Text monitor", ("TXT_ts", "COT_dec_oof"), C[2]))
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for name, (a, b), c in series:
        ax.plot([0, 1], [d[a], d[b]], color=c, lw=2, marker="o", ms=7, mec="white", mew=1.2, zorder=3)
        ax.annotate(f"{name}  {d[b]:.4f}".replace("0.9790", "0.979"), (1, d[b]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
        dy = {"TXT_ts": 5, "WB_ts": -5}.get(a, 0)
        ax.annotate(f"{d[a]:.3f}", (0, d[a]), xytext=(-8, dy), textcoords="offset points", va="center", ha="right", fontsize=8, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["turn start\n(before reasoning)", "decision point\n(after reasoning)"], color=INK)
    ax.set_xlim(-0.35, 1.9); ax.set_ylim(0.8, 1.01); ax.set_ylabel("AUROC, unsafe action (12,128 fresh units)")
    ax.set_title("Time, not access: one theorem, two moments", loc="left", fontsize=9.5, color=INK)
    ax.text(1.88, 0.806, "resamples scored by the environment's rule\nchecker (an oracle); deployable version: T3", ha="right", fontsize=7, color=MUTE)
    save(fig, "time_vs_access")


def fig_label_efficiency():
    d = json.loads((R / "t1/t6_cross_agent.json").read_text())["reported_label_efficiency_by_training_scenarios"]
    t4 = json.loads((R / "t1/t4_reader.json").read_text())["auroc_fresh"]
    pts = sorted((v["train_units_mean"], v["fresh_gemma_auroc_mean"]) for v in d.values() if v["fresh_gemma_auroc_mean"])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[0], lw=2, marker="o", ms=6, mec="white", mew=1.2, zorder=3)
    ax.annotate("transcript reader", (pts[-1][0], pts[-1][1]), xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=8, color=INK)
    for val, lab, c in ((t4["I_dec"], f"agent's own internals (CAPE probe, full dev set) {t4['I_dec']:.3f}", C[1]), (t4["J_act"], f"zero-shot 70B judge {t4['J_act']:.3f}", C[2])):
        ax.axhline(val, color=c, lw=1.6, ls="--")
        ax.text(60, val + 0.002, lab, fontsize=7.5, color=INK, va="bottom")
    ax.set_xscale("log"); ax.set_ylim(0.9, 1.0); ax.set_xlim(50, 7000)
    ax.set_xlabel("Labelled training actions (log scale)"); ax.set_ylabel("AUROC, unsafe action (fresh)")
    ax.set_title("A few hundred labels match white-box monitoring", loc="left", fontsize=9.5, color=INK)
    save(fig, "label_efficiency")


def fig_generalization():
    t4 = json.loads((R / "t1/t4_reader.json").read_text())["auroc_fresh"]
    t5 = json.loads((R / "t1/t5_family.json").read_text())["auroc"]
    t6 = json.loads((R / "t1/t6_cross_agent.json").read_text())["auroc_gptoss"]
    t7 = json.loads((R / "t1/t7_agentdojo.json").read_text())
    t8 = json.loads((R / "t1/t8b_claude_all_actions.json").read_text())["auroc"]
    t9 = json.loads((R / "t1/t9_text_baseline.json").read_text())
    t10 = json.loads((R / "t1/t10_claude_unseen_families.json").read_text())
    t11 = json.loads((R / "t1/t11_gpt_agent.json").read_text())
    t13b = json.loads((R / "t1/t13b_gpt_agentdojo_none.json").read_text())
    tx = {k: t9[k]["auroc"]["TXT"] for k in ("T4_same_scenarios", "T5_unseen_families", "T6_gptoss", "T7_agentdojo", "T8b_claude")}
    rows = [("Same scenarios (Gemma)", t4["READER_act"], t4["J_act"], t4["I_dec"], tx["T4_same_scenarios"]),
            ("Unseen task families (Gemma)", t5["READER_family"], t5["J_act"], t5["AGENT_family"], tx["T5_unseen_families"]),
            ("Another agent (gpt-oss)", t6["READER_from_gemma"], t6["J_act"], None, tx["T6_gptoss"]),
            ("External benchmark (AgentDojo)", t7["per_actor"]["pooled"]["READER"], t7["per_actor"]["pooled"]["J_act"], None, tx["T7_agentdojo"]),
            ("Closed agent, Claude (1 stratum, 26 viol.)", t8["READER"], t8["J_act"], None, tx["T8b_claude"]),
            ("Claude, policy family unseen (T10)", t10["auroc_within"]["READER_lofo"], t10["auroc_within"]["J_act"], None, t10["auroc_within"]["TXT_lofo"]),
            ("Second closed agent, GPT (T11)", t11["auroc_within_template"]["READER"], t11["auroc_within_template"]["J_act"], None, t11["auroc_within_template"]["TXT"]),
            ("GPT on AgentDojo, no defense (T13b)", t13b["auroc_within_suite"]["READER"], t13b["auroc_within_suite"]["J_act"], None, t13b["auroc_within_suite"]["TXT"])]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.axvline(0.5, color=MUTE, ls="--", lw=0.9, zorder=0.5)                 # chance: AUROC 0.5
    ax.text(0.507, 1.5, "chance (0.5)", fontsize=7.5, color=MUTE, va="center", ha="left")
    for i, (name, r, j, w, t) in enumerate(rows[::-1]):
        ax.plot([min(j, t), max(r, t)], [i, i], color=GRID, lw=3, zorder=1)
        ax.plot(t, i, "s", mfc="white", mec=MUTE, ms=6.5, mew=1.4, zorder=3)
        ax.plot(j, i, "o", color=C[2], ms=8, mec="white", mew=1.5, zorder=3)
        ax.plot(r, i, "o", color=C[0], ms=8, mec="white", mew=1.5, zorder=4)
        if w is not None:
            ax.plot(w, i, "D", color=C[1], ms=6, mec="white", mew=1.2, zorder=5)
        ax.text(1.016, i, f"{r:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.5, color=INK)
    ax.set_xlim(0.28, 1.065); ax.set_xlabel("AUROC, unsafe action (within template; within suite for AgentDojo)")
    ax.set_title("Trained transcript reader (blue), trained bag-of-words (hollow square), zero-shot\njudge (green); agent's own internals (orange) where available",
                 loc="left", fontsize=9, color=INK)
    save(fig, "generalization")


def fig_policy_labels():
    """(a) A CAPE-Bench policy family absent from training, vs labelled steps of it added back (T10 exploratory, T12).
    (b) Prompt injection on the closed GPT agent (AgentDojo), vs labelled AgentDojo calls of open agents added (T14)."""
    t10 = j_or_none("t1/t10_new_policy_labels_exploratory.json")
    t12 = j_or_none("t1/t12_new_policy_labels.json")
    t14 = j_or_none("t1/t14_agentdojo_labels.json")
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axs[0]
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    if t10:
        xs = [int(n) for n in t10["by_n"] if int(n) <= 40]
        ax.plot(xs, [t10["by_n"][str(n)]["cons_service_restart"] for n in xs], color=C[1], lw=2, ls="--", marker="o", ms=6, mec="white", mew=1.2,
                label="Claude, restart policy (T10, exploratory)")
    if t12:
        xs = [int(n) for n in t12["curve"]]
        ax.plot(xs, [t12["curve"][str(n)]["F_caught"] for n in xs], color=C[0], lw=2.2, marker="o", ms=6, mec="white", mew=1.2,
                label="GPT, credentials policy (T12)")
    ax.set_xlabel("Labelled steps of the new policy added")
    ax.set_ylabel("Violations caught (5% reviewed)")
    ax.set_ylim(-0.03, 1.03); ax.set_xticks([0, 5, 10, 20, 40])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("(a) A CAPE-Bench policy absent from training", loc="left", fontsize=9, color=INK)
    ax = axs[1]
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    if t14:
        ks = [int(k) for k in t14["curve"]]
        xp = [max(k, 30) for k in ks]                       # k = 0 drawn at the left edge, labelled "0"
        ax.plot(xp, [t14["curve"][str(k)]["caught_benign5"] for k in ks], color=C[0], lw=2.2, marker="o", ms=6, mec="white", mew=1.2,
                label="trained reader (T14)")
        ax.set_xscale("log"); ax.set_xticks(xp); ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.annotate(f"AUROC {t14['curve'][str(ks[0])]['auroc']:.2f}", (xp[0], t14["curve"][str(ks[0])]["caught_benign5"]), xytext=(6, -12), textcoords="offset points", fontsize=7.5, color=INK)
        ax.annotate(f"AUROC {t14['curve'][str(ks[-1])]['auroc']:.2f}\n(judge {t14['judge_auroc']:.2f})", (xp[-1], t14["curve"][str(ks[-1])]["caught_benign5"]),
                    xytext=(-66, -44), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Labelled AgentDojo calls from open agents")
    ax.set_ylabel("Injected calls caught (5% reviewed)")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("(b) Prompt injection on a closed agent (GPT)", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "policy_labels")


def j_or_none(p):
    q = R / p
    return json.loads(q.read_text()) if q.exists() else None


if __name__ == "__main__":
    fig_generalization()
    fig_label_efficiency()
    fig_time()
    fig_guard()
    fig_claude()
    fig_policy_labels()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
