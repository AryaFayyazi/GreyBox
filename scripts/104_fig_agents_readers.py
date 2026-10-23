#!/usr/bin/env python
"""Figures from saved results only.
replica_agents (T17): per open agent, (a) variance of the decision state replicated from the transcript (reader vs
text), (b) a probe trained on TRUE states, applied to true vs replicated states. Gemma-4 is T17's exploratory row (T15's
development units under the same scenario-disjoint split).
small_readers (T19): transfer AUROC against reader size, with TF-IDF (the old browser monitor) for reference.
python scripts/104_fig_agents_readers.py
"""
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
F = importlib.import_module("90_figures")
RES = ROOT / "results/t1"


def replica_agents():
    r = json.loads((RES / "t17_replication_agents.json").read_text())
    rows = [("Gemma-4\n26B", r["exploratory_gemma4_same_split"])] + \
           [(n, r["agents"][k]) for n, k in (("gpt-oss\n20B", "gptoss20b"), ("Qwen3.8\n27B", "qwen38"),
                                              ("Muse Gl.\n30B", "museglimmer"))]
    x = np.arange(len(rows))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axs[0]
    ax.bar(x - 0.19, [a["r2_fresh"]["READER"] for _, a in rows], 0.36, color=F.C[0], label="reader replica")
    ax.bar(x + 0.19, [a["r2_fresh"]["TXT"] for _, a in rows], 0.36, color=F.GRID, edgecolor=F.MUTE, lw=0.6, label="text features")
    for i, (_, a) in enumerate(rows):
        ax.text(i - 0.19, a["r2_fresh"]["READER"] + 0.015, f"{a['r2_fresh']['READER']:.2f}", ha="center", fontsize=7, color=F.INK)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in rows], fontsize=6.5)
    ax.set_ylim(0, 1); ax.set_ylabel("$R^2$ of the agent's state\n(fresh scenarios)")
    ax.set_title("(a) State replicated from the transcript", loc="left", fontsize=9, color=F.INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.2)
    ax.grid(axis="y", color=F.GRID, lw=0.6); ax.set_axisbelow(True)
    ax = axs[1]
    t, rp, tx = ([a["E2_auroc"][k] for _, a in rows] for k in ("true", "READER_replica", "TXT_replica"))
    ax.bar(x - 0.26, t, 0.25, color=F.MUTE, label="probe on the TRUE state")
    ax.bar(x, rp, 0.25, color=F.C[0], label="same probe, reader replica")
    ax.bar(x + 0.26, tx, 0.25, color=F.GRID, edgecolor=F.MUTE, lw=0.6, label="same probe, text replica")
    lo = min(t + rp + tx)
    ax.set_ylim(max(0.5, np.floor((lo - 0.03) * 20) / 20), 1.0)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in rows], fontsize=6.5); ax.set_ylabel("Violation AUROC\n(within template)")
    ax.set_title("(b) White-box readout on the replica", loc="left", fontsize=9, color=F.INK)
    ax.set_yticks([v for v in np.arange(0.80, 1.001, 0.05) if v >= ax.get_ylim()[0]])
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=1.2, columnspacing=0.8)
    ax.grid(axis="y", color=F.GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); F.save(fig, "replica_agents")


SIZE = {"minilm": 0.0227, "mpnet": 0.109, "qemb06": 0.6, "q06": 0.6, "q17": 1.7, "q4": 4.0, "q8": 8.2, "q14": 14.8}
NAME = {"minilm": "MiniLM", "mpnet": "mpnet", "qemb06": "Qwen3-Emb-0.6B", "q06": "0.6B", "q17": "1.7B", "q4": "4B", "q8": "8B (shipped)", "q14": "14B"}


def small_readers():
    """T19 in the paper's common style (scripts/118's palette and fonts). One marker per agent in both panels."""
    from matplotlib.lines import Line2D
    v2 = importlib.import_module("118_paper_figures_v2")
    plt_, BLUE, ORANGE, MUTE, INK = v2.plt, v2.BLUE, v2.ORANGE, v2.MUTE, v2.INK
    r = json.loads((RES / "t19_small_readers.json").read_text())
    A = r["auroc"]
    MK = {"GPT": "o", "Claude": "D", "open agents": "s"}
    fig, axs = plt_.subplots(1, 2, figsize=(v2.FULL, 2.05), gridspec_kw={"wspace": 0.28})
    panels = ((axs[0], (("T13b", "GPT"), ("T7", "open agents")), "(a) A new benchmark (AgentDojo)", (0.3, 0.9)),
              (axs[1], (("T8b", "Claude"), ("T11", "GPT")), "(b) Closed agents on the training benchmark", (0.70, 1.01)))
    for ax, sets, title, ylim in panels:
        v2.grid(ax, "both")
        for (s, lab), alpha in zip(sets, (1.0, 0.55)):
            for fam, col in (("browser", ORANGE), ("decoder", BLUE)):
                ks = sorted((k for k in SIZE if k in A and (k in ("minilm", "mpnet", "qemb06")) == (fam == "browser")), key=lambda k: SIZE[k])
                ax.plot([SIZE[k] for k in ks], [A[k][s] for k in ks], marker=MK[lab], color=col, lw=1.1, ms=3.4, alpha=alpha, mec="white", mew=0.4)
        if "q8" in A:
            ax.annotate("shipped 8B", (SIZE["q8"], A["q8"][sets[0][0]]), xytext=(-4, 7), textcoords="offset points", fontsize=5.7,
                        color=INK, ha="right", arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.5))
        tx = sorted(((A["TXT"][s], lab) for s, lab in sets), reverse=True)
        for n, (v, lab) in enumerate(tx):
            ax.axhline(v, color=MUTE, ls="--" if n == 0 else ":", lw=0.8)
            ax.text(0.017, v + (0.004 if n == 0 else -0.004), f"TF-IDF, {lab}", fontsize=5.7, color=MUTE, ha="left", va="bottom" if n == 0 else "top")
        ax.set_xscale("log"); ax.set_xlim(0.015, 22); ax.set_ylim(*ylim)
        ax.set_xlabel("reader parameters (billions)"); ax.set_ylabel("AUROC")
        ax.set_title(title)
    h = [Line2D([], [], color=ORANGE, lw=1.1, label="browser encoders"), Line2D([], [], color=BLUE, lw=1.1, label="decoder readers (Qwen3)")] + \
        [Line2D([], [], color=MUTE, marker=MK[k], lw=0, ms=3.4, label="Claude Haiku 5.5" if k == "Claude" else k) for k in ("GPT", "Claude", "open agents")]
    fig.legend(handles=h, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, handlelength=1.4, columnspacing=1.2)
    v2.save(fig, "small_readers")


if __name__ == "__main__":
    for f in (replica_agents, small_readers):
        try:
            f()
            print("saved", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
