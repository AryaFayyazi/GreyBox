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
    r = json.loads((RES / "t19_small_readers.json").read_text())
    A = r["auroc"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    for ax, sets, title in ((axs[0], (("T13b", "GPT on AgentDojo"), ("T7", "open agents on AgentDojo")), "(a) A new kind of task (AgentDojo)"),
                            (axs[1], (("T8b", "Claude on CAPE-Bench"), ("T11", "GPT on CAPE-Bench")), "(b) Closed agents, the training benchmark")):
        for (s, lab), mk in zip(sets, ("o", "s")):
            for fam, col in (("browser", F.C[1]), ("decoder", F.C[0])):
                ks = [k for k in SIZE if k in A and (k in ("minilm", "mpnet", "qemb06")) == (fam == "browser")]
                ks.sort(key=lambda k: SIZE[k])
                ax.plot([SIZE[k] for k in ks], [A[k][s] for k in ks], marker=mk, color=col, lw=1.2, ms=4,
                        label=f"{'browser' if fam == 'browser' else 'decoder'} readers, {lab}" if fam == "decoder" or s == sets[0][0] else None,
                        alpha=1 if s == sets[0][0] else 0.55)
            ax.axhline(A["TXT"][s], color=F.MUTE, ls="--" if s == sets[0][0] else ":", lw=0.9)
            ax.text(0.021, A["TXT"][s] + 0.006, f"TF-IDF, {lab}", fontsize=6.5, color=F.MUTE)
        ax.set_xscale("log"); ax.set_xlabel("Reader parameters (billions)"); ax.set_ylabel("AUROC")
        ax.set_title(title, loc="left", fontsize=9, color=F.INK); ax.grid(color=F.GRID, lw=0.6); ax.set_axisbelow(True)
        ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    fig.tight_layout(); F.save(fig, "small_readers")


if __name__ == "__main__":
    for f in (replica_agents, small_readers):
        try:
            f()
            print("saved", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
