#!/usr/bin/env python
"""Figure: the observable latent geometry, training-free (from the G-series result files only; nothing is fit here).
  (a) G1: per open model, Spearman agreement with its own hidden geometry: answer geometry (exact, and 8 samples) vs
      the text geometry, 95% bootstrap intervals
  (b) G1: agreement vs number of questions (Theorem 4: rises and saturates)
  (c) G3: exact recovery from log-probabilities: 1 - Pearson of recovered vs true cosines against N / (d(d+1)/2),
      with the two baselines without the sphere constraint
  (d) G2: Claude Haiku 5.5's observable geometry (classical MDS of its debiased Fisher-Rao distances), by source
python scripts/130_infogeom_figures.py [draft=1 uses the dev dry run and the seed-100 pilot]
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
v2 = importlib.import_module("118_paper_figures_v2")
plt, R, OUT = v2.plt, v2.R, v2.OUT
BLUE, ORANGE, AQUA, YELLOW, INK, MUTE, GRID, LIGHT = v2.BLUE, v2.ORANGE, v2.AQUA, v2.YELLOW, v2.INK, v2.MUTE, v2.GRID, v2.LIGHT
NAMES = {"gemma3_4b": "Gemma-3 4B", "gptoss20b": "gpt-oss 20B", "llama31_8b": "Llama-3.1 8B", "mistral7b": "Mistral 7B",
         "olmo3_7b": "OLMo-3 7B", "phi4mini": "Phi-4 mini", "qwen25_7b": "Qwen2.5 7B", "qwen3_8b": "Qwen3 8B"}
SRC = {"harmful": ("harmful request", ORANGE), "benign": ("benign request", BLUE), "sst2": ("review (SST-2)", AQUA),
       "boolq": ("question (BoolQ)", YELLOW), "gsm8k": ("math (GSM8K)", "#8a5cd1"), "wiki": ("encyclopedia", MUTE)}


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    draft = "draft=1" in sys.argv
    g1 = json.loads((R / "t1" / ("g1_open_geometry_DRYRUN_dev.json" if draft else "g1_open_geometry.json")).read_text())
    g3 = json.loads((R / "t1" / ("g3_exact_recovery_PILOT_seed100.json" if draft else "g3_exact_recovery.json")).read_text())
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.8), gridspec_kw={"width_ratios": [1.2, 1, 1, 1.15], "wspace": 0.62})

    # (a) per model
    ax = axs[0]
    ms = sorted(g1["models"], key=lambda m: g1["models"][m]["rho_answers"])
    y = np.arange(len(ms))
    for i, m in enumerate(ms):
        r = g1["models"][m]
        ax.plot([r["rho_text"], r["rho_answers"]], [i, i], color=LIGHT, lw=1.2, zorder=1)
    ax.scatter([g1["models"][m]["rho_text"] for m in ms], y, s=12, color=MUTE, zorder=3, label="text")
    ax.scatter([g1["models"][m]["rho_answers"] for m in ms], y, s=12, color=BLUE, zorder=3, label="answers, exact")
    ax.scatter([g1["models"][m]["rho_sampled_n8"] for m in ms], y, s=14, facecolor="none", edgecolor=INK, lw=0.6, zorder=4,
               label="answers, 8 samples")
    ax.set_yticks(y, [NAMES[m] for m in ms])
    ax.set_xlim(0, 0.9)
    ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", ".25", ".50", ".75"])
    ax.set_xlabel("Spearman with own hidden geometry")
    ax.set_ylim(-0.6, len(ms) + 1.9)
    ax.legend(loc="upper left", frameon=False, fontsize=5.2, handletextpad=0.1, borderaxespad=0.1, ncol=1, labelspacing=0.25)
    style(ax)
    ax.set_title("(a) open models", loc="left", fontsize=7)

    # (b) G7: own-model excess per output channel, 95% intervals, against the last layer (and the chosen layer)
    ax = axs[1]
    g7f = R / "t1" / "g7_channels.json"
    g7 = json.loads((g7f if g7f.exists() else R / "t1" / "g7_channels_DRYRUN_dev.json").read_text())["channels"]
    order = ["answers", "top1", "top5", "top20", "full", "logprob"]
    labs = ["yes/no", "top-1", "top-5", "top-20", "full", "log-p"]
    xx = np.arange(len(order))
    for off, layer, face in ((-0.13, "chosen_layer", "white"), (0.13, "last_layer", BLUE)):
        e = np.array([g7[layer][c]["excess"] for c in order])
        ax.errorbar(xx + off, e[:, 0], yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], fmt="o", ms=2.8, mfc=face, mec=BLUE, ecolor=INK,
                    elinewidth=0.5, capsize=1.2, label="last layer" if face != "white" else "chosen layer", zorder=3)
    ax.axhline(0, color=MUTE, lw=0.6)
    ax.set_xticks(xx, labs, fontsize=5.0, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("own-model excess")
    ax.legend(loc="upper left", frameon=False, fontsize=5.0, handlelength=1.0)
    style(ax)
    ax.set_title("(b) API channels", loc="left", fontsize=7)

    # (c) exact recovery
    ax = axs[2]
    pts = sorted((float(k[:-1]), v) for k, v in g3["by_N"].items())
    x = [p[0] for p in pts]
    e = [1 - p[1]["pearson_cos"] if np.isfinite(p[1]["pearson_cos"]) else 1 for p in pts]
    floor = 1e-8                                    # Pearson is stored to 8 digits: 1 - Pearson = 0 means <= 1e-8
    bound = [v < floor for v in e]
    e = [max(v, floor) for v in e]
    ax.axvspan(0.2, 1.0, color=LIGHT, alpha=0.35, lw=0)
    ax.axvline(1.0, color=INK, lw=0.6, ls=":")
    ax.plot(x, e, color=BLUE, lw=1.1)
    ax.scatter([a for a, b in zip(x, bound) if not b], [v for v, b in zip(e, bound) if not b], color=BLUE, s=7, zorder=3)
    ax.scatter([a for a, b in zip(x, bound) if b], [v for v, b in zip(e, bound) if b], color=BLUE, marker="v", s=14, zorder=3)
    for b, lab in ((g3["baseline_raw_centered_logprob_pearson"], "raw log-probs"), (g3["baseline_whitened_no_constraint_pearson"], "whitened")):
        ax.axhline(1 - b, color=MUTE, lw=0.6, ls="--")

    ax.text(1.15, 1.6, "2 baselines,\nno sphere", color=MUTE, fontsize=4.8, va="bottom", linespacing=0.95)
    ax.set_yscale("log")
    ax.set_ylim(3e-9, 60)
    ax.set_xlim(0.2, 2.05)
    ax.set_xlabel(r"contexts $N\,/\,\frac{d(d+1)}{2}$")
    ax.set_ylabel("1 − Pearson (cosines)")
    ax.text(0.6, 1e-5, "under-\ndetermined", color=MUTE, fontsize=5.2, ha="center")
    style(ax)
    ax.set_title("(c) exact recovery", loc="left", fontsize=7)

    # (d) G6: answers of model A vs hidden geometry of model B (fresh contexts); the own-model diagonal outlined
    ax = axs[3]
    g6 = json.loads((R / "t1" / "g6_shared_geometry.json").read_text())
    Mo = list(g6["rho_answers_A_vs_hidden_B"])
    Mat = np.array([[g6["rho_answers_A_vs_hidden_B"][a][b] for b in Mo] for a in Mo])
    im = ax.imshow(Mat, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    for i in range(len(Mo)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.0))
    short = {"gemma3_4b": "Gemma", "gptoss20b": "gpt-oss", "llama31_8b": "Llama", "mistral7b": "Mistral", "olmo3_7b": "OLMo",
             "phi4mini": "Phi", "qwen25_7b": "Qwen2.5", "qwen3_8b": "Qwen3"}
    ax.set_xticks(range(len(Mo)), [short[m] for m in Mo], fontsize=4.8, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(Mo)), [short[m] for m in Mo], fontsize=4.8)
    ax.set_xlabel("hidden geometry of model B", fontsize=5.6); ax.set_ylabel("answers of model A", fontsize=5.6)
    ax.yaxis.set_label_coords(-0.36, 0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=4.8)
    ax.text(0.5, -0.44, "outlined: own model; excess %+.3f" % g6["S1_point"], transform=ax.transAxes, ha="center", fontsize=5.2, color=MUTE)
    ax.set_title("(d) shared, not the model's own", loc="left", fontsize=7)
    v2.save(fig, "fig_infogeom" + ("_draft" if draft else ""))
    print("saved", OUT / ("fig_infogeom" + ("_draft" if draft else "") + ".pdf"))


if __name__ == "__main__":
    main()
