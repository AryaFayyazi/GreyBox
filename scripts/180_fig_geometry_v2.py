#!/usr/bin/env python
"""Geometry figures, split for the training-free story (same data and computations as scripts/130):
  main (one column): (a) per open model, Spearman of the answer geometry (exact; 8 samples) and of the input text with
       the model's own hidden geometry (G1); (b) answers of model A vs hidden geometry of model B on fresh contexts (G6).
  appendix: (a) own-model excess per API channel (G7); (b) exact recovery of the final geometry from log-probabilities (G3).
  python scripts/180_fig_geometry_v2.py -> paper/figures/geometry_main.pdf, paper/figures/geometry_channels.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
BLUE, ORANGE, INK, INK2, MUTE, GRID, LIGHT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#d8d7d2"
NAMES = {"gemma3_4b": "Gemma-3 4B", "gptoss20b": "gpt-oss 20B", "llama31_8b": "Llama-3.1 8B", "mistral7b": "Mistral 7B", "olmo3_7b": "OLMo-3 7B",
         "phi4mini": "Phi-4 mini", "qwen25_7b": "Qwen2.5 7B", "qwen3_8b": "Qwen3 8B"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2})


def style(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(MUTE); ax.spines[s_].set_linewidth(0.6)
    ax.tick_params(labelsize=5.6, colors=INK2, length=2.2, width=0.6)


def save(fig, name):
    for p in (ROOT / "paper/figures" / f"{name}.pdf", R / "figures" / f"{name}.png"):
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main_figure():
    g1 = json.loads((R / "t1/g1_open_geometry.json").read_text())
    g6 = json.loads((R / "t1/g6_shared_geometry.json").read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.35, 1.95), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.75})
    ms = sorted(g1["models"], key=lambda m: g1["models"][m]["rho_answers"])
    y = np.arange(len(ms))
    style(ax); ax.grid(axis="y", visible=False)
    for i, m in enumerate(ms):
        r = g1["models"][m]
        ax.plot([r["rho_text"], r["rho_answers"]], [i, i], color=LIGHT, lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter([g1["models"][m]["rho_text"] for m in ms], y, s=16, color=MUTE, zorder=3, label="input text")
    ax.scatter([g1["models"][m]["rho_answers"] for m in ms], y, s=20, color=BLUE, edgecolor="white", lw=0.5, zorder=4, label="answers, exact")
    ax.scatter([g1["models"][m]["rho_sampled_n8"] for m in ms], y, s=18, facecolor="none", edgecolor=INK, lw=0.6, zorder=5, label="answers, 8 samples")
    ax.set_yticks(y, [NAMES[m] for m in ms], fontsize=5.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.1, 0.82); ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Spearman with its own\nhidden geometry", fontsize=5.8)
    ax.set_ylim(-0.6, len(ms) - 0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.55, 1.0), frameon=False, fontsize=5.2, handletextpad=0.1, ncol=1, labelspacing=0.2)
    Mo = list(g6["rho_answers_A_vs_hidden_B"])
    Mat = np.array([[g6["rho_answers_A_vs_hidden_B"][a][b] for b in Mo] for a in Mo])
    im = bx.imshow(Mat, cmap="Blues", vmin=0, vmax=0.8, aspect="equal")
    for i in range(len(Mo)):
        bx.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.0))
    short = {"gemma3_4b": "Gemma", "gptoss20b": "gpt-oss", "llama31_8b": "Llama", "mistral7b": "Mistral", "olmo3_7b": "OLMo",
             "phi4mini": "Phi", "qwen25_7b": "Qwen2.5", "qwen3_8b": "Qwen3"}
    bx.set_xticks(range(len(Mo)), [short[m] for m in Mo], fontsize=4.9, rotation=60, ha="right", rotation_mode="anchor")
    bx.set_yticks(range(len(Mo)), [short[m] for m in Mo], fontsize=4.9)
    bx.tick_params(length=0)
    for s_ in bx.spines.values():
        s_.set_visible(False)
    bx.set_xlabel("hidden geometry of model B", fontsize=5.6, labelpad=1); bx.set_ylabel("answers of model A", fontsize=5.6, labelpad=1)
    cb = fig.colorbar(im, ax=bx, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=4.8, length=1.5); cb.outline.set_visible(False)
    cb.set_label("Spearman", fontsize=5.0)
    fig.canvas.draw()
    top = max(ax.get_position().y1, bx.get_position().y1) + 0.26
    fig.text(0.0, top, "(a) Answers track it", fontsize=6.6, fontweight="bold", color=INK, va="bottom")
    fig.text(bx.get_position().x0 - 0.10, top, "(b) Shared across models", fontsize=6.6, fontweight="bold", color=INK, va="bottom")
    fig.text(bx.get_position().x0 - 0.10, top - 0.07, f"own model (outlined): excess {g6['S1_point']:+.3f}", fontsize=5.2, color=INK2, va="bottom")
    save(fig, "geometry_main")


def channels_figure():
    g3 = json.loads((R / "t1/g3_exact_recovery.json").read_text())
    g7 = json.loads((R / "t1/g7_channels.json").read_text())["channels"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.2, 1.8), gridspec_kw={"wspace": 0.45})
    order = ["answers", "top1", "top5", "top20", "full", "logprob"]
    labs = ["yes/no", "top-1", "top-5", "top-20", "full", "log-p"]
    xx = np.arange(len(order))
    style(ax)
    for off, layer, face in ((-0.13, "chosen_layer", "white"), (0.13, "last_layer", BLUE)):
        e = np.array([g7[layer][c]["excess"] for c in order])
        ax.errorbar(xx + off, e[:, 0], yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], fmt="o", ms=3, mfc=face, mec=BLUE, ecolor=INK,
                    elinewidth=0.5, capsize=1.2, label="last layer" if face != "white" else "chosen layer", zorder=3)
    ax.axhline(0, color=MUTE, lw=0.6)
    ax.set_xticks(xx, labs, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("own-model excess"); ax.legend(loc="upper left", frameon=False, fontsize=5.2)
    ax.set_title("(a) what each API channel reveals of the model's own state (G7)", loc="left", fontsize=6.4, fontweight="bold")
    pts = sorted((float(k[:-1]), v) for k, v in g3["by_N"].items())
    x = [p[0] for p in pts]
    e = [max(1 - p[1]["pearson_cos"] if np.isfinite(p[1]["pearson_cos"]) else 1, 1e-8) for p in pts]
    style(bx)
    bx.axvspan(0.2, 1.0, color=LIGHT, alpha=0.35, lw=0); bx.axvline(1.0, color=INK, lw=0.6, ls=":")
    bx.plot(x, e, color=BLUE, lw=1.1, marker="o", ms=2.5)
    bx.set_yscale("log"); bx.set_ylim(3e-9, 60); bx.set_xlim(0.2, 2.05)
    bx.set_xlabel(r"contexts $N\,/\,\frac{d(d+1)}{2}$"); bx.set_ylabel("1 − Pearson (cosines)")
    bx.text(0.6, 1e-5, "under-\ndetermined", color=MUTE, fontsize=5.2, ha="center")
    bx.set_title("(b) exact recovery from log-probabilities (G3)", loc="left", fontsize=6.4, fontweight="bold")
    save(fig, "geometry_channels")


if __name__ == "__main__":
    main_figure(); channels_figure(); print("saved")
