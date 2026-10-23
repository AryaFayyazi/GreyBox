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


def save(fig, name, tight=True):
    kw = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    for p in (ROOT / "paper/figures" / f"{name}.pdf", R / "figures" / f"{name}.png"):
        fig.savefig(p, dpi=300, **kw)
    plt.close(fig)


def main_figure():
    """One column, drawn at its final size (3.25 in wide, no tight crop), so the type sizes are true sizes."""
    g1 = json.loads((R / "t1/g1_open_geometry.json").read_text())
    g6 = json.loads((R / "t1/g6_shared_geometry.json").read_text())
    W, H = 3.25, 2.36
    fig = plt.figure(figsize=(W, H))

    def axes_in(x0, y0, w, h):  # place axes in inches
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H])

    TT, TK, LB = 7.4, 6.0, 6.3  # title, tick, label sizes (pt)
    # (a) per model: Spearman with its own hidden geometry
    ax = axes_in(0.70, 0.43, 0.74, 1.40)
    ms = sorted(g1["models"], key=lambda m: g1["models"][m]["rho_answers"])
    y = np.arange(len(ms))
    style(ax); ax.grid(axis="y", visible=False); ax.tick_params(labelsize=TK)
    for i, m in enumerate(ms):
        r = g1["models"][m]
        ax.plot([r["rho_text"], r["rho_answers"]], [i, i], color=LIGHT, lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter([g1["models"][m]["rho_text"] for m in ms], y, s=16, color=MUTE, zorder=3, label="input text")
    ax.scatter([g1["models"][m]["rho_answers"] for m in ms], y, s=20, color=BLUE, edgecolor="white", lw=0.5, zorder=4, label="answers, exact")
    ax.scatter([g1["models"][m]["rho_sampled_n8"] for m in ms], y, s=18, facecolor="none", edgecolor=INK, lw=0.6, zorder=5, label="answers, 8 samples")
    ax.set_yticks(y, [NAMES[m] for m in ms], fontsize=TK, color=INK)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.1, 0.82); ax.set_xticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"])
    ax.set_xlabel("Spearman with its own\nhidden geometry", fontsize=LB, color=INK, labelpad=2)
    ax.set_ylim(-0.6, len(ms) - 0.4)
    h, l = ax.get_legend_handles_labels()
    order = [1, 2, 0]  # answers exact, answers 8 samples, input text
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(0.0, (H - 0.19) / H), frameon=False,
               fontsize=LB, handletextpad=0.2, labelspacing=0.25, borderaxespad=0, borderpad=0, handlelength=1.2)
    fig.text(0.0, (H - 0.02) / H, "(a) Answers track it", fontsize=TT, fontweight="bold", color=INK, va="top")
    # (b) answers of model A vs hidden geometry of model B
    S, RM = 1.18, 0.09  # heat-map side, right margin (in)
    bx = axes_in(W - S - RM, 0.50, S, S)
    Mo = list(g6["rho_answers_A_vs_hidden_B"])
    Mat = np.array([[g6["rho_answers_A_vs_hidden_B"][a][b] for b in Mo] for a in Mo])
    im = bx.imshow(Mat, cmap="Blues", vmin=0, vmax=0.8, aspect="equal")
    for i in range(len(Mo)):
        bx.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.1))
    short = {"gemma3_4b": "Gemma", "gptoss20b": "gpt-oss", "llama31_8b": "Llama", "mistral7b": "Mistral", "olmo3_7b": "OLMo",
             "phi4mini": "Phi", "qwen25_7b": "Qwen2.5", "qwen3_8b": "Qwen3"}
    bx.set_xticks(range(len(Mo)), [short[m] for m in Mo], fontsize=TK, rotation=60, ha="right", rotation_mode="anchor", color=INK2)
    bx.set_yticks(range(len(Mo)), [short[m] for m in Mo], fontsize=TK, color=INK2)
    bx.tick_params(length=0, pad=1.5)
    for s_ in bx.spines.values():
        s_.set_visible(False)
    bx.set_xlabel("hidden geometry of model B", fontsize=LB, color=INK, labelpad=1)
    bx.set_ylabel("answers of model A", fontsize=LB, color=INK, labelpad=1)
    # compact horizontal colour bar above the heat map, right-aligned
    cw = 0.62
    cax = axes_in(W - RM - cw, 0.50 + S + 0.07, cw, 0.055)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.4, 0.8])
    cb.ax.xaxis.set_ticks_position("top"); cb.ax.tick_params(labelsize=TK, length=1.5, width=0.5, pad=1, colors=INK2)
    cb.ax.set_xticklabels(["0", "0.4", "0.8"]); cb.outline.set_visible(False)
    fig.text((W - RM - cw - 0.05) / W, (0.50 + S + 0.0775) / H, "Spearman", fontsize=TK, color=INK2, ha="right", va="center")
    bx0 = 1.58
    fig.text(bx0 / W, (H - 0.02) / H, "(b) Shared across models", fontsize=TT, fontweight="bold", color=INK, va="top")
    fig.text(bx0 / W, (H - 0.19) / H, f"own model (outlined): excess {g6['S1_point']:+.3f}", fontsize=TK, color=INK2, va="top")
    save(fig, "geometry_main", tight=False)


def channels_figure():
    """Appendix, drawn at its final size (0.8 x 6.75 in = 5.4 in wide, no tight crop)."""
    g3 = json.loads((R / "t1/g3_exact_recovery.json").read_text())
    g7 = json.loads((R / "t1/g7_channels.json").read_text())["channels"]
    W, H = 5.4, 2.05
    TT, TK, LB = 7.6, 6.4, 6.8
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.62 / W, right=1 - 0.13 / W, bottom=0.46 / H, top=1 - 0.40 / H, wspace=0.42)
    order = ["answers", "top1", "top5", "top20", "full", "logprob"]
    labs = ["yes/no", "top-1", "top-5", "top-20", "full", "log-p"]
    xx = np.arange(len(order))
    style(ax); ax.tick_params(labelsize=TK)
    for off, layer, face in ((-0.13, "chosen_layer", "white"), (0.13, "last_layer", BLUE)):
        e = np.array([g7[layer][c]["excess"] for c in order])
        ax.errorbar(xx + off, e[:, 0], yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], fmt="o", ms=3.2, mfc=face, mec=BLUE, ecolor=INK,
                    elinewidth=0.5, capsize=1.2, label="last layer" if face != "white" else "chosen layer", zorder=3)
    ax.axhline(0, color=MUTE, lw=0.6)
    ax.set_xticks(xx, labs, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("own-model excess", fontsize=LB, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=TK, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    ax.set_title("(a) What each API channel reveals\nof the model's own state", loc="left", fontsize=TT, fontweight="bold",
                 color=INK, pad=4)
    pts = sorted((float(k[:-1]), v) for k, v in g3["by_N"].items())
    x = [p[0] for p in pts]
    e = [max(1 - p[1]["pearson_cos"] if np.isfinite(p[1]["pearson_cos"]) else 1, 1e-8) for p in pts]
    style(bx); bx.tick_params(labelsize=TK)
    bx.axvspan(0.2, 1.0, color=LIGHT, alpha=0.35, lw=0); bx.axvline(1.0, color=INK, lw=0.6, ls=":")
    bx.plot(x, e, color=BLUE, lw=1.1, marker="o", ms=2.8)
    bx.set_yscale("log"); bx.set_ylim(3e-9, 60); bx.set_xlim(0.2, 2.05)
    bx.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0]); bx.minorticks_off()
    bx.set_xlabel(r"contexts $N\,/\,\frac{d(d+1)}{2}$", fontsize=LB, color=INK, labelpad=1)
    bx.set_ylabel("1 \u2212 Pearson (cosines)", fontsize=LB, color=INK)
    bx.text(0.6, 1e-5, "under-\ndetermined", color=MUTE, fontsize=TK, ha="center")
    bx.set_title("(b) Exact recovery from\nlog-probabilities", loc="left", fontsize=TT, fontweight="bold", color=INK, pad=4)
    save(fig, "geometry_channels", tight=False)


if __name__ == "__main__":
    main_figure(); channels_figure(); print("saved")
