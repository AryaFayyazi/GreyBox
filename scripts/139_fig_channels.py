#!/usr/bin/env python
"""Figure: what each API output channel reveals of a model's own state (G7). From results/t1/g7_channels.json (or the
dev dry run with draft=1). (a) own-model excess, 95% intervals, against the chosen and the last layer, for the channel
ladder; (b) how often the model's own hidden geometry is the best match among the 8 (last layer).
python scripts/139_fig_channels.py [draft=1]
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2", ROOT / "scripts/118_paper_figures_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
plt = v2.plt
ORDER = ["answers", "top1", "top5", "top20", "full", "logprob"]
LABEL = {"answers": "yes/no\nanswers", "top1": "top-1", "top5": "top-5", "top20": "top-20", "full": "full\n(Fisher–Rao)",
         "logprob": "log-prob\nvectors"}


def main():
    draft = "draft=1" in sys.argv
    r = json.loads((ROOT / "results/t1" / ("g7_channels_DRYRUN_dev.json" if draft else "g7_channels.json")).read_text())
    L = r["channels"]
    fig, axs = plt.subplots(1, 2, figsize=(v2.FULL, 1.95), gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.32})
    ax = axs[0]
    v2.grid(ax)
    x = np.arange(len(ORDER))
    for off, layer, col, lab in ((-0.12, "chosen_layer", v2.LIGHT, "vs the layer chosen on dev"), (0.12, "last_layer", v2.BLUE, "vs the last layer")):
        e = np.array([L[layer][c]["excess"] for c in ORDER])
        ax.errorbar(x + off, e[:, 0], yerr=[e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]], fmt="o", ms=3.6, color=col if col != v2.LIGHT else v2.MUTE,
                    mfc=col if col != v2.LIGHT else "white", ecolor=v2.INK, elinewidth=0.6, capsize=1.6, label=lab, zorder=3)
    ax.axhline(0, color=v2.MUTE, lw=0.7)
    ax.axvspan(-0.5, 0.5, color=v2.GRID, alpha=0.45, lw=0, zorder=0)
    ax.text(0, 0.19, "what models\nshare", ha="center", va="top", fontsize=5.4, color=v2.MUTE)
    ax.text(4.75, 0.207, "the model's own state", ha="right", va="center", fontsize=5.4, color=v2.MUTE)
    ax.set_xticks(x, [LABEL[c] for c in ORDER], fontsize=5.6)
    ax.set_ylabel("own-model excess (Spearman)")
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0), fontsize=5.4)
    ax.set_title("(a) What crosses the API boundary")
    ax = axs[1]
    v2.grid(ax)
    best = [L["last_layer"][c]["own_best"] for c in ORDER]
    ax.bar(x, best, 0.7, color=[v2.BLUE if c == "logprob" else v2.LIGHT for c in ORDER], edgecolor=v2.MUTE, lw=0.4)
    for xi, b in zip(x, best):
        ax.text(xi, b + 0.2, f"{b}/8", ha="center", fontsize=5.4, color=v2.INK)
    ax.axhline(1, color=v2.MUTE, ls=":", lw=0.7)
    ax.text(len(ORDER) - 0.6, 1.2, "chance", fontsize=5.2, color=v2.MUTE, ha="right")
    ax.set_xticks(x, [LABEL[c].split("\n")[0] for c in ORDER], fontsize=5.4, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 9.2)
    ax.set_ylabel("models whose own internals\nare the best match")
    ax.set_title("(b) Identifying the model")
    name = "fig_channels" + ("_draft" if draft else "")
    v2.save(fig, name)
    print("saved", name)


if __name__ == "__main__":
    main()
