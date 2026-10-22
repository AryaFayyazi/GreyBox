#!/usr/bin/env python
"""Figure: state tomography of an agent through its own answers (G27, G28, G27b). Formats saved results; refits nothing
beyond G27's own cross-fit filter (scripts/195 estimates()).
(a) Gemma-4: the read-out of the forecast and of the filtered state, by steps before the first violating call, against
    episodes with no violation (mean, 95% interval);
(b) within-stratum AUROC per agent: raw answers, filtered state, forecast, prior (step index only);
(c) agreement (Spearman over pairs) of each estimate's geometry with the true hidden-state geometry.
python scripts/199_fig_tomography.py  -> paper/figures/tomography.pdf, results/figures/tomography.png"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/t1"
BLUE, ORANGE, INK, INK2, MUTE, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514d", "#8a8984", "#e6e5e1"
NAMES = {"gemma4": "Gemma-4", "gptoss20b": "gpt-oss", "qwen38": "Qwen3.8", "museglimmer": "Muse-Glimmer"}


def style(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color(MUTE); ax.spines["bottom"].set_color(MUTE)
    ax.tick_params(labelsize=6, colors=INK2, length=2, width=0.6)


def panel_traj(ax):
    s = importlib.util.spec_from_file_location("g28", ROOT / "scripts/195_g28_tomography_vs_truth.py")
    g28 = importlib.util.module_from_spec(s); s.loader.exec_module(g28)
    g27 = g28._mod("g27", "scripts/194_g27_state_tomography.py")
    S, Y, fc, fl = g28.estimates()
    sg = np.array([q[1] for q in g27.QUESTIONS])
    rf, rk = fc @ sg, fl @ sg
    S = S.assign(rf=rf, rk=rk)
    last = S.groupby("episode_id").t.transform("max")
    S = S.assign(k=S.t - last)                                   # 0 = the violating call (or the last step)
    rng = np.random.default_rng(0)
    for viol, col, lab in ((1, ORANGE, "episodes that violate"), (0, BLUE, "episodes that do not")):
        for score, ls in (("rk", "-"), ("rf", ":")):
            xs, ms, lo, hi = [], [], [], []
            for k in range(-5, 1):
                v = S[(S.episode_violates == viol) & (S.k == k)][score].values
                if len(v) < 10:
                    continue
                b = [rng.choice(v, len(v)).mean() for _ in range(400)]
                xs.append(k); ms.append(v.mean()); lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
            ax.plot(xs, ms, ls, color=col, lw=1.4 if score == "rk" else 1.1, label=f"{lab}, {'estimated' if score == 'rk' else 'forecast'}")
            if score == "rk":
                ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
    style(ax)
    ax.set_xlabel("steps before the violating (or last) call", fontsize=6.4, color=INK)
    ax.set_ylabel("risk read-out of the state", fontsize=6.4, color=INK)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left")


def panel_auc(ax):
    rows = []
    g = json.loads((R / "g27_state_tomography.json").read_text())
    w = g["within_stratum"]
    rows.append(("gemma4", w["raw"], w["filtered"], w["forecast_steps_after_first"], g["prior_only_after_first"]))
    f = R / "g27b_replication.json"
    if f.exists():
        b = json.loads(f.read_text())
        for a in ("gptoss20b", "qwen38", "museglimmer"):
            if isinstance(b.get(a), dict):
                ws = b[a]["within_stratum"]
                rows.append((a, ws["raw"], ws["filtered"], ws["forecast_after_first"], ws["prior_after_first"]))
    x = np.arange(len(rows))
    for j, (lab, col, hatch) in enumerate((("raw answers", MUTE, None), ("estimated state", BLUE, None), ("forecast", ORANGE, None), ("prior (step only)", GRID, "///"))):
        ax.bar(x + (j - 1.5) * 0.19, [r[j + 1] for r in rows], 0.17, color=col, edgecolor=INK2 if hatch else "white", hatch=hatch, lw=0.4, label=lab)
    ax.axhline(0.5, color=INK2, lw=0.6, ls="--")
    style(ax); ax.grid(axis="x", visible=False)
    ax.set_xticks(x); ax.set_xticklabels([NAMES[r[0]] for r in rows], fontsize=6)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUROC (within template and tool)", fontsize=6.4, color=INK)
    ax.legend(fontsize=5.2, frameon=False, ncol=2, loc="upper right")


def panel_truth(ax):
    rows = []
    f = R / "g28_tomography_vs_truth.json"
    if f.exists():
        t = json.loads(f.read_text())["rsa_with_true_state"]
        rows.append(("gemma4", t["raw"], t["filtered"], t["forecast"]))
    f = R / "g27b_replication.json"
    if f.exists():
        b = json.loads(f.read_text())
        for a in ("gptoss20b", "qwen38", "museglimmer"):
            tt = b.get(a, {}).get("G28_rsa_with_true_state") if isinstance(b.get(a), dict) else None
            if tt:
                rows.append((a, tt["raw"], tt["filtered"], tt["forecast"]))
    x = np.arange(len(rows))
    for j, (lab, col) in enumerate((("raw answers", MUTE), ("estimated state", BLUE), ("forecast", ORANGE))):
        ax.bar(x + (j - 1) * 0.24, [r[j + 1] for r in rows], 0.22, color=col, edgecolor="white", lw=0.4, label=lab)
    style(ax); ax.grid(axis="x", visible=False)
    ax.set_xticks(x); ax.set_xticklabels([NAMES[r[0]] for r in rows], fontsize=6)
    ax.set_ylabel("agreement with the true hidden geometry", fontsize=6.4, color=INK)
    ax.legend(fontsize=5.2, frameon=False, loc="upper right")


def main():
    fig, ax = plt.subplots(1, 3, figsize=(6.75, 1.95), gridspec_kw={"wspace": 0.42})
    panel_traj(ax[0]); panel_auc(ax[1]); panel_truth(ax[2])
    for a_, t in zip(ax, ("(a) The state before a violation", "(b) Ranking the next call", "(c) Against the true state")):
        a_.set_title(t, fontsize=7.2, loc="left", fontweight="bold", color=INK)
    for out in (ROOT / "paper/figures/tomography.pdf", ROOT / "results/figures/tomography.png"):
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("saved")


if __name__ == "__main__":
    main()
