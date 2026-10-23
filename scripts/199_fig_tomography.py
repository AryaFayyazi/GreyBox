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
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
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
        for score, ls in (("rk", "-"),):
            xs, ms, lo, hi = [], [], [], []
            for k in range(-5, 1):
                v = S[(S.episode_violates == viol) & (S.k == k)][score].values
                if len(v) < 10:
                    continue
                b = [rng.choice(v, len(v)).mean() for _ in range(400)]
                xs.append(k); ms.append(v.mean()); lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
            ax.plot(xs, ms, ls, color=col, lw=1.4 if score == "rk" else 1.1, label=lab)
            if score == "rk":
                ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
    style(ax)
    ax.set_xlabel("steps before the violating (or last) call", fontsize=6.2, color=INK)
    ax.set_ylabel("estimated risk", fontsize=6.4, color=INK)
    ax.legend(fontsize=5.4, frameon=False, loc="upper left")


def _agent_metrics():
    """per agent with a step battery: within template x tool x step AUROC (raw, filtered, forecast) and same-step
    agreement with the true decision state (raw, filtered, forecast). G27b's cross-fit estimates (scripts/197)."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    import pandas as pd
    s = importlib.util.spec_from_file_location("g27b", ROOT / "scripts/197_g27b_replication.py")
    b = importlib.util.module_from_spec(s); s.loader.exec_module(b)
    s2 = importlib.util.spec_from_file_location("m171", ROOT / "scripts/171_decision_greybox_LOCKED.py")
    M = importlib.util.module_from_spec(s2); s2.loader.exec_module(M)
    from graybox.stats import within_auroc
    sg = np.array([q[1] for q in b.G27.QUESTIONS])
    out = []
    for a in os.environ.get("FIG_AGENTS", "gemma4,gptoss20b,qwen38,museglimmer").split(","):
        f = b.G27.OUT / f"{a}_steps.parquet"
        if not f.exists():
            continue
        S = pd.read_parquet(f)
        Y, fc, fl, pr, nis, _ = b.estimates(S)
        y = S.y.values.astype(int)
        st = np.array([f"{x}|{z}|{t}" for x, z, t in zip(S.template, S.tool, S.t)])
        auc = [within_auroc(y, E @ sg, st) for E in (Y, fl, fc)]
        U, H = M.states(a)
        key = {(e, int(v)): i for i, (e, v) in enumerate(zip(U.episode_id, U.event_id))}
        idx = np.array([key.get((e, int(v)), -1) for e, v in zip(S.episode_id, S.event_id)])
        ok = (idx >= 0) & np.array([bool(U.ok.values[k]) if k >= 0 else False for k in idx])
        same = pdist(S.t.values[ok].astype(float)[:, None]) == 0
        dt = pdist(H[idx[ok], 3].astype(np.float32))[same]
        rsa = [spearmanr(pdist(E[ok])[same], dt).correlation for E in (Y, fl, fc)]
        out.append((a, auc, rsa))
    return out


METRICS = None


def panel_auc(ax):
    rows = METRICS
    x = np.arange(len(rows))
    for j, (lab, col) in enumerate((("answers before the call", MUTE), ("estimated state", BLUE), ("forecast from earlier steps", ORANGE))):
        ax.bar(x + (j - 1) * 0.24, [r[1][j] for r in rows], 0.22, color=col, edgecolor="white", lw=0.4, label=lab)
    ax.axhline(0.5, color=INK2, lw=0.6, ls="--")
    style(ax); ax.grid(axis="x", visible=False)
    ax.set_xticks(x); ax.set_xticklabels([NAMES[r[0]] for r in rows], fontsize=6)
    ax.set_ylim(0.35, 0.9); ax.set_ylabel("AUROC, same step and task", fontsize=6.2, color=INK)
    ax.legend(fontsize=5.0, frameon=False, loc="upper right")


def panel_truth(ax):
    rows = METRICS
    x = np.arange(len(rows))
    for j, (lab, col) in enumerate((("answers before the call", MUTE), ("estimated state", BLUE), ("forecast", ORANGE))):
        ax.bar(x + (j - 1) * 0.24, [r[2][j] for r in rows], 0.22, color=col, edgecolor="white", lw=0.4, label=lab)
    ax.axhline(0, color=INK2, lw=0.6)
    style(ax); ax.grid(axis="x", visible=False)
    ax.set_xticks(x); ax.set_xticklabels([NAMES[r[0]] for r in rows], fontsize=6)
    ax.set_ylabel("Spearman with true geometry", fontsize=6.2, color=INK)


def main():
    global METRICS
    METRICS = _agent_metrics()
    fig, ax = plt.subplots(1, 3, figsize=(6.75, 2.15), gridspec_kw={"wspace": 0.5})
    panel_traj(ax[0]); panel_auc(ax[1]); panel_truth(ax[2])
    for a_, t in zip(ax, ("(a) Risk before a violation", "(b) Ranking the coming call", "(c) Against the true state")):
        a_.set_title(t, fontsize=7.0, loc="left", fontweight="bold", color=INK, pad=6)
    for out in (ROOT / "paper/figures/tomography.pdf", ROOT / "results/figures/tomography.png"):
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("saved")


if __name__ == "__main__":
    main()
