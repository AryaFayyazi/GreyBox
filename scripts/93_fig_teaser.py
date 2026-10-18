#!/usr/bin/env python
"""First-page figure: seeing inside closed agents.
All panels share the axes of one reference open agent's hidden state (Gemma-4, decision point, development PCs 1-2;
T15's standardization and PCA). (a) the agent's TRUE state on fresh steps; (b) its grey-box replica from the
transcript (T15's ridge map from the reader's layer 32, alpha 1000); (c, d) the replicas of Claude Haiku 5.5's and
GPT-6-luna's actions (T8b, T11 units), through the same map: shown in the reference agent's coordinates, since a closed
model's own coordinates are not identifiable (Proposition 1). (e) live prevention (T16), if its result exists.
Numbers on the panels come from results/t1/t15_latent_replication.json, t8b, t11 and t16 result files.
python scripts/93_fig_teaser.py
"""
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

F = importlib.import_module("90_figures")
I, T4 = Path("/data/models/graybox_ws/outputs/interrogate"), Path("/data/models/graybox_ws/outputs/t4")
RED, INK, MUTE = "#d1462f", F.INK, "#6b7682"


def main():
    Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
    Dd, Df = np.load(I / "gemma4_clean_decision_hidden.npz"), np.load(I / "gemma4_fresh_decision_hidden.npz")
    okd, okf = Dd["ok"].astype(bool), Df["ok"].astype(bool)
    Yd_full = Dd["h"][okd].reshape(okd.sum(), -1).astype(np.float32)
    Yf_full = Df["h"][okf].reshape(okf.sum(), -1).astype(np.float32)
    sc = StandardScaler().fit(Yd_full)
    pca = PCA(128, random_state=0).fit(sc.transform(Yd_full))
    Yd, Yf = pca.transform(sc.transform(Yd_full)), pca.transform(sc.transform(Yf_full))
    Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz")
    j = [int(x) for x in Hd["layers"]].index(32)
    xs = StandardScaler().fit(Hd["h"][okd, j].astype(np.float32))
    ridge = Ridge(alpha=1000).fit(xs.transform(Hd["h"][okd, j].astype(np.float32)), Yd)
    rep = ridge.predict(xs.transform(Hf["h"][okf, j].astype(np.float32)))
    t15 = json.loads((ROOT / "results/t1/t15_latent_replication.json").read_text())
    assert abs(r2_score(Yf, rep, multioutput="variance_weighted") - t15["r2_fresh"]["READER"]) < 1e-3
    lf = json.loads((ROOT / "results/t1/label_free_cuts.json").read_text())   # one operating point: the label-free cut
    r17 = [a["r2_fresh"]["READER"] for a in json.loads((ROOT / "results/t1/t17_replication_agents.json").read_text())["agents"].values()]
    r17lo, r17hi = min(r17), max(r17)
    yg = Uf.y.values[okf].astype(bool)
    t8, t11 = importlib.import_module("76_t8b_claude_all_actions"), importlib.import_module("81_t11_gpt_monitor")
    closed = {}
    for key, mod, feats in (("claude", t8, t8.OUT / "features.npz"), ("gpt", t11, t11.OUT / "reader.npz")):
        U = mod.units()
        closed[key] = (ridge.predict(xs.transform(np.load(feats)["h"].astype(np.float32))), U.y.values.astype(bool))
    r8 = json.loads((ROOT / "results/t1/t8b_claude_all_actions.json").read_text())
    r11 = json.loads((ROOT / "results/t1/t11_gpt_agent.json").read_text())
    t16p = ROOT / "results/t1/t16_live_gate.json"
    t16 = json.loads(t16p.read_text()) if t16p.exists() else None
    t16bp = ROOT / "results/t1/t16b_claude_live_gate.json"
    t16b = json.loads(t16bp.read_text()) if t16bp.exists() else None

    # shared axes: robust range over all four panels (no panel's points are cut off)
    allz = np.vstack([Yf[:, :2], rep[:, :2], closed["claude"][0][:, :2], closed["gpt"][0][:, :2]])
    lo, hi = np.percentile(allz, [0.2, 99.8], axis=0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    gx, gy = np.mgrid[lo[0]:hi[0]:160j, lo[1]:hi[1]:160j]

    def panel(ax, Z, y, title, sub, n_safe_pts=2500, seed=0):
        safe = Z[~y, :2]
        kde = gaussian_kde(safe[np.random.default_rng(seed).choice(len(safe), min(4000, len(safe)), replace=False)].T, bw_method=0.18)
        dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.contourf(gx, gy, dens, levels=np.linspace(dens.max() * 0.04, dens.max(), 9), cmap="Blues", alpha=0.85)
        sel = np.random.default_rng(seed).choice(len(safe), min(n_safe_pts, len(safe)), replace=False)
        ax.scatter(safe[sel, 0], safe[sel, 1], s=0.4, c="#3d5a80", alpha=0.25, lw=0)
        bad = Z[y, :2]
        ax.scatter(bad[:, 0], bad[:, 1], s=9, c=RED, alpha=0.18, lw=0)          # glow
        ax.scatter(bad[:, 0], bad[:, 1], s=2.2, c=RED, alpha=0.95, lw=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c9d0d6")
        ax.text(0.0, 1.2, title, transform=ax.transAxes, fontsize=6.9, color=INK, fontweight="bold", va="bottom")
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=5.6, color=MUTE, va="bottom", linespacing=1.1)

    ncol = 5 if t16 else 4
    fig = plt.figure(figsize=(6.9, 2.2))
    gs = fig.add_gridspec(1, ncol, width_ratios=[1, 1, 1, 1] + ([0.78] if t16 else []), wspace=0.08 if not t16 else 0.1)
    axs = [fig.add_subplot(gs[0, k]) for k in range(ncol)]
    panel(axs[0], Yf, yg, "(a) Open agent", "true hidden state, weights open\n12,128 fresh actions")
    panel(axs[1], rep, yg, "(b) Its replica",
          f"transcript only; {t15['r2_fresh']['READER']:.0%} of variance\n3 more agents: {r17lo:.0%}–{r17hi:.0%}", seed=1)
    Zc, yc = closed["claude"]
    panel(axs[2], Zc, yc, "(c) Claude (closed)",
          f"Haiku 5.5; weights never seen\nprobe: {lf['claude_T8b']['READER']['label_free_cut']:.1%} caught at 5% review", seed=2)
    Zg, yg2 = closed["gpt"]
    panel(axs[3], Zg, yg2, "(d) GPT (closed)",
          f"6-luna; weights never seen\nprobe: {r11['label_free_cut']['READER']['all']:.1%} caught at 5% review", seed=3)
    axs[0].set_xlabel("latent axis 1 (reference agent)", fontsize=5.6, color=MUTE, labelpad=1)
    axs[0].set_ylabel("latent axis 2", fontsize=5.6, color=MUTE, labelpad=1)
    if t16:
        ax = axs[4]
        groups = [("GPT", t16)] + ([("Claude", t16b)] if t16b else [])
        for g, (name, d) in enumerate(groups):
            r = d["unsafe_episode_rate"]
            for k, (arm, col) in enumerate((("ungated", "#9aa5b1"), ("gated", F.C[0]))):
                x = g * 2.6 + k
                ax.bar(x, r[arm], color=col, width=0.9)
                ax.text(x, r[arm] + 0.002, f"{r[arm]:.1%}", ha="center", va="bottom", fontsize=5.6, color=INK, fontweight="bold")
        ax.set_xticks([g * 2.6 + 0.5 for g in range(len(groups))]); ax.set_xticklabels([n for n, _ in groups], fontsize=6)
        vmax = max(d["unsafe_episode_rate"]["ungated"] for _, d in groups)
        ax.set_ylim(0, vmax * 1.35 + 0.01)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", labelsize=5.4)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.text(0.0, 1.2, "(e) Live prevention", transform=ax.transAxes, fontsize=6.9, color=INK, fontweight="bold", va="bottom")
        ax.text(0.0, 1.02, "grey: no gate; blue: gate\nin the agent's loop", transform=ax.transAxes, fontsize=5.6, color=MUTE, va="bottom", linespacing=1.1)
        ax.set_ylabel("episodes with a violation", fontsize=5.4, color=MUTE, labelpad=1)
    h = [plt.Line2D([], [], marker="o", ls="", color="#3d5a80", alpha=0.6, ms=4, label="safe action (density)"),
         plt.Line2D([], [], marker="o", ls="", color=RED, ms=4, label="violation")]
    fig.subplots_adjust(left=0.025, right=0.995, top=0.74, bottom=0.1)
    F.save(fig, "teaser")
    print("teaser saved; panels", ncol)


if __name__ == "__main__":
    main()
