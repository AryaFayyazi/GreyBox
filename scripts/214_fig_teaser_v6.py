#!/usr/bin/env python
"""First-page figure (v6: the chain; panel c adds the second closed agent at its call and before it). Based on v5.
(a) An open model's hidden geometry and (b) the same contexts placed only by its yes/no answers (scripts/134's
computation, unchanged: classical MDS (graybox.geometry.mds) of chord distances between hidden states, and of exact
Fisher-Rao distances between answer distributions, then Procrustes-rotated onto (a) for display).
(c) Monitoring a decision (held-out): per agent, the fixed read-out of its own 8 answers vs a probe trained on its hidden states; the closed agent before the call exists is the preregistered confirmation (G40,
within template and tool). (d) Prevention, predicted: observed vs predicted live reduction, with the failed
predictions (G11b's two misses and the position gate of G42) shown as well.
Layout: drawn at print size (6.75 in = ICML text width), so every font size is a true point size; axes are placed in
inches, and the three legends share one strip at the bottom.
  python scripts/214_fig_teaser_v6.py  -> paper/figures/teaser_v6.pdf, results/figures/teaser_v6.png
"""
import importlib.util
import logging
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _mod("t173", "scripts/173_fig_teaser_greybox.py")           # geometry() from scripts/134
L160 = _mod("t160", "scripts/160_fig_law.py")
plt = T.plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # the condensed face reports weight 380
plt.rcParams.update({"font.family": "DejaVu Sans Condensed", "pdf.fonttype": 42, "axes.linewidth": 0.6})  # the papers's
# figure family (DejaVu Sans), condensed: the four panels fit 6.75 in at true 6-7.5 pt
BLUE, ORANGE, AQUA, INK, MUTE = "#2A78D6", "#EB6834", "#1BAF7A", "#1B2126", "#5D6873"
GRID, BAND, FRAME = "#E4E7EA", "#F1F3F5", "#C9CED3"
W, H = 6.75, 2.55                                               # inches: print size
FS_TITLE, FS_SUB, FS_LAB, FS_TICK, FS_LEG = 7.5, 6.1, 6.5, 6.0, 6.0
LAB_W = 1.12                                                    # (c): the label column, inches
PROBE_DY = 0.36                                                 # (c): probe diamonds sit this share of a row above their row


def tint(c, a):
    """Solid colour: `a` of c over white (crisp whiskers without transparency)."""
    r = np.array(to_rgb(c))
    return tuple(1 - a * (1 - r))


def box(x0, y0, w, h):
    return [x0 / W, y0 / H, w / W, h / H]


def style(ax):
    ax.grid(color=GRID, lw=0.45); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(MUTE); ax.spines[s_].set_linewidth(0.6)
    ax.tick_params(labelsize=FS_TICK, colors=MUTE, labelcolor=INK, length=2.2, width=0.6, pad=1.5)


def whisker(ax, x, y, xerr=None, yerr=None, col=BLUE, z=2):
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=tint(col, 0.55), elinewidth=1.1, capsize=1.1,
                capthick=0.8, zorder=z)


def panel_decision(ax):
    """CAPE-Bench rows (held-out) and, in the shaded block below the line, the second benchmarks: tau2-bench (Claude)
    and, when its result exists, AgentDojo (open agents)."""
    R = json.loads((ROOT / "results/t1/g15_training_free.json").read_text())
    CI = json.loads((ROOT / "results/t1/g15_per_agent_ci_exploratory.json").read_text())
    rows = [(N, R[a]["within"]["mean8"], CI[a][1:], R[a]["within"]["white"], BLUE) for a, N in
            (("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss"), ("qwen38", "Qwen3.8"), ("museglimmer", "Muse-Glimmer"))]
    rows.append(("Claude Haiku 5.5, restarts", R["claude"]["restart_mean8"], R["claude"]["restart_mean8_ci"], None, ORANGE))
    f38 = ROOT / "results/t1/g38_gpt_readout.json"
    if f38.exists():
        g38 = json.loads(f38.read_text())
        rows.append(("GPT-6-luna (closed)", g38["A1_at_call"][0], g38["A1_at_call"][1:], None, ORANGE))
        # before the call exists: the PREREGISTERED confirmation (G40), within template and tool (not G38's exploratory 0.765)
        g40 = json.loads((ROOT / "results/t1/g40_gpt_precall_confirm.json").read_text())["C2_within_template_tool"]
        rows.append(("GPT-6-luna, before call", g40[0], g40[1:], None, "hollow"))
    second = []
    f17 = ROOT / "results/t1/g17_agentdojo_readout.json"
    if f17.exists():
        g = json.loads(f17.read_text())
        second += [(f"{N}, AgentDojo", g[a]["within_suite"]["read_out_8"], g[a]["read_out_8_ci"], None, BLUE)
                   for a, N in (("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss")) if a in g]
    g = json.loads((ROOT / "results/t1/g18_tau2_claude_readout.json").read_text())
    second.append(("Claude Haiku 5.5, τ²-bench", g["R1_within_tool_read_out_8"][0], g["R1_within_tool_read_out_8"][1:], None, ORANGE))
    # (another model answering is a comparison outside the method: not shown)
    style(ax)
    ax.grid(axis="y", visible=False)
    X0 = 0.5
    ys, y = [], 0.0
    for k in range(len(second)):                      # bottom block first
        ys.append(y); y += 1
    ydiv = y - 0.5 + 0.15; y += 0.3
    ys2 = []
    for k in range(len(rows)):
        ys2.append(y); y += 1
    ylo, yhi = -0.6, y - 0.4
    # the second benchmarks: a shaded block (through the label column) under a thin line
    lab_w = LAB_W / (ax.get_position().width * W)      # label column, in axes fractions
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.add_patch(Rectangle((-lab_w, ylo), 1 + lab_w, ydiv - ylo, transform=tr, facecolor=BAND, edgecolor="none",
                           zorder=0, clip_on=False))
    ax.plot([-lab_w, 1], [ydiv, ydiv], transform=tr, color=MUTE, lw=0.5, zorder=1, clip_on=False)
    ax.text(X0 + 0.007, ys[-1], "second benchmarks", fontsize=FS_TICK, color=MUTE, style="italic", va="center", ha="left")
    allrows = list(zip(second[::-1], ys)) + list(zip(rows[::-1], ys2))
    for (name, eight, ci_, wb, col), yy in allrows:
        c_ = ORANGE if col == "hollow" else col
        whisker(ax, [eight], [yy], xerr=[[eight - ci_[0]], [ci_[1] - eight]], col=c_)
        if wb is not None:                            # the probe: x exact, drawn PROBE_DY of a row above (never hidden by the dot)
            ax.scatter([wb], [yy + PROBE_DY], s=18, marker="D", facecolor="white", edgecolor=INK, lw=0.8, zorder=3)
        if col == "hollow":
            ax.scatter([eight], [yy], s=30, facecolor="white", edgecolor=ORANGE, lw=1.3, zorder=4)
        else:
            ax.scatter([eight], [yy], s=34, color=c_, edgecolor="white", lw=0.7, zorder=4)
        ax.text(-0.025, yy, name, transform=tr, fontsize=FS_LAB - 0.3, color=INK, ha="right", va="center")
    for a_ in ax.collections + ax.lines:
        a_.set_clip_on(False)                             # markers at 0.99 stay whole at the axis end
    ax.set_xlim(X0, 1.0); ax.set_ylim(ylo, yhi); ax.set_yticks([])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("violations ranked above allowed calls (AUROC)", fontsize=FS_LAB, color=INK, labelpad=2)
    lb, lo = tint(BLUE, 0.55), tint(ORANGE, 0.55)
    h = [(Line2D([], [], color=lb, lw=1.25), Line2D([], [], ls="", marker="o", ms=5.6, mfc=BLUE, mec="white", mew=0.7)),
         (Line2D([], [], color=lo, lw=1.25), Line2D([], [], ls="", marker="o", ms=5.6, mfc=ORANGE, mec="white", mew=0.7)),
         (Line2D([], [], color=lo, lw=1.25), Line2D([], [], ls="", marker="o", ms=5.0, mfc="white", mec=ORANGE, mew=1.3)),
         Line2D([], [], ls="", marker="D", ms=4.0, mfc="white", mec=INK, mew=0.8)]
    lab = ["open agent: its answers", "closed agent, API only", "before the call exists", "probe trained on hidden states"]
    return h, lab


FS_TAG = 6.0                                                    # (d): point tags
TAG_D = {  # (d): (agent, test) -> tag position (data units), alignment, tag; placed clear of points, whiskers and text
    ("Qwen3.8", "g12"): (0.075, -0.04, "left", "Qwen3.8"), ("gpt-oss", "g12"): (0.19, 0.12, "left", "gpt-oss"),
    ("Muse-Glimmer", "g12"): (0.215, 0.36, "right", "Muse"), ("Claude Haiku 5.5", "g12"): (0.425, 0.27, "left", "Claude"),
    ("Gemma-4", "g12"): (0.62, 0.66, "right", "Gemma-4"), ("gpt-oss", "g13"): (0.625, 0.425, "left", "gpt-oss\n(AD)"),
    ("Gemma-4", "g13"): (1.06, 0.565, "right", "Gemma-4\n(AD)"), "g42": (0.69, 0.81, "right", "GPT-6-luna,\npre-call")}


def panel_law(ax):
    style(ax)
    pts = [p for p in L160.points() if p[6] in ("g12", "g13")]
    xs = np.array([-0.05, 1.05])
    ax.fill_between(xs, xs - 0.15, xs + 0.15, color=BAND, lw=0, zorder=0)
    ax.plot(xs, xs, color=MUTE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for name, x, y, lo, hi, c, kind in pts:
        col = ORANGE if "Claude" in name else BLUE
        whisker(ax, [x], [y], yerr=[[y - lo], [hi - y]], col=col)
        ax.scatter([x], [y], s=30, marker="s" if kind == "g13" else "o", color=col, edgecolor="white", lw=0.7, zorder=3)
        tx, ty, ha, tag = TAG_D[(name, kind)]
        ax.text(tx, ty, tag, fontsize=FS_TAG, color=INK, ha=ha, va="center", linespacing=0.95, zorder=5)
    g42 = json.loads((ROOT / "results/t1/g42_gate_law.json").read_text())
    x, y = g42["predicted"]["pre"], g42["observed"]["pre"]   # the pre-call gate on a closed agent
    lo, hi = g42["observed_ci"]["pre"]
    whisker(ax, [x], [y], yerr=[[y - lo], [hi - y]], col=ORANGE)
    ax.scatter([x], [y], s=34, marker="D", color=ORANGE, edgecolor="white", lw=0.7, zorder=4)
    tx, ty, ha, tag = TAG_D["g42"]
    ax.text(tx, ty, tag, fontsize=FS_TAG, color=INK, ha=ha, va="center", linespacing=0.95, zorder=5)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.12, 1.05)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"]); ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel("predicted from offline flag rates", fontsize=FS_LAB, color=INK, labelpad=2)
    ax.set_ylabel("observed live reduction", fontsize=FS_LAB, color=INK, labelpad=1)
    e12 = np.mean([abs(y - x) for _, x, y, *_r, kind in pts if kind == "g12"])
    eall = np.mean([abs(y - x) for _, x, y, *_r in pts])
    ax.text(0.03, 0.99, f"law: mean |error| {e12:.3f}", transform=ax.transAxes, fontsize=FS_TICK, color=INK, va="top")
    ax.text(0.03, 0.91, f"with injections: {eall:.3f}", transform=ax.transAxes, fontsize=FS_TICK, color=MUTE, va="top")
    dot = dict(ls="", marker="o", ms=5.2, mec="white", mew=0.7)
    h = [(Line2D([], [], mfc=BLUE, **dot), Line2D([], [], mfc=ORANGE, **dot)),   # open agents and Claude (orange: closed)
         Line2D([], [], ls="", marker="s", ms=4.8, mfc=BLUE, mec="white", mew=0.7),
         Line2D([], [], ls="", marker="D", ms=4.6, mfc=ORANGE, mec="white", mew=0.7)]
    lab = ["violations", "injections", "GPT-6-luna, pre-call gate"]
    return h, lab


def main():
    Xh, Xa, src = T.geometry()
    fig = plt.figure(figsize=(W, H), facecolor="white")
    TOP = H - 0.35                                    # all axes: top
    BOT, BOT_AB = 0.53, 0.33                          # (c, d) axes bottom; (a, b) frames bottom
    LEG_TOP, XLAB_TOP = 0.255, 0.385                  # the legend strip; the x-axis titles (inches from the bottom)
    xa, xb, wab = 0.03, 1.22, 1.10                    # (a), (b) frames
    xc0 = 2.52; xc, wc = xc0 + LAB_W, 1.34            # (c): label column from xc0, axis from xc
    xd0 = 5.14; xd, wd = 5.41, 1.31                   # (d): block from xd0, axis from xd
    TOP_AB = TOP - 0.10                               # (a, b) frames: top, under the two-line subtitles
    ax = [fig.add_axes(box(xa, BOT_AB, wab, TOP_AB - BOT_AB)), fig.add_axes(box(xb, BOT_AB, wab, TOP_AB - BOT_AB)),
          fig.add_axes(box(xc, BOT, wc, TOP - BOT)), fig.add_axes(box(xd, BOT, wd, TOP - BOT))]
    for a_, X in zip(ax[:2], (Xh, Xa)):
        T.T.scatter(a_, X, src, s=2.4)
        a_.margins(0.035)
        for sp_ in a_.spines.values():
            sp_.set_color(FRAME); sp_.set_linewidth(0.5)
    hc, lc = panel_decision(ax[2])
    hd, ld = panel_law(ax[3])
    for a_, xmid in ((ax[2], (xc0 + xc + wc) / 2), (ax[3], (xd0 + W) / 2 - 0.03)):   # x-axis titles at one height, in inches
        a_.xaxis.set_label_coords((xmid - a_.get_position().x0 * W) / (a_.get_position().width * W),
                                  -(BOT - XLAB_TOP) / (TOP - BOT))
        a_.xaxis.label.set_va("top")
    heads = [(xa, "a", "Hidden geometry", f"{T.T.NAME} hidden states,\nclassical MDS"),       # graybox.geometry.mds
             (xb, "b", "From its answers", "its answers, exact Fisher–Rao,\nclassical MDS; ρ = 0.74"),   # then rotated onto (a)
             (xc0, "c", "Monitor and forecast", "held-out calls; 95% CI; preregistered"),
             (xd0, "d", "Prevent, predicted", "live; 95% CI; preregistered")]
    for x0, k, title, sub in heads:
        fig.text(x0 / W, (H - 0.03) / H, f"({k}) {title}", fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")
        fig.text(x0 / W, (H - 0.175) / H, sub, fontsize=FS_SUB, color=MUTE, va="top", linespacing=1.1)
    # one legend strip: context type under (a, b), the markers of (c), the markers of (d)
    cat = [Line2D([], [], ls="", marker="o", ms=3.6, mfc=col, mec="none", alpha=0.9) for lab, col in T.T.SRC.values()]
    catl = [lab for lab, _ in T.T.SRC.values()]
    kw = dict(frameon=False, fontsize=FS_LEG, handletextpad=0.35, labelspacing=0.3, borderaxespad=0, borderpad=0,
              handlelength=1.3, columnspacing=0.9, labelcolor=INK)
    from matplotlib.legend_handler import HandlerTuple
    l1 = fig.legend(cat, catl, loc="upper left", bbox_to_anchor=(xa / W, LEG_TOP / H), ncol=3, **{**kw, "handlelength": 0.7})
    l2 = fig.legend(hc, lc, loc="upper left", bbox_to_anchor=(xc0 / W, LEG_TOP / H), ncol=2,
                    handler_map={tuple: HandlerTuple(ndivide=1)}, **kw)
    l3 = fig.legend(hd, ld, loc="upper right", bbox_to_anchor=((W - 0.02) / W, LEG_TOP / H), ncol=2,
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)}, **{**kw, "handlelength": 1.4})
    for out in (ROOT / "paper/figures/teaser_v6.pdf", ROOT / "results/figures/teaser_v6.png"):
        fig.savefig(out, dpi=300)
    print("saved", fig.get_size_inches())
    return fig


if __name__ == "__main__":
    main()
