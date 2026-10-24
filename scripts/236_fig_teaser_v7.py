#!/usr/bin/env python
"""First-page figure, v7: the paper's chain in four panels, one message each (redesign of scripts/214, v6).
  (a) What the API reveals: Llama-3.1 8B's hidden geometry (classical MDS of chord distances between hidden states) and
      the same 1,440 held-out contexts placed by its yes/no answers only (exact Fisher-Rao, classical MDS, then an
      orthogonal Procrustes rotation onto the hidden map for display). Computation: scripts/134 via scripts/173's
      geometry(), unchanged. Spearman 0.74, range over 8 models 0.41-0.74 (paper/geometry_numbers.tex, \\GOneAns).
  (b) Monitor at the call: the fixed read-out of the agent's own 8 answers, AUROC with 95% CI per agent.
      open agents: results/t1/g15_training_free.json (points), g15_per_agent_ci_exploratory.json (intervals);
      GPT-6-luna: g38_gpt_readout.json A1_at_call; Claude restarts: g15 claude restart_mean8(_ci);
      Claude tau2-bench: g18_tau2_claude_readout.json R1_within_tool_read_out_8.
  (c) Forecast before the call: what the agent says about its next action vs what its sampled next action does,
      AUROC among calls at the same position. GPT says: g45_per_agent_ci.json own_within_position.gpt; Claude says:
      g46_rollout_forecast.json within_position.report = 0.4746, interval [0.4323, 0.5109] from scripts/228's
      cluster bootstrap (not stored in a results file); Claude does: g46 R1_own_within_position.
  (d) Prevent, with no human: violating episodes (%), live. Claude: g47_selfgate.json violating_episodes;
      GPT-6-luna: g39b_matched_gate.json violating_episodes (a checker reviews the held calls).
Drawn at print size (6.75 in = ICML text width; no tight bbox), so every font size is a true point size.
  python scripts/236_fig_teaser_v7.py  -> paper/figures/teaser_v7.pdf, results/figures/teaser_v7.png
"""
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.transforms import blended_transform_factory, offset_copy

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _mod("t173", "scripts/173_fig_teaser_greybox.py")           # geometry() from scripts/134
plt = T.plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({"font.family": "DejaVu Sans Condensed", "pdf.fonttype": 42, "axes.linewidth": 0.6})
BLUE, ORANGE, AQUA, INK, MUTE, LIGHT = "#2A78D6", "#EB6834", "#1BAF7A", "#1B2126", "#5D6873", "#E6E9ED"
FRAME = "#C9CED3"
W, H = 6.75, 2.45                                               # inches: print size
FS_TITLE, FS_SUB, FS_LAB, FS_TICK, FS_HERO = 7.5, 6.2, 6.3, 6.0, 9.0
RHO, RHO_RANGE = 0.74, "0.41–0.74"                              # paper/geometry_numbers.tex (\GOneAns); G1 table
TOP = H - 0.40                                                  # top of every axis
BOT = 0.36                                                      # bottom of the (b, c, d) axes
J = lambda f: json.loads((ROOT / "results/t1" / f).read_text())  # noqa: E731


def tint(c, a):
    r = np.array(to_rgb(c))
    return tuple(1 - a * (1 - r))


def box(x0, y0, w, h):
    return [x0 / W, y0 / H, w / W, h / H]


def style(ax):
    ax.grid(axis="x", color=LIGHT, lw=0.5); ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(MUTE); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", labelsize=FS_TICK, colors=MUTE, labelcolor=INK, length=2.2, width=0.6, pad=1.5)
    ax.set_yticks([])


def inch_x(ax, dx):
    """Transform: x offset by dx inches from the axis' left edge, y in data units."""
    return offset_copy(blended_transform_factory(ax.transAxes, ax.transData), fig=ax.figure, x=dx, y=0, units="inches")


# ---------------------------------------------------------------- (a)
KINDS = [("harmful", "harmful"), ("benign", "benign"), ("sst2", "review"),       # kinds of context (scripts/134 SRC)
         ("boolq", "question"), ("gsm8k", "maths"), ("wiki", "encyclopedia")]


def panel_geometry(fig, x0, w_map, w_col):
    Xh, Xa, src = T.geometry()
    X = np.vstack([Xh, Xa])
    lo, hi = X.min(0), X.max(0)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    h_map = w_map * (hi[1] - lo[1]) / (hi[0] - lo[0])            # equal aspect: one unit is the same in x and y
    gap = 0.10
    y_top = TOP - h_map
    y_bot = y_top - gap - h_map
    axs = []
    for X_, yb, tag in ((Xh, y_top, "its hidden states"), (Xa, y_bot, "its answers only")):
        ax = fig.add_axes(box(x0, yb, w_map, h_map))
        for k, _ in KINDS:
            i = src == k
            ax.scatter(X_[i, 0], X_[i, 1], s=1.6, color=T.T.SRC[k][1], alpha=0.7, lw=0, rasterized=True)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(True); sp_.set_color(FRAME); sp_.set_linewidth(0.5)
        ax.text(0.97, 0.94, tag, transform=ax.transAxes, fontsize=FS_TICK, color=INK, ha="right", va="top")
        axs.append(ax)
    # right column: the agreement between the two maps, then the key to the colours
    xc = (x0 + w_map + 0.08) / W
    ymid = (y_top - gap / 2) / H
    fig.text(xc, ymid + 0.30 / H, "Spearman", fontsize=FS_TICK, color=MUTE, va="bottom")
    fig.text(xc, ymid + 0.30 / H, "ρ = %.2f" % RHO, fontsize=FS_HERO, fontweight="bold", color=INK, va="top")
    fig.text(xc, ymid + 0.12 / H, f"{RHO_RANGE}\nover 8 models", fontsize=FS_TICK, color=MUTE, va="top",
             linespacing=1.1)
    y = ymid - 0.22 / H
    for k, lab in KINDS:
        fig.text(xc, y, "●", fontsize=5.0, color=T.T.SRC[k][1], va="center")
        fig.text(xc + 0.075 / W, y, lab, fontsize=FS_TICK, color=INK, va="center")
        y -= 0.118 / H
    return axs, (y_bot, x0 + w_map + w_col)


# ---------------------------------------------------------------- shared dot-and-interval rows
def dot_row(ax, y, m, lo, hi, col, hollow=False, value=True, val_dx=None):
    ax.plot([lo, hi], [y, y], color=tint(col, 0.5), lw=1.6, solid_capstyle="butt", zorder=2)
    ax.scatter([m], [y], s=26 if hollow else 30, facecolor="white" if hollow else col, edgecolor=col if hollow else "white",
               lw=1.2 if hollow else 0.7, zorder=4, clip_on=False)
    if value:
        ax.text(1.0, y, f"{m:.2f}", transform=offset_copy(blended_transform_factory(ax.transAxes, ax.transData),
                fig=ax.figure, x=val_dx, y=0, units="inches"), fontsize=FS_TICK, color=INK, ha="left", va="center")


def auroc_axis(ax, xlabel, chance_label=False):
    ax.set_xlim(0.4, 1.0)
    ax.set_xticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])
    ax.axvline(0.5, color=MUTE, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.set_xlabel(xlabel, fontsize=FS_LAB, color=INK, labelpad=1.5)
    if chance_label:
        ax.text(0.5, ax.get_ylim()[0], " chance", fontsize=FS_TICK, color=MUTE, ha="left", va="bottom")


YLO, HEAD_F = -0.6, 0.955                     # (b-d): bottom of the y range; the top header sits at this axes fraction


def set_rows(ax, y_head):
    """y range so that the top group header sits at the same height in every panel."""
    ax.set_ylim(YLO, YLO + (y_head - YLO) / HEAD_F)


def header(ax, y, text, dx=-0.02):
    ax.text(0, y, text, transform=inch_x(ax, dx), fontsize=FS_TICK, color=MUTE, ha="left", va="center",
            style="italic", zorder=6, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))


# ---------------------------------------------------------------- (b)
def panel_monitor(ax, lab_w):
    R, CI = J("g15_training_free.json"), J("g15_per_agent_ci_exploratory.json")
    g38, g18 = J("g38_gpt_readout.json"), J("g18_tau2_claude_readout.json")
    opn = [(N, R[a]["within"]["mean8"], *CI[a][1:]) for a, N in
           (("gemma4", "Gemma-4"), ("gptoss20b", "gpt-oss-20b"), ("qwen38", "Qwen3.8"), ("museglimmer", "Muse-Glimmer"))]
    cls = [("GPT-6-luna", *g38["A1_at_call"]),
           ("Claude, restarts", R["claude"]["restart_mean8"], *R["claude"]["restart_mean8_ci"]),
           ("Claude, τ²-bench", *g18["R1_within_tool_read_out_8"])]
    style(ax)
    rows, y = [], 0.0
    for name, m, lo, hi in cls[::-1]:
        rows.append((y, name, m, lo, hi, ORANGE)); y += 1
    y_hc = y - 0.15; y += 0.95
    for name, m, lo, hi in opn[::-1]:
        rows.append((y, name, m, lo, hi, BLUE)); y += 1
    y_ho = y - 0.15
    set_rows(ax, y_ho)
    for yy, name, m, lo, hi, col in rows:
        dot_row(ax, yy, m, lo, hi, col, val_dx=0.07)
        ax.text(0, yy, name, transform=inch_x(ax, -0.05), fontsize=FS_LAB, color=INK, ha="right", va="center")
    header(ax, y_ho, "open weights", dx=-lab_w)
    header(ax, y_hc, "closed, API only", dx=-lab_w)
    auroc_axis(ax, "AUROC: violations vs allowed calls")
    return {r[1]: r[2:5] for r in rows}


# ---------------------------------------------------------------- (c)
CLAUDE_SAYS = (0.4746, 0.4323, 0.5109)          # g46 within_position.report; interval: scripts/228's cluster bootstrap


def panel_forecast(ax):
    g45, g46 = J("g45_per_agent_ci.json"), J("g46_rollout_forecast.json")
    assert abs(g46["within_position"]["report"] - CLAUDE_SAYS[0]) < 1e-4
    gpt = g45["own_within_position"]["gpt"]
    does = g46["R1_own_within_position"]
    style(ax)
    rows = [(3.35, "says", *gpt, True), (1.0, "says", *CLAUDE_SAYS, True), (0.0, "does", *does, False)]
    for yy, lab, m, lo, hi, hollow in rows:
        dot_row(ax, yy, m, lo, hi, ORANGE, hollow=hollow, val_dx=0.07)
        ax.text(0, yy, lab, transform=inch_x(ax, -0.05), fontsize=FS_LAB, color=INK, ha="right", va="center")
    gap = g46["R3_own_minus_report_position"]              # does - says, same calls, 95% CI
    ax.annotate("", xy=(does[0] - 0.012, 0.16), xytext=(CLAUDE_SAYS[0] + 0.012, 0.84),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text((does[0] + CLAUDE_SAYS[0]) / 2 + 0.035, 0.55, f"+{gap[0]:.2f}", fontsize=FS_TICK, color=INK, ha="left",
            va="center")
    set_rows(ax, 4.3)
    header(ax, 4.3, "GPT-6-luna", dx=-0.30)
    header(ax, 1.95, "Claude Haiku 5.5", dx=-0.30)
    auroc_axis(ax, "AUROC among calls at one position", chance_label=True)
    return {"GPT says": gpt, "Claude says": CLAUDE_SAYS, "Claude does": does, "Claude does - says": gap}


# ---------------------------------------------------------------- (d)
def panel_prevent(ax, lab_w):
    s, b = J("g47_selfgate.json")["violating_episodes"], J("g39b_matched_gate.json")["violating_episodes"]
    style(ax)
    grey = tint(MUTE, 0.35)
    rows = [(6.0, "no gate", s["none"], grey), (5.0, "random holds", s["random"], grey),
            (4.0, "own read-out", s["self"], ORANGE),
            (2.0, "no gate", b["none"], grey), (1.0, "step gate", b["position"], grey),
            (0.0, "pre-call gate", b["pre"], ORANGE)]
    for yy, lab, v, col in rows:
        ax.barh([yy], [100 * v], height=0.62, color=col, lw=0, zorder=2)
        ax.text(100 * v + 1.5, yy, f"{100 * v:.0f}%", fontsize=FS_TICK, color=INK, ha="left", va="center")
        ax.text(0, yy, lab, transform=inch_x(ax, -0.05), fontsize=FS_LAB, color=INK, ha="right", va="center")
    set_rows(ax, 7.05)
    header(ax, 7.05, "Claude Haiku 5.5", dx=-lab_w)
    header(ax, 3.0, "GPT-6-luna (checker reviews held calls)", dx=-lab_w)
    ax.set_xlim(0, 80)
    ax.set_xticks([0, 25, 50, 75], ["0", "25", "50", "75%"])
    ax.set_xlabel("violating episodes, live", fontsize=FS_LAB, color=INK, labelpad=1.5)
    return {lab: v for _, lab, v, _ in rows}, s, b


# ---------------------------------------------------------------- checks
def check_layout(fig):
    """Every text inside the figure; no two texts overlap (bounding boxes, 0.5 pt slack)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(lambda a: hasattr(a, "get_text") and a.get_visible()) if t.get_text().strip()]
    bbs = [(t.get_text(), t.get_window_extent(r)) for t in texts]
    fw, fh = fig.bbox.width, fig.bbox.height
    bad = [f"clipped: {s!r}" for s, b in bbs if b.x0 < 0 or b.y0 < 0 or b.x1 > fw or b.y1 > fh]
    sl = 0.5 * fig.dpi / 72
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            a, b = bbs[i][1], bbs[j][1]
            if a.x0 + sl < b.x1 and b.x0 + sl < a.x1 and a.y0 + sl < b.y1 and b.y0 + sl < a.y1:
                bad.append(f"overlap: {bbs[i][0]!r} / {bbs[j][0]!r}")
    return bad


def main():
    fig = plt.figure(figsize=(W, H), facecolor="white")
    xa, wa_map, wa_col = 0.03, 0.99, 0.60
    xb0, lab_b = 1.76, 0.70; xb, wb = xb0 + lab_b, 0.78
    xc0, lab_c = 3.60, 0.30; xc, wc = xc0 + lab_c, 0.78
    xd0, lab_d = 5.20, 0.60; xd, wd = xd0 + lab_d, 0.90
    panel_geometry(fig, xa, wa_map, wa_col)
    axb = fig.add_axes(box(xb, BOT, wb, TOP - BOT))
    axc = fig.add_axes(box(xc, BOT, wc, TOP - BOT))
    axd = fig.add_axes(box(xd, BOT, wd, TOP - BOT))
    vb = panel_monitor(axb, lab_b)
    vc = panel_forecast(axc)
    vd, s, b = panel_prevent(axd, lab_d)
    heads = [(xa, "a", "What the API reveals", "its answers redraw its hidden map"),
             (xb0, "b", "Monitor at the call", "its own answers rank its violations"),
             (xc0, "c", "Forecast before the call", "what it says vs what it does"),
             (xd0, "d", "Prevent before harm", "its own read-out holds risky calls")]
    for x0, k, title, sub in heads:
        fig.text(x0 / W, (H - 0.04) / H, f"({k}) {title}", fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")
        fig.text(x0 / W, (H - 0.20) / H, sub, fontsize=FS_SUB, color=MUTE, va="top")
    bad = check_layout(fig)
    for out in (ROOT / "paper/figures/teaser_v7.pdf", ROOT / "results/figures/teaser_v7.png"):
        fig.savefig(out, dpi=300)
    print("saved", fig.get_size_inches(), "layout problems:", bad or "none")
    print(json.dumps({"b": vb, "c": vc, "d": vd}, indent=None, default=float))
    return fig


if __name__ == "__main__":
    main()
