#!/usr/bin/env python
"""Build the Closed Agent Monitor page: web/claude_monitor_demo.html + episode data + the monitor's 3D space.

Inputs: web/data/claude_monitor_demo.json (18 Claude episodes with per-action reader/judge scores; scored steps are
T8b units in order) and outputs/t8b/space.json (scripts/78 stage=coords). Each scored step gets `pi`, the index of its
point in the space, matched by episode and unit order and asserted by action text and score.
python web/build_monitor_demo.py <space.json> <out.html>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
space_p, out = Path(sys.argv[1]), Path(sys.argv[2])
D = json.loads((ROOT / "web/data/claude_monitor_demo.json").read_text())
if space_p.exists():
    S = json.loads(space_p.read_text())
    x, y, z = (np.array(S[k], float) for k in ("x", "y2", "z"))
    # display box: X linear in the logit (clipped to the 0.5th-99.5th percentile range, clipped points flagged); Y/Z share one scale, unclipped
    lo, hi = np.quantile(x, [0.005, 0.995])
    X = 2 * (np.clip(x, lo, hi) - lo) / (hi - lo) - 1
    sc = 1.0 / max(np.abs(np.r_[y, z]).max(), 1e-9)          # no clipping on axes 2-3: violations are often the extremes
    Y, Z = y * sc * 0.8, z * sc * 0.8
    cut = float(2 * (np.clip(S["cut_logit_label_free"], lo, hi) - lo) / (hi - lo) - 1)
    by_ep = {}
    for i, e in enumerate(S["episode_id"]):
        by_ep.setdefault(e, []).append(i)
    for e in D["episodes"]:
        idx, k = by_ep.get(e["id"], []), 0
        for s in e["steps"]:
            if s.get("risk") is None:
                continue
            i = idx[k]; k += 1
            assert S["act"][i].split(" ", 1)[0] == s["tool"], (S["act"][i][:60], s["tool"])
            assert abs(1 / (1 + np.exp(-x[i])) - s["risk"]) < 2e-3, (e["id"], x[i], s["risk"])
            s["pi"] = i
        assert k == len(idx), e["id"]
D["pooled"] = json.loads((ROOT / "results/t1/t8b_breakdown_exploratory.json").read_text())["auroc_pooled"]
D["label_free"] = json.loads((ROOT / "results/t1/label_free_cuts.json").read_text())["claude_T8b"]   # the 5% cut set without labels
if space_p.exists():
    xs = np.sort(x)
    pct = (np.searchsorted(xs, x, "left") + np.searchsorted(xs, x, "right")) / 2 / len(x)   # share of actions with a lower score (midrank)
    esc = x >= S["cut_logit_label_free"]
    assert abs(D["label_free"]["READER"]["label_free_cut"] - esc[np.array(S["y"]) == 1].mean()) < 1e-3, "space cut must match the label-free result"
    D["space"] = {"p": np.round(np.c_[X, Y, Z], 3).ravel().tolist(), "y": S["y"], "act": S["act"], "ep": S["episode_id"],
                  "s": [float(f"{v:.3g}") for v in 1 / (1 + np.exp(-x))], "pct": np.round(pct, 4).tolist(), "esc": esc.astype(int).tolist(),
                  "cut": round(cut, 4), "logit_range": [round(float(lo), 2), round(float(hi), 2)],
                  "var23": S["var_explained_axes23"], "clipped": int(((x < lo) | (x > hi)).sum())}
t = (ROOT / "web/claude_monitor_demo.html").read_text()
out.write_text(t.replace("<!--DATA-->", "<script>window.DEMO=" + json.dumps(D, separators=(",", ":")).replace("</", "<\\/") + ";</script>"))
print("wrote", out, f"{out.stat().st_size / 1e6:.2f} MB")
