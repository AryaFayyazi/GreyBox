#!/usr/bin/env python
"""L3c re-analysis with GRADED moderation baselines (not preregistered; same role as scripts/67 for L3b/L3d).

L3c's E1 baseline is Qwen3Guard's 3-level verdict (mod_score), which L3e showed is weak and tie-heavy. Here E1 and E2
are recomputed against graded P(unsafe) from Qwen3Guard-Gen-8B and Llama-Guard-3-8B (outputs/livebench/
moderation_graded.parquet), on exactly L3c's eval rows (scripts/51 construction). Bootstrap over conversations, 2,000.
python scripts/51b_l3c_graded.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
a51 = importlib.import_module("51_claude_batch_analysis")
B = 2000


def boot(f, n, rng):
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try:
            v.append(f(i))
        except ValueError:
            continue
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


R = np.load(a51.BUNDLE, allow_pickle=True)
ev = pd.concat([pd.read_parquet(a51.CB / f"{s}.parquet").assign(part=s) for s in ("evalA", "evalB")], ignore_index=True)
ev = ev.merge(pd.concat([pd.read_parquet(a51.CB / f"{s}_refusals.parquet") for s in ("evalA", "evalB")]), on="id")
ev = ev[ev.refused >= 0].reset_index(drop=True)
sc, _ = a51.scores(ev, R)
r = sc["will_refuse"]
G = pd.read_parquet(a51.LB / "moderation_graded.parquet").set_index("id")
y_h, y_r = ev.harmful.values.astype(int), ev.refused.values.astype(int)
hm = y_h == 1
event = (hm & (y_r == 0)).astype(int)
rng = np.random.default_rng(0)
res = {"n_eval": int(len(ev)), "n_harmful": int(hm.sum()), "graded_missing": {k: int(G[k].reindex(ev.id).isna().sum()) for k in ("qg_p", "lg_p")}}
for k in ("qg_p", "lg_p"):
    g = G[k].reindex(ev.id).values
    ok = np.isfinite(g)
    yy, rr, gg = y_r[hm & ok], r[hm & ok], g[hm & ok]
    f1 = lambda i: roc_auc_score(yy[i], rr[i]) - roc_auc_score(yy[i], gg[i])  # noqa: E731
    e, rk, gm = event[ok], (g * (1 - r))[ok], g[ok]
    f2 = lambda i: average_precision_score(e[i], rk[i]) - average_precision_score(e[i], gm[i])  # noqa: E731
    res[k] = {"E1_refusal": {"greybox": round(float(roc_auc_score(yy, rr)), 4), "graded": round(float(roc_auc_score(yy, gg)), 4),
                             "diff": round(float(f1(np.arange(len(yy)))), 4), "ci95": boot(f1, len(yy), rng)},
              "E2_compliance_ap": {"graded_x_not_refuse": round(float(average_precision_score(e, rk)), 4), "graded": round(float(average_precision_score(e, gm)), 4),
                                   "diff": round(float(f2(np.arange(len(e)))), 4), "ci95": boot(f2, len(e), rng)}}
(ROOT / "results/l3/l3c_graded_reanalysis.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
