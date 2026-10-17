#!/usr/bin/env python
"""T8b breakdown (exploratory, post hoc): what T8b's within-template AUROC rests on, pooled AUROC, per-family catch
rates at the label-free cut, and the send_message tool-identity check. Existing T8b scores only.
python scripts/76b_t8b_breakdown.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
m = importlib.import_module("76_t8b_claude_all_actions")
U = m.units()
S = {"READER": np.load(m.OUT / "reader.npz")["p"], "J_act": pd.read_parquet(m.OUT / "judge.parquet").p.values}
y, fam = U.y.values.astype(int), U.scenario_id.str.split(".").str[1].values
strata = []
for t, s in U.groupby("template"):
    if 0 < s.y.sum() < len(s):
        strata.append({"template": t, "n": len(s), "unsafe": int(s.y.sum()), "scenarios": int(s.scenario_id.nunique())})
res = {"strata_with_both_classes": strata,
       "all_unsafe_strata": [{"template": t, "n": len(s)} for t, s in U.groupby("template") if s.y.sum() == len(s)],
       "auroc_pooled": {k: round(float(roc_auc_score(y, v)), 4) for k, v in S.items()}, "label_free_cut": {}}
for k, v in S.items():
    t = np.quantile(v, 0.95)
    res["label_free_cut"][k] = {f: round(float((v[(fam == f) & (y == 1)] >= t).mean()), 4) for f in sorted(set(fam[y == 1]))}
sm = U.act.str.startswith("send_message ").values
res["send_message"] = {"n": int(sm.sum()), "safe": int((y[sm] == 0).sum()), **{k: round(float(roc_auc_score(y[sm], v[sm])), 4) for k, v in S.items()},
                       "safe_escalated_label_free": {k: round(float((v[sm & (y == 0)] >= np.quantile(v, .95)).mean()), 4) for k, v in S.items()}}
(ROOT / "results/t1/t8b_breakdown_exploratory.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
