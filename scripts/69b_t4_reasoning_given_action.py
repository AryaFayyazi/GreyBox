#!/usr/bin/env python
"""Corollary 3 check (exploratory): given the proposed action, does the agent's hidden reasoning add information?
T4's two reader variants on the same 12,128 fresh units: READER_act (transcript + action) and READER_cot (transcript +
the agent's reasoning for this turn + action), each with T4's probe at T4's dev-chosen layer (act L32, cot L30).
Reports the paired within-template AUROC difference, scenario-cluster bootstrap (1,000, seed 69), and the same
difference at the 5% review budget. Corollary 3 predicts a difference of about 0.
python scripts/69b_t4_reasoning_given_action.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

I, OUT = Path("/data/models/graybox_ws/outputs/interrogate"), Path("/data/models/graybox_ws/outputs/t4")
LAYER = {"act": 32, "cot": 30}                                  # T4's dev choice (results/t1/t4_reader.json)
Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
yd, yf = Ud.y.values.astype(int), Uf.y.values.astype(int)
S = {}
for var, l in LAYER.items():
    Hd, Hf = np.load(OUT / f"clean_{var}.npz"), np.load(OUT / f"fresh_{var}.npz")
    j = list(Hd["layers"]).index(l)
    m = make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=4000)).fit(Hd["h"][:, j].astype(np.float32), yd)
    S[var] = m.predict_proba(Hf["h"][:, j].astype(np.float32))[:, 1]
ok = np.isfinite(np.load(I / "gemma4_fresh_decision_hidden.npz")["I_dec"])
y, st, g = yf[ok], Uf.template.values[ok], Uf.scenario_id.values[ok]
a, c = S["act"][ok], S["cot"][ok]
ref = json.loads((ROOT / "results/t1/t4_reader.json").read_text())["auroc_fresh"]
res = {"n": int(ok.sum()), "auroc": {"READER_act": round(within_auroc(y, a, st), 4), "READER_cot": round(within_auroc(y, c, st), 4)}}
assert abs(res["auroc"]["READER_act"] - ref["READER_act"]) < 2e-3 and abs(res["auroc"]["READER_cot"] - ref["READER_cot"]) < 2e-3, (res, ref)
d = cluster_boot(lambda i: within_auroc(y[i], c[i], st[i]) - within_auroc(y[i], a[i], st[i]), g, 1000, seed=69)
res["cot_minus_act"] = [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
for k, v in (("act", a), ("cot", c)):
    t = np.quantile(v[y == 0], 0.95)
    res.setdefault("caught_at_5pct", {})[k] = round(float((v[y == 1] >= t).mean()), 4)
(ROOT / "results/t1/t4_reasoning_given_action_exploratory.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
