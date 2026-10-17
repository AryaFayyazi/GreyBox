#!/usr/bin/env python
"""T7 R2 as the prereg TEXT defines it: gpt-oss calls with a logged CAPE gate score (1,267 calls). scripts/73's code
used all calls with a logged score (1,805, including Gemma calls scored by Gemma's gate); both are reported in the
record. This saves the prereg-text version (same bootstrap: episodes, 1,000 resamples, seed 72).
python scripts/73b_t7_r2_prereg_text.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

t7 = importlib.import_module("73_t7_agentdojo")
U = t7.units()
U["READER"] = np.load(t7.OUT / "reader.npz")["p"]
U["J_act"] = pd.read_parquet(t7.OUT / "judge.parquet").p.values
m = np.isfinite(U.cape.values) & (U.actor == "gptoss20b").values
y, st, g = U.y.values[m], U.suite.values[m], U.episode.values[m]
r, c = U.READER.values[m], U.cape.values[m]
d = cluster_boot(lambda i: within_auroc(y[i], r[i], st[i]) - within_auroc(y[i], c[i], st[i]), g, 1000, seed=72)
res = {"n": int(m.sum()), "injected": int(y.sum()), "READER": round(within_auroc(y, r, st), 4), "cape": round(within_auroc(y, c, st), 4),
       "R2_READER_minus_CAPE": [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
(ROOT / "results/t1/t7_r2_prereg_text.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
