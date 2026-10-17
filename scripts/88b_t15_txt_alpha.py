#!/usr/bin/env python
"""T15 sensitivity (exploratory): the TXT ridge chose alpha = 1, the lower edge of the locked grid. Refit TXT with alpha
0.1 and 0.01 on development (no CV) and report fresh R^2 (128 PCs) and the E2 readout on the TXT replica.
python scripts/88b_t15_txt_alpha.py"""
import importlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import within_auroc  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
I = Path("/data/models/graybox_ws/outputs/interrogate")
t9 = importlib.import_module("77_t9_text_baseline")
Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
Dd, Df = np.load(I / "gemma4_clean_decision_hidden.npz"), np.load(I / "gemma4_fresh_decision_hidden.npz")
okd, okf = Dd["ok"].astype(bool), Df["ok"].astype(bool)
Yd_full, Yf_full = Dd["h"][okd].reshape(okd.sum(), -1).astype(np.float32), Df["h"][okf].reshape(okf.sum(), -1).astype(np.float32)
sc = StandardScaler().fit(Yd_full); pca = PCA(128, random_state=0).fit(sc.transform(Yd_full))
Yd, Yf = pca.transform(sc.transform(Yd_full)), pca.transform(sc.transform(Yf_full))
Td, Tf = t9.gemma_texts("clean"), t9.gemma_texts("fresh")
tx = t9.TXT(); tx.v = [t9.vec(), t9.vec()]
Xd, Xf = tx._X(Td.ctx.values[okd], Td.act.values[okd], True), tx._X(Tf.ctx.values[okf], Tf.act.values[okf], False)
probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000)).fit(Yd, Ud.y.values[okd].astype(int))
yf, st = Uf.y.values[okf].astype(int), Uf.template.values[okf]
res = {}
for a in (0.1, 0.01):
    rec = Ridge(alpha=a).fit(Xd, Yd).predict(Xf)
    res[str(a)] = {"r2_fresh": round(float(r2_score(Yf, rec, multioutput="variance_weighted")), 4), "E2_auroc_txt_replica": round(within_auroc(yf, probe.decision_function(rec), st), 4)}
    print(a, res[str(a)], flush=True)
(ROOT / "results/t1/t15_txt_alpha_exploratory.json").write_text(json.dumps(res, indent=1))
