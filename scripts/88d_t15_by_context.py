#!/usr/bin/env python
"""T15 exploratory: does replication degrade as the transcript grows? An associative-memory view of attention
(e.g. Wang, 2026, alexwang.ai/posts/associative-memory) predicts more interference with more stored tokens. T15's ridge
replica (reader layer 32 -> 128 agent-state PCs), fresh steps binned by context length (characters of the rendered
context, quintiles) and by the number of earlier tool calls. Per bin: variance-weighted R^2 and the probe AUROC on
the replica vs the true state (pooled within the bin).
python scripts/88d_t15_by_context.py"""
import importlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
I, T4 = Path("/data/models/graybox_ws/outputs/interrogate"), Path("/data/models/graybox_ws/outputs/t4")
t9 = importlib.import_module("77_t9_text_baseline")
Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
Dd, Df = np.load(I / "gemma4_clean_decision_hidden.npz"), np.load(I / "gemma4_fresh_decision_hidden.npz")
okd, okf = Dd["ok"].astype(bool), Df["ok"].astype(bool)
Yd_full, Yf_full = Dd["h"][okd].reshape(okd.sum(), -1).astype(np.float32), Df["h"][okf].reshape(okf.sum(), -1).astype(np.float32)
sc = StandardScaler().fit(Yd_full); pca = PCA(128, random_state=0).fit(sc.transform(Yd_full))
Yd, Yf = pca.transform(sc.transform(Yd_full)), pca.transform(sc.transform(Yf_full))
Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz"); j = [int(x) for x in Hd["layers"]].index(32)
xs = StandardScaler().fit(Hd["h"][okd, j].astype(np.float32))
rep = Ridge(alpha=1000).fit(xs.transform(Hd["h"][okd, j].astype(np.float32)), Yd).predict(xs.transform(Hf["h"][okf, j].astype(np.float32)))
probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000)).fit(Yd, Ud.y.values[okd].astype(int))
y = Uf.y.values[okf].astype(int)
Tf = t9.gemma_texts("fresh")
clen = Tf.ctx.str.len().values[okf]
ncalls = Tf.ctx.str.count(r"\[Agent called\]").values[okf]
st, sr = probe.decision_function(Yf), probe.decision_function(rep)
res = {"overall_r2": round(float(r2_score(Yf, rep, multioutput="variance_weighted")), 4)}
for name, v in (("context_chars_quintile", clen), ("earlier_tool_calls", ncalls)):
    if name == "earlier_tool_calls":
        bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)]
        idx = [(f"{a}-{b}", (v >= a) & (v <= b)) for a, b in bins]
    else:
        q = np.quantile(v, [0, .2, .4, .6, .8, 1])
        idx = [(f"{int(q[k])}-{int(q[k+1])}", (v >= q[k]) & (v <= q[k + 1]) if k == 4 else (v >= q[k]) & (v < q[k + 1])) for k in range(5)]
    out = {}
    for lab, m in idx:
        if m.sum() < 200:
            continue
        o = {"n": int(m.sum()), "r2": round(float(r2_score(Yf[m], rep[m], multioutput="variance_weighted")), 4)}
        if 0 < y[m].sum() < m.sum():
            o.update(auroc_true=round(float(roc_auc_score(y[m], st[m])), 4), auroc_replica=round(float(roc_auc_score(y[m], sr[m])), 4), unsafe=int(y[m].sum()))
        out[lab] = o
    res[name] = out
    rho = np.corrcoef(v, ((Yf - rep) ** 2).sum(1))[0, 1]
    res[name + "_corr_with_squared_error"] = round(float(rho), 4)
(ROOT / "results/t1/t15_by_context_exploratory.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
