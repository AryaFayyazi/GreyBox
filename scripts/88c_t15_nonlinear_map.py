#!/usr/bin/env python
"""T15 sensitivity (exploratory): is the replica limited by the LINEAR map? Zhou & Klindt (2026, arXiv 2610.20851)
show that encoders are linear in the latents when the representation is complete, and nonlinear when it is
undercomplete. Here an MLP map (one hidden layer of 1,024 units, ReLU, early stopping on a 10% development split,
seed 0) replaces T15's ridge map from the reader's layer 32 to the agent's 128 state PCs. The data, target, split and
evaluation are T15's. If the MLP's fresh R^2 is far above the ridge's 0.664, the linear replica understates replication.
python scripts/88c_t15_nonlinear_map.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.stats import within_auroc  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
I, T4 = Path("/data/models/graybox_ws/outputs/interrogate"), Path("/data/models/graybox_ws/outputs/t4")
Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
Dd, Df = np.load(I / "gemma4_clean_decision_hidden.npz"), np.load(I / "gemma4_fresh_decision_hidden.npz")
okd, okf = Dd["ok"].astype(bool), Df["ok"].astype(bool)
Yd_full, Yf_full = Dd["h"][okd].reshape(okd.sum(), -1).astype(np.float32), Df["h"][okf].reshape(okf.sum(), -1).astype(np.float32)
sc = StandardScaler().fit(Yd_full); pca = PCA(128, random_state=0).fit(sc.transform(Yd_full))
Yd, Yf = pca.transform(sc.transform(Yd_full)), pca.transform(sc.transform(Yf_full))
ys = StandardScaler().fit(Yd)
Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz"); j = [int(x) for x in Hd["layers"]].index(32)
xs = StandardScaler().fit(Hd["h"][okd, j].astype(np.float32))
Xd, Xf = xs.transform(Hd["h"][okd, j].astype(np.float32)), xs.transform(Hf["h"][okf, j].astype(np.float32))
res = {}
for alpha in (1e-3, 1e-2, 1e-1):
    m = MLPRegressor(hidden_layer_sizes=(1024,), alpha=alpha, early_stopping=True, validation_fraction=0.1, max_iter=300, random_state=0).fit(Xd, ys.transform(Yd))
    rec = ys.inverse_transform(m.predict(Xf))
    res[str(alpha)] = {"r2_fresh": round(float(r2_score(Yf, rec, multioutput="variance_weighted")), 4), "epochs": int(m.n_iter_)}
    print(alpha, res[str(alpha)], flush=True)
best = max(res, key=lambda k: res[k]["r2_fresh"])
m = MLPRegressor(hidden_layer_sizes=(1024,), alpha=float(best), early_stopping=True, validation_fraction=0.1, max_iter=300, random_state=0).fit(Xd, ys.transform(Yd))
rec = ys.inverse_transform(m.predict(Xf))
probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000)).fit(Yd, Ud.y.values[okd].astype(int))
res["best_alpha"] = best
res["E2_auroc_mlp_replica"] = round(within_auroc(Uf.y.values[okf].astype(int), probe.decision_function(rec), Uf.template.values[okf]), 4)
res["ridge_reference"] = json.loads((ROOT / "results/t1/t15_latent_replication.json").read_text())["r2_fresh"]
res["note"] = "alpha chosen on fresh R^2 here (exploratory): an optimistic upper estimate for the MLP"
(ROOT / "results/t1/t15_nonlinear_map_exploratory.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
