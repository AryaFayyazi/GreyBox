#!/usr/bin/env python
"""Exploratory sensitivity check for T17: the TXT map chose alpha = 1, the lowest value of the preregistered grid. Does a
smaller alpha make the text replica better? Extended grid {0.01, 0.03, 0.1, 0.3, 1}, chosen by the same 5-fold GroupKFold
on development; fresh R^2 for each alpha and for the CV-chosen one; READER unchanged (T17's value).
Output: results/t1/t17_txt_alpha_exploratory.json
python scripts/96b_t17_txt_alpha.py
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "8")
import importlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
GRID = (0.01, 0.03, 0.1, 0.3, 1)


def run(U, Hd, T, m, t9):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    ok = Hd["ok"].astype(bool)
    dev = m.is_dev(U.scenario_id.values)
    d, f = ok & dev, ok & ~dev
    Y = Hd["h"].reshape(len(U), -1).astype(np.float32)
    sc = StandardScaler().fit(Y[d])
    pca = PCA(128, random_state=0).fit(sc.transform(Y[d]))
    Yd, Yf = pca.transform(sc.transform(Y[d])), pca.transform(sc.transform(Y[f]))
    tx = t9.TXT()
    tx.v = [t9.vec(), t9.vec()]
    Xd, Xf = tx._X(T.ctx.values[d], T.act.values[d], True), tx._X(T.ctx.values[f], T.act.values[f], False)
    gd = U.scenario_id.values[d]
    cv, fresh = {}, {}
    for a in GRID:
        cv[a] = float(np.mean([r2_score(Yd[te], Ridge(alpha=a).fit(Xd[tr], Yd[tr]).predict(Xd[te]), multioutput="variance_weighted")
                               for tr, te in GroupKFold(5).split(Xd, Yd, gd)]))
        fresh[a] = float(r2_score(Yf, Ridge(alpha=a).fit(Xd, Yd).predict(Xf), multioutput="variance_weighted"))
    best = max(cv, key=cv.get)
    return {"cv_r2": {str(a): round(v, 4) for a, v in cv.items()}, "fresh_r2": {str(a): round(v, 4) for a, v in fresh.items()},
            "cv_chosen_alpha": best, "fresh_r2_cv_chosen": round(fresh[best], 4), "fresh_r2_best_any_alpha": round(max(fresh.values()), 4)}


def main():
    m = importlib.import_module("96_t17_replication_agents")
    t9 = importlib.import_module("77_t9_text_baseline")
    r17 = json.loads((ROOT / "results/t1/t17_replication_agents.json").read_text())
    out = {}
    for a in m.AGENTS:
        U = pd.read_parquet(m.units_path(a))
        out[a] = run(U, np.load(m.OUT / f"{a}_decision_hidden.npz"), m.texts(a), m, t9)
        out[a]["reader_r2_T17"] = r17["agents"][a]["r2_fresh"]["READER"]
        print(a, out[a], flush=True)
    (ROOT / "results/t1/t17_txt_alpha_exploratory.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
