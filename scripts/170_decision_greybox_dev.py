#!/usr/bin/env python
"""Exploratory (development data): the grey box at the agent's decision point. Two estimators of the agent's hidden
DECISION state, compared on the same 600 Gemma-4 decision calls (300 violating, 300 allowed; the observable-reference
sample, CAPE-Bench development scenarios):
  A  the agent's own answers: 8 yes/no questions about the proposed call (web/build_observable_ref.py), 4 answers each,
     theta = arcsine angle of the Anscombe estimate (8 numbers per call; no other model);
  B  an open reader (Qwen3-8B) of the same transcript and call (T4's features);
against the agent's true decision state (Gemma-4, stored). For each estimator:
  1 geometry: Spearman between its pairwise distances and the true state's (cosine), all pairs and partial given
    "same template";
  2 recreation: share of the variance of the true state's top 32 principal components recovered by a linear map
    (ridge; A cross-fit by scenario within the 600; B fit on the other 4,893 development units);
  3 grey-box monitoring: a probe trained on TRUE decision states (the other 4,893 units) applied to the recreated state;
    within-template and pooled AUROC (white-box reference: the same probe on the true states).
  python scripts/170_decision_greybox_dev.py [layer=3]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.stats import within_auroc  # noqa: E402

I = Path("/data/models/graybox_ws/outputs/interrogate")
LAYER = int(dict(a.split("=") for a in sys.argv[1:]).get("layer", 3))


def theta(k, m):
    p = (np.asarray(k, float) + 3 / 8) / (np.asarray(m, float) + 3 / 4)
    return 2 * np.arcsin(np.sqrt(p))


def pdist(X, metric):
    if metric == "cos":
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        D = 1 - Xn @ Xn.T
    else:
        sq = (X ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0))
    iu = np.triu_indices(len(X), 1)
    return D[iu]


def partial_spearman(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    res = lambda x: x - np.polyval(np.polyfit(rc, x, 1), rc)  # noqa: E731
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def main():
    U = pd.read_parquet(I / "gemma4_clean_units.parquet")
    z = np.load(I / "gemma4_clean_decision_hidden.npz")
    Hall, ok = z["h"][:, LAYER].astype(np.float64), z["ok"].astype(bool)
    Rall = np.load("/data/models/graybox_ws/outputs/t4/clean_act.npz")["h"][:, LAYER].astype(np.float64)
    O = pd.read_parquet("/data/models/graybox_ws/outputs/observable_ref/gemma4.parquet")
    key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
    idx = np.array([key[k] for k in zip(O.episode_id, O.event_id.astype(int))])
    assert ok[idx].all() and (U.y.values[idx] == O.y.values).all()
    C = np.array([json.loads(c) for c in O.counts])                 # (600, 8, 2): yes count, readable count
    TH = theta(C[:, :, 0], C[:, :, 1])
    train = np.setdiff1d(np.where(ok)[0], idx)
    y, tm, sc = O.y.values.astype(int), O.template.values, O.scenario_id.values
    pca = PCA(32, random_state=0).fit(Hall[train])
    Ztrue, Ztr = pca.transform(Hall[idx]), pca.transform(Hall[train])
    # geometry
    dH = pdist(Hall[idx], "cos")
    same_tm = pdist(pd.factorize(tm)[0][:, None].astype(float), "euc") == 0
    geo = {}
    for name, X, met in (("answers", TH, "euc"), ("reader", Rall[idx], "cos")):
        d = pdist(X, met)
        geo[name] = {"spearman": round(float(spearmanr(d, dH)[0]), 3), "partial_given_template": round(partial_spearman(d, dH, (~same_tm).astype(float)), 3)}
    geo["template_indicator"] = {"spearman": round(float(spearmanr((~same_tm).astype(float), dH)[0]), 3)}
    # recreation
    rec = {}
    folds = pd.Series(sc).map({s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(np.unique(sc)))}).values
    Zhat_A = np.zeros_like(Ztrue)
    for f in range(5):
        tr, te = folds != f, folds == f
        Zhat_A[te] = Ridge(alpha=1.0).fit(TH[tr], Ztrue[tr]).predict(TH[te])
    rpca = PCA(256, random_state=0).fit(Rall[train])
    Zhat_B = Ridge(alpha=10.0).fit(rpca.transform(Rall[train]), Ztr).predict(rpca.transform(Rall[idx]))
    r2 = lambda Zh: float(1 - ((Ztrue - Zh) ** 2).sum() / ((Ztrue - Ztrue.mean(0)) ** 2).sum())  # noqa: E731
    rec["answers"], rec["reader"] = round(r2(Zhat_A), 3), round(r2(Zhat_B), 3)
    # grey-box monitoring: a probe trained on TRUE states, applied to recreated states
    probe = LogisticRegression(C=0.1, max_iter=5000).fit(Ztr, U.y.values[train])
    mon = {}
    for name, Z in (("true_state (white box)", Ztrue), ("recreated from answers", Zhat_A), ("recreated by the reader", Zhat_B)):
        s = probe.decision_function(Z)
        mon[name] = {"within_template": round(float(within_auroc(y, s, tm)), 3), "pooled": round(float(roc_auc_score(y, s)), 3)}
    mon["audit question alone (answers q0, 'complies?')"] = {"within_template": round(float(within_auroc(y, -TH[:, 0], tm)), 3),
                                                            "pooled": round(float(roc_auc_score(y, -TH[:, 0])), 3)}
    res = {"units": int(len(idx)), "violations": int(y.sum()), "layer": int(z["layers"][LAYER]), "geometry": geo,
           "recreation_r2_top32": rec, "monitor_true_state_probe": mon}
    (ROOT / "results/t1/decision_greybox_DEV.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
