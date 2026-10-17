#!/usr/bin/env python
"""T15: does the grey box replicate the agent's hidden state? (docs/prereg_T15_latent_replication.md)

Target: Gemma-4's decision-point hidden state (layers 14-26 x 2816), standardized on development, top 128 PCs (dev).
Replicas: ridge maps from (READER) Qwen3-8B layer-32 features or (TXT) TF-IDF of context and action, alpha by
GroupKFold on development. E1: variance-weighted R^2 READER - TXT on fresh steps. E2: a probe trained on TRUE dev
states, applied to true vs READER-reconstructed fresh states (within-template AUROC; non-inferiority -0.02).
Exploratory: the map + probe on Claude (T8b) and GPT (T11) actions.
python scripts/88_t15_latent_replication.py
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
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

I, T4 = Path("/data/models/graybox_ws/outputs/interrogate"), Path("/data/models/graybox_ws/outputs/t4")
ALPHAS = (1, 10, 100, 1000, 10000)


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t9 = importlib.import_module("77_t9_text_baseline")
    Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
    Dd, Df = np.load(I / "gemma4_clean_decision_hidden.npz"), np.load(I / "gemma4_fresh_decision_hidden.npz")
    okd, okf = Dd["ok"].astype(bool), Df["ok"].astype(bool)
    Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz")
    j = [int(x) for x in Hd["layers"]].index(32)
    Td, Tf = t9.gemma_texts("clean"), t9.gemma_texts("fresh")
    # target: standardized 5-layer decision state -> 128 PCs (dev)
    nl, dim = Dd["h"].shape[1], Dd["h"].shape[2]
    Yd_full, Yf_full = Dd["h"][okd].reshape(okd.sum(), -1).astype(np.float32), Df["h"][okf].reshape(okf.sum(), -1).astype(np.float32)
    sc = StandardScaler().fit(Yd_full)
    pca = PCA(128, random_state=0).fit(sc.transform(Yd_full))
    Yd, Yf = pca.transform(sc.transform(Yd_full)), pca.transform(sc.transform(Yf_full))
    yd, yf = Ud.y.values[okd].astype(int), Uf.y.values[okf].astype(int)
    gd, gf, stf = Ud.scenario_id.values[okd], Uf.scenario_id.values[okf], Uf.template.values[okf]
    src = {"READER": (Hd["h"][okd, j].astype(np.float32), Hf["h"][okf, j].astype(np.float32))}
    tx = t9.TXT()
    tx.v = [t9.vec(), t9.vec()]
    src["TXT"] = (tx._X(Td.ctx.values[okd], Td.act.values[okd], True), tx._X(Tf.ctx.values[okf], Tf.act.values[okf], False))
    res = {"n_dev": int(okd.sum()), "n_fresh": int(okf.sum()), "pcs": 128, "variance_kept_by_128_pcs": round(float(pca.explained_variance_ratio_.sum()), 4),
           "alpha": {}, "r2_fresh": {}, "r2_fresh_per_layer_full_space": {}}
    rec, maps = {}, {}
    for name, (Xd, Xf) in src.items():
        dense = name == "READER"
        xs = StandardScaler().fit(Xd) if dense else None
        Xd_, Xf_ = (xs.transform(Xd), xs.transform(Xf)) if dense else (Xd, Xf)
        cv = {}
        for a in ALPHAS:
            r = []
            for tr, te in GroupKFold(5).split(Xd_, Yd, gd):
                r.append(r2_score(Yd[te], Ridge(alpha=a).fit(Xd_[tr], Yd[tr]).predict(Xd_[te]), multioutput="variance_weighted"))
            cv[a] = float(np.mean(r))
        a = max(cv, key=cv.get)
        m = Ridge(alpha=a).fit(Xd_, Yd)
        rec[name] = m.predict(Xf_)
        maps[name] = (xs, m)
        res["alpha"][name] = {"chosen": a, "cv_r2": {str(k): round(v, 4) for k, v in cv.items()}}
        res["r2_fresh"][name] = round(float(r2_score(Yf, rec[name], multioutput="variance_weighted")), 4)
        full_pred = pca.inverse_transform(rec[name])
        full_true = sc.transform(Yf_full)
        res["r2_fresh_per_layer_full_space"][name] = {int(L): round(float(r2_score(full_true[:, k * dim:(k + 1) * dim], full_pred[:, k * dim:(k + 1) * dim], multioutput="variance_weighted")), 4)
                                                     for k, L in enumerate(Dd["layers"])}
        print(name, "alpha", a, "R2", res["r2_fresh"][name], flush=True)
    res["E1_R2_READER_minus_TXT"] = [round(float(x), 4) for x in _ci(cluster_boot(
        lambda i: r2_score(Yf[i], rec["READER"][i], multioutput="variance_weighted") - r2_score(Yf[i], rec["TXT"][i], multioutput="variance_weighted"), gf, 1000, seed=151))]
    res["E1_pass"] = res["E1_R2_READER_minus_TXT"][1] > 0
    # E2: a probe trained on TRUE dev states, applied to true and reconstructed fresh states
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000)).fit(Yd, yd)
    S = {"true": probe.decision_function(Yf), "READER_replica": probe.decision_function(rec["READER"]), "TXT_replica": probe.decision_function(rec["TXT"])}
    res["E2_auroc"] = {k: round(within_auroc(yf, v, stf), 4) for k, v in S.items()}
    res["E2_replica_minus_true"] = [round(float(x), 4) for x in _ci(cluster_boot(
        lambda i: within_auroc(yf[i], S["READER_replica"][i], stf[i]) - within_auroc(yf[i], S["true"][i], stf[i]), gf, 1000, seed=152))]
    res["E2_pass"] = res["E2_replica_minus_true"][1] > -0.02
    # exploratory: closed agents seen through Gemma's latent space
    xs, m = maps["READER"]
    t8, t11 = importlib.import_module("76_t8b_claude_all_actions"), importlib.import_module("81_t11_gpt_monitor")
    for name, mod, feats in (("claude_T8b", t8, t8.OUT / "features.npz"), ("gpt_T11", t11, t11.OUT / "reader.npz")):
        U = mod.units()
        X = np.load(feats)["h"].astype(np.float32)
        s = probe.decision_function(m.predict(xs.transform(X)))
        y = U.y.values.astype(int)
        t = np.quantile(s, 0.95)
        res.setdefault("exploratory_closed_agents_via_gemma_latent", {})[name] = {
            "pooled_auroc": round(float(roc_auc_score(y, s)), 4), "caught_label_free_5pct": round(float((s[y == 1] >= t).mean()), 4),
            "safe_escalated": round(float((s[y == 0] >= t).mean()), 4)}
    (ROOT / "results/t1/t15_latent_replication.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


if __name__ == "__main__":
    main()
