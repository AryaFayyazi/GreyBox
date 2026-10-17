#!/usr/bin/env python
"""Pilot (development, exploratory): does the interrogation state Ψ forecast the violation at turn start, and how much of
the true hidden state does it recover?

Forecasters at the same unit (turn start), all out-of-fold with GroupKFold over scenario_id (twins kept together):
  PSI_lp      logistic regression on θ(Ψ) from top-20 logprobs (logprob tier)
  PSI_s{n}    same, with Ψ estimated from n sampled answers per question (text-only tier)
  WB_L{l}     logistic regression on the true hidden state at turn start (white-box comparator), per layer
  TXT         TF-IDF + LR on the turn-start transcript (what a text monitor sees at this moment)
  CAPE_I_dec  CAPE's frozen gate score at the LATER decision point (after the turn's reasoning): reference only
Recovery: CV ridge R² from θ(Ψ) (and from TF-IDF, input-only) to the top hidden PCs and to the white-box score.

python scripts/03_pilot_analysis.py actor=gemma4 tag=pilot
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.stats import cluster_boot, fisher_rao, sample_tier, within_auroc, yes_prob  # noqa: E402

IN = Path("/data/models/graybox_ws/outputs/interrogate")
RES = Path("/home/arya/graybox/results/pilot")
D = OmegaConf.create({"actor": "gemma4", "tag": "pilot", "folds": 5, "n_boot": 500, "hidden": True})
TAGS = re.compile(r"<\|?[^<>]{0,40}\|?>")


def oof(X, y, groups, make, folds):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        m = make().fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return s


def oof_ridge_r2(X, Y, groups, folds, alpha=10.0):
    P = np.zeros_like(Y)
    for tr, te in GroupKFold(folds).split(X, Y, groups):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Y[tr])
        P[te] = m.predict(sc.transform(X[te])).reshape(len(te), -1)
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ((Y - P) ** 2).sum(0) / ss


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    stem = IN / f"{cfg.actor}_{cfg.tag}"
    U = pd.read_parquet(f"{stem}_units.parquet")
    Z = np.load(f"{stem}.npz", allow_pickle=True)
    q = yes_prob(Z["p_yes"], Z["p_no"])
    mass = Z["p_yes"] + Z["p_no"]
    y = (U.cls == "VIOL").astype(int).values
    g = U.scenario_id.values
    strata = U.template.values
    lr = lambda C=1.0: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))  # noqa: E731
    S = {}
    S["PSI_lp"] = oof(fisher_rao(q), y, g, lambda: lr(0.1), cfg.folds)
    for n in (1, 4, 16, 64):
        S[f"PSI_s{n}"] = oof(fisher_rao(sample_tier(q, n, seed=n)), y, g, lambda: lr(0.1), cfg.folds)
    texts = [TAGS.sub(" ", t) for t in pd.read_parquet(f"{stem}_turn_prompts.parquet").turn_prompt]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xt = tf.fit_transform(texts)  # vocabulary only (unsupervised); classifier is fit out-of-fold
    s = np.zeros(len(y))
    for tr, te in GroupKFold(cfg.folds).split(Xt, y, g):
        s[te] = LogisticRegression(C=1.0, max_iter=2000).fit(Xt[tr], y[tr]).decision_function(Xt[te])
    S["TXT"] = s
    S["CAPE_I_dec"] = U.I.values.astype(float)
    hid_path = Path(f"{stem}_hidden.npz")
    H = None
    if cfg.hidden and hid_path.exists():
        Hz = np.load(hid_path)
        H, layers = Hz["h"].astype(np.float32), Hz["layers"]
        for j, l in enumerate(layers):
            Xh = PCA(256, random_state=0).fit_transform(StandardScaler().fit_transform(H[:, j]))
            S[f"WB_L{l}"] = oof(Xh, y, g, lambda: lr(0.05), cfg.folds)
    contrasts = {"VIOL_vs_all": np.ones(len(y), bool), "VIOL_vs_XOTHER": U.cls.isin(["VIOL", "XOTHER"]).values,
                 "VIOL_vs_XPLUS": U.cls.isin(["VIOL", "XPLUS"]).values}
    res = {"n": U.cls.value_counts().to_dict(), "yes_no_mass_median": float(np.median(mass)),
           "yes_no_mass_lt_0.5": float((mass < 0.5).mean()), "auroc": {}}
    for cn, m in contrasts.items():
        res["auroc"][cn] = {}
        for k, v in S.items():
            pt = within_auroc(y[m], v[m], strata[m])
            bs = cluster_boot(lambda i: within_auroc(y[m][i], v[m][i], strata[m][i]), g[m], cfg.n_boot)
            res["auroc"][cn][k] = [round(pt, 4), round(float(np.nanpercentile(bs, 2.5)), 4),
                                   round(float(np.nanpercentile(bs, 97.5)), 4)]
    # per-item univariate within-AUROC (which questions carry the signal)
    res["item_auroc"] = {str(k): round(within_auroc(y, fisher_rao(q[:, j]), strata), 4)
                         for j, k in enumerate(Z["items"])}
    # recovery of the hidden state from θ(Ψ) vs from text (input-only)
    if H is not None:
        rec = {}
        Xpsi = fisher_rao(q)
        Xtxt = PCA(256, random_state=0).fit_transform(Xt.toarray() if Xt.shape[1] < 20000 else
                                                       TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000,
                                                                       sublinear_tf=True).fit_transform(texts).toarray())
        for j, l in enumerate(layers):
            Hs = StandardScaler().fit_transform(H[:, j])
            pca = PCA(32, random_state=0).fit(Hs)
            Y = pca.transform(Hs)
            w = pca.explained_variance_ratio_
            r_psi = oof_ridge_r2(Xpsi, Y, g, cfg.folds)
            r_txt = oof_ridge_r2(Xtxt, Y, g, cfg.folds)
            r_both = oof_ridge_r2(np.hstack([Xpsi, Xtxt]), Y, g, cfg.folds)
            wb = S[f"WB_L{l}"][:, None]
            rec[f"L{l}"] = {"pc_var_explained_top32": round(float(w.sum()), 4),
                            "R2_psi_weighted": round(float((r_psi * w).sum() / w.sum()), 4),
                            "R2_txt_weighted": round(float((r_txt * w).sum() / w.sum()), 4),
                            "R2_both_weighted": round(float((r_both * w).sum() / w.sum()), 4),
                            "R2_psi_pc1to5": [round(float(x), 3) for x in r_psi[:5]],
                            "R2_txt_pc1to5": [round(float(x), 3) for x in r_txt[:5]],
                            "R2_wbscore_from_psi": round(float(oof_ridge_r2(Xpsi, wb, g, cfg.folds)[0]), 4),
                            "R2_wbscore_from_txt": round(float(oof_ridge_r2(Xtxt, wb, g, cfg.folds)[0]), 4)}
        res["recovery"] = rec
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"{cfg.actor}_{cfg.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    pd.DataFrame(S).assign(cls=U.cls.values, template=strata, scenario_id=g).to_parquet(RES / f"{cfg.actor}_{cfg.tag}_scores.parquet")
    print(json.dumps({k: res[k] for k in ("n", "yes_no_mass_median", "yes_no_mass_lt_0.5", "auroc")}, indent=1))
    top = sorted(res["item_auroc"].items(), key=lambda kv: -abs(kv[1] - 0.5))[:12]
    print("top items:", top)
    if "recovery" in res:
        print(json.dumps(res["recovery"], indent=1))


if __name__ == "__main__":
    main()
