#!/usr/bin/env python
"""T1 test on clean units (past-only selection; scripts/15): does any monitor beat the rollout forecaster?

Monitors at the turn start of each unit (out-of-fold, GroupKFold by scenario, twins together):
  R_K        fraction of K resampled turns that are unsafe (no training; the behavioural forecast π̂_K)
  WB_L*      logistic probe on the true hidden state (per layer), calibrated out of fold
  PSI        logistic on θ(Ψ) (logprob tier) ; PSI_s16 text-only tier
  TXT        TF-IDF + logistic on the turn-start transcript
Metrics: within-(template x tool) AUROC with scenario bootstrap; Brier score and log loss (proper scores, T1.1) on the
natural mix; paired bootstrap of AUROC(R) - AUROC(m). Trained monitors give out-of-fold logistic probabilities
(fit under log loss); R uses (k + 0.5)/(K + 1) to avoid log(0).

python scripts/17_clean_t1.py actor=gemma4 tag=clean
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.stats import cluster_boot, fisher_rao, sample_tier, within_auroc, yes_prob  # noqa: E402

pil = importlib.import_module("03_pilot_analysis")
IN = Path("/data/models/graybox_ws/outputs/interrogate")
RES = Path("/home/arya/graybox/results/t1")
D = OmegaConf.create({"actor": "gemma4", "tag": "clean", "K": 16, "folds": 5, "n_boot": 1000})


def oof_prob(X, y, groups, make, folds):
    """Out-of-fold probabilities (the classifier's own predict_proba; fit inside each fold)."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        p[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    stem = IN / f"{cfg.actor}_{cfg.tag}"
    U = pd.read_parquet(f"{stem}_units.parquet")
    Z = np.load(f"{stem}.npz", allow_pickle=True)
    R = pd.read_parquet(f"{stem}_rollouts_K{cfg.K}.parquet")
    assert (R.episode_id.values == U.episode_id.values).all() and (R.event_id.values == U.event_id.values).all()
    y = U.y.values.astype(int)
    g, st = U.scenario_id.values, U.template.values
    lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))  # noqa: E731
    k = np.array([np.sum(eval(s)) for s in R.R_unsafe])
    n = np.array([len(eval(s)) for s in R.R_unsafe])
    P = {f"R_K{cfg.K}": (k + 0.5) / (n + 1)}
    q = yes_prob(Z["p_yes"], Z["p_no"])
    lo = np.clip(np.log(Z["p_yes"] + 1e-12) - np.log(Z["p_no"] + 1e-12), -25, 25)   # logprob tier keeps saturated detail
    P["PSI"] = oof_prob(np.hstack([fisher_rao(q), lo]), y, g, lambda: lr(0.1), cfg.folds)
    P["PSI_s16"] = oof_prob(fisher_rao(sample_tier(q, 16, seed=16)), y, g, lambda: lr(0.1), cfg.folds)
    texts = [pil.TAGS.sub(" ", t) for t in pd.read_parquet(f"{stem}_turn_prompts.parquet").turn_prompt]
    Xt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    P["TXT"] = oof_prob(Xt, y, g, lambda: LogisticRegression(C=1.0, max_iter=3000), cfg.folds)
    Hz = np.load(f"{stem}_hidden.npz")
    for j, l in enumerate(Hz["layers"]):
        Xh = PCA(256, random_state=0).fit_transform(StandardScaler().fit_transform(Hz["h"][:, j].astype(np.float32)))
        P[f"WB_L{l}"] = oof_prob(Xh, y, g, lambda: lr(0.05), cfg.folds)
    res = {"n": U.cls.value_counts().to_dict(), "base_rate": round(float(y.mean()), 4), "monitors": {}}
    contrasts = {"all": np.ones(len(y), bool), "vs_XSAFE": U.cls.isin(["XUNSAFE", "XSAFE"]).values,
                 "vs_XPLUS": U.cls.isin(["XUNSAFE", "XPLUS"]).values}
    ref = P[f"R_K{cfg.K}"]
    for name, p in P.items():
        row = {}
        for cn, m in contrasts.items():
            pt = within_auroc(y[m], p[m], st[m])
            bs = cluster_boot(lambda i: within_auroc(y[m][i], p[m][i], st[m][i]), g[m], cfg.n_boot)
            row[f"auroc_{cn}"] = [round(pt, 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]
            if name != f"R_K{cfg.K}":
                d = cluster_boot(lambda i: within_auroc(y[m][i], ref[m][i], st[m][i]) - within_auroc(y[m][i], p[m][i], st[m][i]),
                                 g[m], cfg.n_boot, seed=1)
                row[f"R_minus_this_{cn}"] = [round(float(np.nanmean(d)), 4), round(float(np.nanpercentile(d, 2.5)), 4),
                                             round(float(np.nanpercentile(d, 97.5)), 4)]
        pc = np.clip(p, 1e-4, 1 - 1e-4)
        row["brier"] = round(float(np.mean((pc - y) ** 2)), 5)
        row["log_loss"] = round(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), 5)
        res["monitors"][name] = row
        print(name, row, flush=True)
    a_r = res["monitors"][f"R_K{cfg.K}"]["auroc_all"][0]
    res["relative_efficiency_vs_R"] = {k: round((v["auroc_all"][0] - 0.5) / (a_r - 0.5), 4) for k, v in res["monitors"].items()}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{cfg.actor}_{cfg.tag}.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame(P).assign(y=y, cls=U.cls.values, template=st, scenario_id=g).to_parquet(RES / f"{cfg.actor}_{cfg.tag}_scores.parquet")


if __name__ == "__main__":
    main()
