#!/usr/bin/env python
"""Is the refusal forecast specific to the model being read? (follow-up to scripts/38; budget-matched control)

scripts/38 found that the refusal readout applied to the AVERAGE of five other models' answers predicts model m's own
refusal slightly better than m's own answers (-0.028). That comparison spends 5x the queries. Here, for each model m
and each subset (all eval / harmful requests / XSTest safe prompts), with the same leave-one-family-out refusal
readout as 38:
  self        m's own Ψ
  single      ONE other model's Ψ (each other model in turn; mean AUROC) -- same query budget as self
  ensemble    mean score over the other models' Ψ (5x budget)
  self+ens    mean of self and ensemble scores
and the label-level ceiling: how well does ANOTHER model's actual refusal predict m's refusal (label agreement)?
Pooled: mean over models, paired item bootstrap (500).
python scripts/44_refusal_specificity.py      # env GB_FEAT / GB_NSAMP as in 38
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
a38 = importlib.import_module("38_livebench_analysis")
run12 = a38.run12
LB, RES = a38.LB, a38.RES


def main():
    items = pd.read_parquet(LB / "items.parquet")
    models = [d.name for d in sorted(LB.iterdir()) if (d / "psi.npz").exists() and (d / "refusals.parquet").exists()]
    X = {m: a38.feats(np.load(LB / m / "psi.npz")) for m in models}
    XE = {m: a38.feats(np.load(LB / m / "psi.npz"), sampled=True) for m in models}
    REF = {m: pd.read_parquet(LB / m / "refusals.parquet").set_index("id").refused.reindex(items.id).values for m in models}
    tr, ev = (items.split == "train").values, (items.split == "eval").values
    h = items.harmful.values
    masks = {"all": ev, "harm": ev & (h == 1), "xs": ev & (items.kind == "xstest").values & (h == 0)}
    S = {}
    for m in models:
        others = [o for o in models if run12.FAMILY[o] != run12.FAMILY[m]]
        ok_o = [(o, REF[o] >= 0) for o in others]
        r = a38.clf().fit(np.vstack([X[o][tr & ok] for o, ok in ok_o]), np.concatenate([REF[o][tr & ok] for o, ok in ok_o]))
        S[m] = {"self": r.predict_proba(XE[m])[:, 1], "others": {o: r.predict_proba(XE[o])[:, 1] for o in others}}
    rng = np.random.default_rng(0)
    ids = np.arange(len(items))

    def stats(mk, idx):
        out = {k: [] for k in ("self", "single", "ensemble", "self_plus_ensemble", "label_other")}
        for m in models:
            sel = idx[masks[mk][idx] & (REF[m][idx] >= 0)]
            y = REF[m][sel]
            if not 0 < y.sum() < len(y):
                continue
            s = S[m]
            ens = np.mean([v for v in s["others"].values()], 0)
            out["self"].append(roc_auc_score(y, s["self"][sel]))
            out["single"].append(np.mean([roc_auc_score(y, v[sel]) for v in s["others"].values()]))
            out["ensemble"].append(roc_auc_score(y, ens[sel]))
            out["self_plus_ensemble"].append(roc_auc_score(y, (s["self"] + ens)[sel] / 2))
            lab = []
            for o in s["others"]:
                ok = REF[o][sel] >= 0
                if ok.all() or ok.sum() > 10:
                    lab.append(roc_auc_score(y[ok], REF[o][sel][ok]))
            out["label_other"].append(np.mean(lab))
        return {k: float(np.mean(v)) for k, v in out.items()}

    res = {"features": a38.FEAT, "eval_samples": a38.NSAMP or "exact", "models": models, "pooled": {}}
    for mk in masks:
        evi = ids[masks[mk]]
        point = stats(mk, evi)
        boots = [stats(mk, rng.choice(evi, len(evi))) for _ in range(500)]
        ci = lambda f: [round(float(np.percentile([f(b) for b in boots], 2.5)), 4), round(float(np.percentile([f(b) for b in boots], 97.5)), 4)]
        res["pooled"][mk] = {"auroc": {k: round(v, 4) for k, v in point.items()},
                             "self_minus_single": [round(point["self"] - point["single"], 4), ci(lambda b: b["self"] - b["single"])],
                             "self_minus_ensemble": [round(point["self"] - point["ensemble"], 4), ci(lambda b: b["self"] - b["ensemble"])],
                             "selfens_minus_ensemble": [round(point["self_plus_ensemble"] - point["ensemble"], 4),
                                                        ci(lambda b: b["self_plus_ensemble"] - b["ensemble"])],
                             "n_items": int(len(evi))}
        print(mk, json.dumps(res["pooled"][mk]), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"refusal_specificity_{a38.FEAT}_n{a38.NSAMP or 'exact'}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
