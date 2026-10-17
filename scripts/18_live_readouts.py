#!/usr/bin/env python
"""Build the Grey Box Live readout bundle from L2 development data of the open models.

1. Live battery: black-box design (column-pivoted QR on the pooled θ answers of all models; no labels) -> m items.
2. Shared 3D frame: PCA of pooled θ on those items (model-agnostic: the same coordinates for every model).
3. Concept decoders (logistic regression on θ of the live items, pooled across models), each validated
   leave-one-family-out (train on the other families, test on the held-out family as an unseen black box):
     harmful_request   user asks for something harmful (source label)
     will_refuse       THIS model's actual reply is a refusal (model-specific behaviour, scripts/12 stage=generate)
     negative_tone     negative sentiment (SST-2 label)
   An axis is kept only if its worst held-out-family AUROC >= MIN_AUROC; dropped axes are listed with their numbers.
   Thresholds: the pooled out-of-family scores at 5% false-positive rate.

python scripts/18_live_readouts.py m=24
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.linalg import qr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib  # noqa: E402

run12 = importlib.import_module("12_l2_run")
from graybox.battery_v2 import BATTERY_V2  # noqa: E402

L2 = Path("/data/models/graybox_ws/outputs/l2")
OUTP = Path("/data/models/graybox_ws/models/live/readouts.npz")
RES = Path("/home/arya/graybox/results/live")
D = OmegaConf.create({"m": 24, "min_auroc": 0.75})


def theta(z):
    q = (z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6)
    return 2 * np.arcsin(np.sqrt(q))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    C = pd.read_parquet(L2 / "contexts.parquet")
    dev = (C.split == "dev").values
    models = [d.name for d in sorted(L2.iterdir()) if (d / "psi.npz").exists()]
    rows = []
    for m in models:
        th = theta(np.load(L2 / m / "psi.npz"))
        rep = pd.read_parquet(L2 / m / "replies.parquet").set_index("id") if (L2 / m / "replies.parquet").exists() else None
        for i in np.flatnonzero(dev):
            r = C.iloc[i]
            rows.append({"model": m, "family": run12.FAMILY[m], "source": r.source, "theta": th[i],
                         "harmful": r.harmful, "sentiment": r.sentiment,
                         "refused": (rep.refused.get(r.id) if rep is not None and r.id in rep.index else pd.NA)})
    df = pd.DataFrame(rows)
    T = np.stack(df.theta.values)
    _, _, piv = qr(T - T.mean(0), pivoting=True, mode="economic")
    items_idx = np.sort(piv[: cfg.m])
    item_ids = [BATTERY_V2[j][0] for j in items_idx]
    X = T[:, items_idx]
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    dirs = Vt[:3].T
    P3 = (X - mu) @ dirs
    scale = 1.0 / np.percentile(np.linalg.norm(P3, axis=1), 97)
    specs = {"harmful_request": (df.source.isin(["harmful", "benign"]).values, df.harmful),
             "will_refuse": (df.source.isin(["harmful", "benign"]).values & df.refused.notna().values, df.refused),
             "negative_tone": ((df.source == "sst2").values, 1 - df.sentiment.fillna(1))}
    concepts, report = [], {"models": models, "items": item_ids,
                            "frame_var_explained": round(float((S[:3] ** 2).sum() / (S ** 2).sum()), 4), "axes": {}}
    fams = sorted(df.family.unique())
    for name, (mask, lab) in specs.items():
        if mask.sum() < 50:
            continue
        y = lab[mask].astype(int).values
        Xm, fm = X[mask], df.family.values[mask]
        per_fam, oof = {}, np.full(len(y), np.nan)
        for f in fams:
            te = fm == f
            if te.sum() < 20 or len(np.unique(y[te])) < 2 or len(np.unique(y[~te])) < 2:
                continue
            clf = LogisticRegression(C=0.5, max_iter=3000).fit(Xm[~te], y[~te])
            oof[te] = clf.decision_function(Xm[te])
            per_fam[f] = round(float(roc_auc_score(y[te], oof[te])), 4)
        clf = LogisticRegression(C=0.5, max_iter=3000).fit(Xm, y)
        ok = np.isfinite(oof)
        thr_z = float(np.quantile(oof[ok & (y == 0)], 0.95)) if ok.any() else 0.0
        worst = min(per_fam.values()) if per_fam else None
        keep = worst is not None and len(per_fam) >= 2 and worst >= cfg.min_auroc
        report["axes"][name] = {"heldout_family_auroc": per_fam, "worst": worst, "kept": keep, "n": int(mask.sum())}
        if keep:
            concepts.append({"name": name, "w": clf.coef_[0].astype(np.float32), "b": float(clf.intercept_[0]),
                             "threshold": round(float(1 / (1 + np.exp(-thr_z))), 4), "kind": "label",
                             "validation": {"metric": "AUROC on held-out model family", "per_family": per_fam,
                                            "worst": worst}})
    pick = np.random.default_rng(0).choice(len(X), min(2500, len(X)), replace=False)
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTP, items=np.array(item_ids), mu=mu, dirs=dirs, scale=scale, concepts=np.array(concepts, dtype=object),
             cloud=(P3[pick] * scale).astype(np.float32), cloud_label=np.array(df.source.values[pick]))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "readouts_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
