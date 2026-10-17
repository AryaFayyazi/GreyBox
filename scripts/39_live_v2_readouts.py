#!/usr/bin/env python
"""Build the Grey Box Live v2 readout bundle from the Live benchmark (scripts/35-38).

Features: θ = 2 arcsin sqrt q on the 88 battery-v2m items ("about the user's LATEST message"), no log-odds, so the
same readout works for text-only APIs (Claude) where q comes from n sampled answers.
  frame       3D PCA of the pooled θ of all benchmark TRAIN conversations of all open models (model-agnostic)
  readouts    harmful_latest  the latest user message asks for something harmful (input property; text moderation
                              also gives this; shown for context)
              will_refuse     THIS model's own reply will be a refusal (model-specific forecast, made before it answers)
              Each is fit on the TRAIN items of all open models. Its validation is copied from the benchmark analysis
              (scripts/38, leave-one-family-out, text-tier simulation n=8), so the page shows held-out numbers only.
  thresholds  per known model: the 95th percentile of the model's own scores on its benign TRAIN conversations
              (target 5% per-turn false alarms); 'default' for an unseen black box: the median of those.
python scripts/39_live_v2_readouts.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.battery_v2m import BATTERY_V2M, battery_sha  # noqa: E402

LB = Path(os.environ.get("GB_LB", "/data/models/graybox_ws/outputs/livebench"))
M = int(os.environ.get("GB_M", "0"))    # >0: compact bundle of M items, black-box QR design on all open models' train θ
NAME = os.environ.get("GB_BUNDLE_NAME", f"readouts_v2{f'_m{M}' if M else ''}")   # readouts_v3 for the one-call battery
TAG = os.environ.get("GB_BATTERY_TAG", "v2m")                                       # v2m | v2m_batch
OUTP = Path(os.environ.get("GB_LIVE_DIR", "/data/models/graybox_ws/models/live")) / f"{NAME}.npz"
RES = Path(os.environ.get("GB_LB_RES", "/home/arya/graybox/results/livebench"))
VAL = Path(os.environ["GB_VAL"]) if os.environ.get("GB_VAL") else RES / f"livebench_theta_n8{f'_m{M}' if M else ''}.json"


def theta(z):
    q = (z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6)
    return 2 * np.arcsin(np.sqrt(q))


def main():
    items = pd.read_parquet(LB / "items.parquet")
    models = [d.name for d in sorted(LB.iterdir()) if (d / "psi.npz").exists() and (d / "refusals.parquet").exists()]
    val = json.loads(VAL.read_text())
    tr = (items.split == "train").values
    TH = {m: theta(np.load(LB / m / "psi.npz")) for m in models}
    names = [k for k, _ in BATTERY_V2M]
    if M:
        from scipy.linalg import qr
        T = np.vstack([TH[m][tr] for m in models])
        _, _, piv = qr(T - T.mean(0), pivoting=True, mode="economic")
        cols = np.sort(piv[:M])
        TH = {m: v[:, cols] for m, v in TH.items()}
        names = [names[j] for j in cols]
    REF = {m: pd.read_parquet(LB / m / "refusals.parquet").set_index("id").refused.reindex(items.id).values for m in models}
    X = np.vstack([TH[m][tr] for m in models])
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    dirs = Vt[:3].T
    P3 = (X - mu) @ dirs
    scale = 1.0 / np.percentile(np.linalg.norm(P3, axis=1), 97)
    benign_tr = tr & (items.harmful.values == 0)
    concepts = []
    for name, ys, kind in (("harmful_latest", {m: items.harmful.values for m in models}, "input"),
                           ("will_refuse", REF, "model-specific")):
        ok = {m: tr & (ys[m] >= 0) for m in models}
        Xc = np.vstack([TH[m][ok[m]] for m in models])
        yc = np.concatenate([ys[m][ok[m]] for m in models]).astype(int)
        m_ = Xc.mean(0)
        s_ = Xc.std(0) + 1e-6
        clf = LogisticRegression(C=0.3, max_iter=4000).fit((Xc - m_) / s_, yc)
        w = clf.coef_[0] / s_                                     # fold the standardisation into w, b
        b = float(clf.intercept_[0] - (clf.coef_[0] * m_ / s_).sum())
        thr = {}
        for m in models:
            p = 1 / (1 + np.exp(-(TH[m][benign_tr] @ w + b)))
            thr[m] = round(float(np.quantile(p, 0.95)), 4)
        thr["default"] = round(float(np.median(list(thr.values()))), 4)
        pm = val["per_model"]
        if name == "harmful_latest":
            v = {"metric": "held-out family, text tier n=8",
                 "toxicchat_auroc": {m: pm[m]["toxicchat"]["greybox_auroc"] for m in pm},
                 "toxicchat_fpr": {m: pm[m]["toxicchat"]["greybox_fpr"] for m in pm},
                 "carryover_fpr": {m: pm[m]["carryover_fpr"]["greybox"] for m in pm},
                 "moderation_toxicchat_auroc": {m: pm[m]["toxicchat"]["moderation_auroc"] for m in pm}}
        else:
            v = {"metric": "held-out family, text tier n=8",
                 "auroc": {m: pm[m]["refusal"]["auroc_self"] for m in pm},
                 "harmful_only_auroc": {m: pm[m]["refusal"]["harmful_refusal_auroc_self"] for m in pm},
                 "xstest_safe_auroc": {m: pm[m]["refusal"]["xstest_safe_auroc_self"] for m in pm},
                 "pooled": val.get("pooled", {})}
        concepts.append({"name": name, "w": w.astype(np.float32), "b": b, "threshold": thr, "kind": kind,
                         "validation": v})
    rng = np.random.default_rng(0)
    lab = np.concatenate([np.where(items.harmful.values[tr] == 1, "harmful", "benign") for _ in models])
    pick = rng.choice(len(X), min(3000, len(X)), replace=False)
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTP, items=np.array(names), mu=mu, dirs=dirs, scale=scale,
             concepts=np.array(concepts, dtype=object), cloud=(P3[pick] * scale).astype(np.float32),
             cloud_label=np.array(lab[pick]), battery=TAG, battery_sha=battery_sha(), models=np.array(models))
    rep = {"models": models, "battery_sha": battery_sha(), "frame_var_explained": round(float((S[:3] ** 2).sum() / (S ** 2).sum()), 4),
           "thresholds": {c["name"]: c["threshold"] for c in concepts}}
    rep["items"] = names
    (RES / f"{NAME}_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
