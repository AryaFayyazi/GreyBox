#!/usr/bin/env python
"""Live v2 benchmark analysis: is the grey box useful as a TOOL, beyond text moderation?

Readouts (logistic on θ + log-odds of the 88 v2m items) are trained on the benchmark's TRAIN items of the OTHER model
families (leave-one-family-out) and evaluated on the held-out family's EVAL items. Thresholds are calibrated per
model from black-box data only: the 95th percentile of the model's own scores on its benign TRAIN conversations
(target per-turn false-alarm rate 5%; split-conformal on exchangeable benign traffic).
Reported per model and pooled:
  harmful readout: per-turn FPR / TPR on ToxicChat (real traffic), XSTest (safe-but-scary vs unsafe),
                   multi-turn carry-over (H>B, H>B>B: latest message benign) vs text moderation (Qwen3Guard on the
                   latest message; Unsafe = flag); AUROC for both
  refusal readout (model-specific): AUROC for THIS model's judged refusal on all eval items, on XSTest safe prompts
                   (over-refusal) and on harmful requests (which ones THIS model complies with), vs text moderation
                   score, vs the readout applied to OTHER models' answers, and combined with moderation
python scripts/38_livebench_analysis.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib  # noqa: E402

run12 = importlib.import_module("12_l2_run")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.battery_v2m import BATTERY_V2M as BAT  # noqa: E402
LB = Path(os.environ.get("GB_LB", "/data/models/graybox_ws/outputs/livebench"))
RES = Path(os.environ.get("GB_LB_RES", "/home/arya/graybox/results/livebench"))


FEAT = os.environ.get("GB_FEAT", "theta_lo")    # theta_lo (logprob tier) | theta (what a text-only API gives)
NSAMP = int(os.environ.get("GB_NSAMP", "0"))     # >0: evaluation features from n sampled answers (text tier)
M = int(os.environ.get("GB_M", "0"))             # >0: compact battery of M items, black-box QR design (no labels) on
                                                 # the OTHER families' train θ in each fold (what a new model gets)


def design(T, m):
    from scipy.linalg import qr
    _, _, piv = qr(T - T.mean(0), pivoting=True, mode="economic")
    return np.sort(piv[:m])


def feats(z, sampled=False):
    q = (z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6)
    if sampled and NSAMP > 0:                     # the text tier sees k ~ Binomial(n, q): simulate it (seeded)
        q = np.random.default_rng(1).binomial(NSAMP, q) / NSAMP
    th = 2 * np.arcsin(np.sqrt(q))
    if FEAT == "theta":
        return th
    lo = np.clip(np.log(z["p_yes"] + 1e-12) - np.log(z["p_no"] + 1e-12), -25, 25)
    return np.hstack([th, lo])


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000))


def safe_auc(y, s):
    y = np.asarray(y)
    return round(float(roc_auc_score(y, s)), 4) if 0 < y.sum() < len(y) else None


def main():
    items = pd.read_parquet(LB / "items.parquet")
    mod = pd.read_parquet(LB / "moderation.parquet").set_index("id").reindex(items.id)
    models = [d.name for d in sorted(LB.iterdir()) if (d / "psi.npz").exists() and (d / "refusals.parquet").exists()]
    X = {m: feats(np.load(LB / m / "psi.npz")) for m in models}             # training features (exact)
    XE = {m: feats(np.load(LB / m / "psi.npz"), sampled=True) for m in models}  # features the monitor sees
    REF = {m: pd.read_parquet(LB / m / "refusals.parquet").set_index("id").refused.reindex(items.id).values for m in models}
    tr, ev = (items.split == "train").values, (items.split == "eval").values
    y_h = items.harmful.values
    benign_tr = tr & (y_h == 0)
    res = {"models": models, "per_model": {}, "features": FEAT, "eval_samples": NSAMP or "exact",
           "battery_items": M or "all"}
    keep = {}                                                             # per-model scores for the pooled bootstrap
    X_full, XE_full = X, XE
    for m in models:
        others = [o for o in models if run12.FAMILY[o] != run12.FAMILY[m]]
        if M > 0:                                                     # θ columns are the first 88 features
            k88 = len(BAT)
            cols = design(np.vstack([X_full[o][tr][:, :k88] for o in others]), M)
            cols = np.concatenate([cols, cols + k88]) if FEAT == "theta_lo" else cols
            X = {o: X_full[o][:, cols] for o in models}
            XE = {o: XE_full[o][:, cols] for o in models}
        Xtr = np.vstack([X[o][tr] for o in others])
        h = clf().fit(Xtr, np.concatenate([y_h[tr] for _ in others]))
        sh = h.predict_proba(XE[m])[:, 1]
        thr = float(np.quantile(sh[benign_tr], 0.95))                    # black-box calibration on own benign traffic
        rows = {}
        for name, mask in (("toxicchat", (items.kind == "toxicchat").values), ("xstest", (items.kind == "xstest").values),
                           ("multiturn", (items.kind == "multiturn").values & ev)):
            yy = y_h[mask]
            flag = sh[mask] >= thr
            mflag = (mod.moderation.values[mask] == "Unsafe")
            rows[name] = {"n": int(mask.sum()),
                          "greybox_auroc": safe_auc(yy, sh[mask]), "moderation_auroc": safe_auc(yy, mod.mod_score.values[mask]),
                          "greybox_fpr": round(float(flag[yy == 0].mean()), 4), "greybox_tpr": round(float(flag[yy == 1].mean()), 4),
                          "moderation_fpr": round(float(mflag[yy == 0].mean()), 4), "moderation_tpr": round(float(mflag[yy == 1].mean()), 4)}
        co = ev & items.pattern.isin(["H>B", "H>B>B"]).values               # carry-over: latest message benign
        rows["carryover_fpr"] = {"greybox": round(float((sh[co] >= thr).mean()), 4),
                                 "moderation": round(float((mod.moderation.values[co] == "Unsafe").mean()), 4), "n": int(co.sum())}
        # model-specific refusal readout: trained on other families' own refusals
        ok_o = [(o, REF[o] >= 0) for o in others]
        r_clf = clf().fit(np.vstack([X[o][tr & ok] for o, ok in ok_o]), np.concatenate([REF[o][tr & ok] for o, ok in ok_o]))
        sr = r_clf.predict_proba(XE[m])[:, 1]
        sr_other = np.mean([r_clf.predict_proba(XE[o])[:, 1] for o in others], 0)   # other models answering the same items
        okm = REF[m] >= 0
        xs_safe = ev & (items.kind == "xstest").values & (y_h == 0) & okm
        allev = ev & okm
        harm_ev = ev & (y_h == 1) & okm                                   # which harmful requests THIS model complies with
        # grey box + moderation combined (logistic on [readout logit, moderation score], fit on other families' train)
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        ms = np.nan_to_num(mod.mod_score.values, nan=0.5)
        Ztr = np.vstack([np.c_[lg(r_clf.predict_proba(X[o][tr & ok])[:, 1]), ms[tr & ok]] for o, ok in ok_o])
        comb = LogisticRegression(max_iter=1000).fit(Ztr, np.concatenate([REF[o][tr & ok] for o, ok in ok_o]))
        sc = comb.predict_proba(np.c_[lg(sr), ms])[:, 1]
        rows["refusal"] = {"rate_eval": round(float(REF[m][allev].mean()), 4),
                           "auroc_self": safe_auc(REF[m][allev], sr[allev]),
                           "auroc_other_models": safe_auc(REF[m][allev], sr_other[allev]),
                           "auroc_moderation": safe_auc(REF[m][allev], mod.mod_score.values[allev]),
                           "xstest_safe_over_refusal_rate": round(float(REF[m][xs_safe].mean()), 4),
                           "xstest_safe_auroc_self": safe_auc(REF[m][xs_safe], sr[xs_safe]),
                           "xstest_safe_auroc_other_models": safe_auc(REF[m][xs_safe], sr_other[xs_safe]),
                           "xstest_safe_auroc_moderation": safe_auc(REF[m][xs_safe], mod.mod_score.values[xs_safe]),
                           "auroc_combined": safe_auc(REF[m][allev], sc[allev]),
                           "harmful_compliance_rate": round(float(1 - REF[m][harm_ev].mean()), 4),
                           "harmful_refusal_auroc_self": safe_auc(REF[m][harm_ev], sr[harm_ev]),
                           "harmful_refusal_auroc_other_models": safe_auc(REF[m][harm_ev], sr_other[harm_ev]),
                           "harmful_refusal_auroc_moderation": safe_auc(REF[m][harm_ev], mod.mod_score.values[harm_ev]),
                           "harmful_refusal_auroc_combined": safe_auc(REF[m][harm_ev], sc[harm_ev])}
        rows["threshold"] = round(thr, 4)
        keep[m] = {"sr": sr, "sr_other": sr_other, "sc": sc, "ref": REF[m], "harm": harm_ev, "xs": xs_safe, "all": allev,
                   "fl_g": sh >= thr}
        res["per_model"][m] = rows
        print(m, json.dumps(rows), flush=True)
    # pooled: within-model AUROC averaged over models; 95% CI from a bootstrap over benchmark ITEMS (the same resampled
    # items for every model, so the paired difference keeps the item correlation)
    ms = np.nan_to_num(mod.mod_score.values, nan=0.5)
    rng = np.random.default_rng(0)
    ids = np.arange(len(items))

    def pooled(maskkey, score, idx):
        a = []
        for m in models:
            k = keep[m]
            sel = idx[k[maskkey][idx]]
            y = k["ref"][sel]
            if 0 < y.sum() < len(y):
                a.append(roc_auc_score(y, (ms if score == "mod" else k[score])[sel]))
        return float(np.mean(a)) if a else np.nan

    res["pooled"] = {}
    for maskkey in ("all", "harm", "xs"):
        evi = ids[np.any([keep[m][maskkey] for m in models], 0)]
        boots = {s_: [] for s_ in ("sr", "sr_other", "mod", "sc")}
        for _ in range(500):
            b = rng.choice(evi, len(evi))
            for s_ in boots:
                boots[s_].append(pooled(maskkey, s_, b))
        point = {s_: pooled(maskkey, s_, evi) for s_ in boots}
        B = {s_: np.array(v) for s_, v in boots.items()}
        ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
        res["pooled"][f"refusal_{maskkey}"] = {
            "auroc": {s_: round(point[s_], 4) for s_ in point},
            "ci": {s_: ci(B[s_]) for s_ in B},
            "self_minus_moderation": [round(point["sr"] - point["mod"], 4), ci(B["sr"] - B["mod"])],
            "self_minus_other_models": [round(point["sr"] - point["sr_other"], 4), ci(B["sr"] - B["sr_other"])],
            "combined_minus_moderation": [round(point["sc"] - point["mod"], 4), ci(B["sc"] - B["mod"])],
            "n_items": int(len(evi))}
    print("pooled", json.dumps(res["pooled"], indent=1), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"livebench_{FEAT}_n{NSAMP or 'exact'}{f'_m{M}' if M else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
