#!/usr/bin/env python
"""L2 analysis (development split only; exploratory): how much of each open LLM's hidden state does the black-box
interrogation state recover, beyond what the input text alone explains?

Feature sets (5-fold CV on dev contexts, ridge with an inner alpha grid, standardized):
  PSI_self    θ(Ψ) from the model itself (88 items, battery v2)
  PSI_other   θ(Ψ) of a DIFFERENT model on the same contexts (mean over all other available models): a reader control
  TXT_emb     MiniLM sentence embedding of the user message (input-only)
  TXT_emb+PSI_self, TXT_emb+PSI_other
Targets: top-64 PCs of the residual stream at each saved layer (variance-weighted R²), and the white-box concept
probe scores (harmful vs benign, sentiment) as R² of the out-of-fold probe logit.
The key quantity is the increment  ΔR² = R²(TXT+PSI_self) − R²(TXT+PSI_other): model-specific state recovered by
asking the model itself, beyond any reader of the same text.

python scripts/13_l2_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

L2 = Path("/data/models/graybox_ws/outputs/l2")
RES = Path("/home/arya/graybox/results/l2")
ALPHAS = np.logspace(-1, 4, 11)


def theta(z):
    """Logprob-tier features: θ = 2 arcsin sqrt(q) plus the clipped log-odds (keeps detail where answers saturate)."""
    q = (z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6)
    lo = np.clip(np.log(z["p_yes"] + 1e-12) - np.log(z["p_no"] + 1e-12), -25, 25)
    return np.hstack([2 * np.arcsin(np.sqrt(q)), lo])


def cv_r2(X, Y, folds=5):
    P = np.zeros_like(Y)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        sx = StandardScaler().fit(X[tr])
        # clip standardized features: near-constant columns (e.g. sparse logprob-tier SVD components of a 262k vocab)
        # otherwise explode on held-out rows (seen 2026-10-15: R² −272 for Gemma-3 LP features)
        m = RidgeCV(alphas=ALPHAS).fit(np.clip(sx.transform(X[tr]), -10, 10), Y[tr])
        P[te] = m.predict(np.clip(sx.transform(X[te]), -10, 10)).reshape(len(te), -1)
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def oof_logit(X, y, folds=5):
    s = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        sx = StandardScaler().fit(X[tr])
        s[te] = LogisticRegression(C=0.05, max_iter=3000).fit(sx.transform(X[tr]), y[tr]).decision_function(sx.transform(X[te]))
    return s


def logprob_features(z, k=128):
    """Logprob tier: top-20 next-token probabilities at the answer start, as a sparse vocab vector reduced by SVD."""
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    ids, lp = z["top_ids"], z["top_lp"]
    n = len(ids)
    p = np.exp(np.where(np.isfinite(lp), lp, -50.0))
    M = csr_matrix((np.sqrt(p).ravel(), (np.repeat(np.arange(n), ids.shape[1]), ids.ravel())), shape=(n, int(ids.max()) + 1))
    return TruncatedSVD(k, random_state=0).fit_transform(M)


def text_embeddings(texts):
    cache = L2 / "minilm.npy"
    if cache.exists():
        return np.load(cache)
    # all-MiniLM-L6-v2 with its standard pooling (attention-masked mean, then L2 normalisation), via transformers
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    p = snapshot_download("sentence-transformers/all-MiniLM-L6-v2", cache_dir="/data/models", local_files_only=True)
    tok, enc = AutoTokenizer.from_pretrained(p), AutoModel.from_pretrained(p).eval()
    out = []
    texts = list(texts)
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = enc(**b).last_hidden_state
            m = b["attention_mask"][..., None].float()
            e = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    E = np.concatenate(out)
    np.save(cache, E)
    return E


def main():
    C = pd.read_parquet(L2 / "contexts.parquet")
    dev = (C.split == "dev").values
    models = [d.name for d in sorted(L2.iterdir()) if (d / "psi.npz").exists() and (d / "hidden.npz").exists()]
    print("models:", models)
    PSI = {m: theta(np.load(L2 / m / "psi.npz")) for m in models}
    E = text_embeddings(C.text)
    res = {}
    for m in models:
        Z = np.load(L2 / m / "hidden.npz")
        H, layers = Z["h"].astype(np.float32)[dev], Z["layers"]
        others = [o for o in models if o != m]
        feats = {"PSI_self": PSI[m][dev], "TXT_emb": E[dev]}
        if others:
            feats["PSI_other"] = np.mean([PSI[o][dev] for o in others], 0)
            feats["TXT_emb+PSI_other"] = np.hstack([E[dev], feats["PSI_other"]])
        feats["TXT_emb+PSI_self"] = np.hstack([E[dev], PSI[m][dev]])
        LP = logprob_features(np.load(L2 / m / "psi.npz"))[dev]
        feats["LP_self"] = LP
        feats["TXT_emb+LP_self+PSI_self"] = np.hstack([E[dev], LP, PSI[m][dev]])
        res[m] = {"layers": layers.tolist(), "n_dev": int(dev.sum()), "hidden": {}, "concepts": {}}
        for j, l in enumerate(layers):
            Hl = H[:, j]
            if not np.isfinite(Hl).all():   # float16 overflow (Gemma-3 deep layers exceed 65,504); never silently drop
                res[m]["hidden"][f"L{l}"] = {"skipped": f"{int((~np.isfinite(Hl)).sum())} non-finite values (fp16 overflow)"}
                print(m, f"L{l}", "SKIPPED non-finite", flush=True)
                continue
            mu = Hl.mean(0)
            U, S, Vt = np.linalg.svd(Hl - mu, full_matrices=False)
            Y = (Hl - mu) @ Vt[:64].T
            w = S[:64] ** 2 / (S ** 2).sum()
            row = {"var_top64": round(float(w.sum()), 4)}
            for k, X in feats.items():
                r2 = cv_r2(X, Y)
                row[k] = round(float((r2 * w).sum() / w.sum()), 4)
            res[m]["hidden"][f"L{l}"] = row
            print(m, f"L{l}", row, flush=True)
        # concept probes: white-box out-of-fold logit on the mid-depth layer; how well is it recovered from each view?
        jm = len(layers) // 2
        Hm = H[:, jm]
        for cname, mask, lab in (("harmful", C.source[dev].isin(["harmful", "benign"]).values, C.harmful[dev]),
                                 ("sentiment", (C.source[dev] == "sst2").values, C.sentiment[dev])):
            y = lab[mask].astype(int).values
            wb = oof_logit(Hm[mask], y)[:, None]
            row = {"n": int(mask.sum())}
            for k, X in feats.items():
                row[f"R2_wbscore_from_{k}"] = round(float(cv_r2(X[mask], wb)[0]), 4)
            single = {o: round(float(cv_r2(PSI[o][dev][mask], wb)[0]), 4) for o in others}   # budget-matched controls
            if single:
                row["R2_wbscore_from_PSI_single_other"] = single
                row["single_other_best"], row["single_other_mean"] = max(single.values()), round(float(np.mean(list(single.values()))), 4)
            res[m]["concepts"][cname] = row
            print(m, cname, row, flush=True)
        # model-specific behaviour: does THIS model refuse? (scripts/12 stage=generate)
        rp = L2 / m / "replies.parquet"
        if rp.exists():
            from sklearn.metrics import roc_auc_score
            rep = pd.read_parquet(rp).set_index("id")
            ids = C.id.values[dev]
            mask = np.array([i in rep.index for i in ids])
            y = rep.refused.reindex(ids[mask]).astype(int).values
            row = {"n": int(mask.sum()), "refusal_rate": round(float(y.mean()), 4)}
            if 0 < y.sum() < len(y):
                for k, X in list(feats.items()) + [("whitebox_mid", Hm)]:
                    row[f"auroc_{k}"] = round(float(roc_auc_score(y, oof_logit(X[mask], y))), 4)
                # within harmful prompts only: who refuses which harmful request (the model-specific part)
                hm = (C.source.values[dev][mask] == "harmful")
                if 0 < y[hm].sum() < hm.sum():
                    for k, X in list(feats.items()) + [("whitebox_mid", Hm)]:
                        row[f"auroc_harmful_only_{k}"] = round(float(roc_auc_score(y[hm], oof_logit(X[mask][hm], y[hm]))), 4)
                    single = {o: round(float(roc_auc_score(y[hm], oof_logit(PSI[o][dev][mask][hm], y[hm]))), 4) for o in others}
                    if single:
                        row["auroc_harmful_only_PSI_single_other"] = single
                        row["single_other_best"] = max(single.values())
            res[m]["refusal"] = row
            print(m, "refusal", row, flush=True)
        # MLP neurons (privileged basis): which individual neurons can a black box see?
        npath = L2 / m / "neurons.npz"
        if npath.exists():
            Nz = np.load(npath)
            A, width = Nz["a"].astype(np.float32)[dev], int(Nz["width"])
            res[m]["neurons"] = {}
            for j, l in enumerate(Nz["layers"]):
                Al = A[:, j * width:(j + 1) * width]
                top = np.argsort(-Al.var(0))[:2000]                     # the 2,000 most variable neurons
                row = {}
                for k in ("PSI_self", "TXT_emb", "PSI_other", "TXT_emb+PSI_self", "TXT_emb+PSI_other"):
                    if k not in feats:
                        continue
                    r2 = cv_r2(feats[k], Al[:, top])
                    row[k] = {"median_R2": round(float(np.median(r2)), 4), "frac_R2_gt_0.5": round(float((r2 > 0.5).mean()), 4),
                              "frac_R2_gt_0.8": round(float((r2 > 0.8).mean()), 4)}
                    if k == "TXT_emb+PSI_self":
                        np.save(L2 / m / f"neuron_r2_L{l}.npy", np.stack([top, r2]))
                res[m]["neurons"][f"L{l}"] = row
                print(m, "neurons", f"L{l}", row, flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "l2_dev.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
