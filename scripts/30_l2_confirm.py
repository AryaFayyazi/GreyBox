#!/usr/bin/env python
"""L2 preregistered confirmation on the held-out 30% of contexts (docs/prereg_L2.md once locked).

stage=freeze   for every model: fit ridge decoders on ALL dev contexts (alpha chosen by inner CV on dev):
                 Ψ_self -> top-64 hidden PCs at the mid layer (PCA frame from dev), text -> same, text+Ψ_self,
                 text+Ψ_other; and Ψ_self / text -> the white-box concept-probe logits (probe fit on dev)
               save decoders + frames to models/l2/frozen_<model>.npz
stage=confirm  apply only frozen objects to the confirmation contexts; per-model R² and pooled tests with a context
               bootstrap (2,000 resamples):
  H1  mean over models of [R²(Ψ_self) − R²(text)] on hidden PCs > 0
  H2  mean over models of [R²(Ψ_self → own sentiment-probe logit) − mean over single other models o of
      R²(Ψ_o → the same logit)] > 0      (self-specificity, budget-matched; redesigned 2026-10-15 22:00 after dev
      showed raw-variance self-specificity is absent: see docs/prereg_L2.md)
  H3  mean over models of [R²(Ψ_self) − R²(text)] on the sentiment-probe logit > 0
  Holm over H1-H3 at alpha 0.05.

python scripts/30_l2_confirm.py stage=freeze
python scripts/30_l2_confirm.py stage=confirm
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
an = importlib.import_module("13_l2_analysis")
L2, RES = an.L2, an.RES
MOD = Path("/data/models/graybox_ws/models/l2")
D = OmegaConf.create({"stage": "confirm", "n_boot": 2000})
ALPHAS = np.logspace(-1, 4, 11)


def views(models, PSI, E, idx, m):
    others = [o for o in models if o != m]
    v = {"psi": PSI[m][idx], "txt": E[idx], "txt_psi": np.hstack([E[idx], PSI[m][idx]])}
    if others:
        v["txt_other"] = np.hstack([E[idx], np.mean([PSI[o][idx] for o in others], 0)])
    return v


def fit_ridge(X, Y):
    sx = StandardScaler().fit(X)
    r = RidgeCV(alphas=ALPHAS).fit(sx.transform(X), Y)
    return {"mu": sx.mean_, "sd": sx.scale_, "coef": r.coef_, "b": r.intercept_}


def predict(d, X):
    # Deviation 1 (2026-10-15 22:35): clip standardized features to ±10, the rule the dev analysis (scripts/13) uses.
    # Without it, near-constant dev columns (saturated log-odds) explode on held-out rows (R² ~ -1e12).
    return np.clip((X - d["mu"]) / d["sd"], -10, 10) @ np.atleast_2d(d["coef"]).T + d["b"]


def setup():
    C = pd.read_parquet(L2 / "contexts.parquet")
    models = [d.name for d in sorted(L2.iterdir()) if (d / "psi.npz").exists() and (d / "hidden.npz").exists()]
    PSI = {m: an.theta(np.load(L2 / m / "psi.npz")) for m in models}
    return C, models, PSI, an.text_embeddings(C.text)


def freeze():
    C, models, PSI, E = setup()
    dev = np.flatnonzero((C.split == "dev").values)
    MOD.mkdir(parents=True, exist_ok=True)
    sst = (C.source.values == "sst2")
    for m in models:
        Z = np.load(L2 / m / "hidden.npz")
        j = len(Z["layers"]) // 2
        H = Z["h"][:, j].astype(np.float32)
        mu = H[dev].mean(0)
        _, S, Vt = np.linalg.svd(H[dev] - mu, full_matrices=False)
        comps, w = Vt[:64], S[:64] ** 2 / (S ** 2).sum()
        Y = (H[dev] - mu) @ comps.T
        out = {"mu": mu, "comps": comps, "w": w, "layer": int(Z["layers"][j]), "models_at_freeze": np.array(models)}
        for k, X in views(models, PSI, E, dev, m).items():
            for kk, vv in fit_ridge(X, Y).items():
                out[f"h_{k}_{kk}"] = vv
        # sentiment white-box probe (fit on dev SST-2 contexts) and its recovery from black-box views
        sd = dev[sst[dev]]
        sh = StandardScaler().fit(H[sd])
        probe = LogisticRegression(C=0.05, max_iter=4000).fit(sh.transform(H[sd]), C.sentiment.values[sd].astype(int))
        out.update({"probe_mu": sh.mean_, "probe_sd": sh.scale_, "probe_w": probe.coef_[0], "probe_b": probe.intercept_[0]})
        logit = probe.decision_function(sh.transform(H[sd]))[:, None]
        for k, X in views(models, PSI, E, sd, m).items():
            for kk, vv in fit_ridge(X, logit).items():
                out[f"s_{k}_{kk}"] = vv
        for o in models:
            if o != m:
                for kk, vv in fit_ridge(PSI[o][sd], logit).items():
                    out[f"so_{o}_{kk}"] = vv
        np.savez(MOD / f"frozen_{m}.npz", **out)
        print("frozen", m, flush=True)


def confirm(cfg):
    C, models, PSI, E = setup()
    conf = np.flatnonzero((C.split == "confirm").values)
    sst = C.source.values[conf] == "sst2"
    per, cache = {}, {}
    for m in models:
        F = dict(np.load(MOD / f"frozen_{m}.npz", allow_pickle=True))
        assert list(F["models_at_freeze"]) == models, "model set changed since freeze"
        Z = np.load(L2 / m / "hidden.npz")
        H = Z["h"][conf, list(Z["layers"]).index(int(F["layer"]))].astype(np.float32)
        Y = (H - F["mu"]) @ F["comps"].T
        V = views(models, PSI, E, conf, m)
        dec = lambda pre, k: {kk: F[f"{pre}_{k}_{kk}"] for kk in ("mu", "sd", "coef", "b")}  # noqa: E731
        P = {k: predict(dec("h", k), V[k]) for k in V}
        logit = ((H[sst] - F["probe_mu"]) / F["probe_sd"]) @ F["probe_w"] + F["probe_b"]
        Ps = {k: predict(dec("s", k), V[k][sst])[:, 0] for k in ("psi", "txt")}
        Pso = {o: predict({kk: F[f"so_{o}_{kk}"] for kk in ("mu", "sd", "coef", "b")}, PSI[o][conf][sst])[:, 0]
               for o in models if o != m}
        cache[m] = (Y, P, F["w"], logit, Ps, Pso)

    def r2w(Y, P, w, i):
        r = 1 - ((Y[i] - P[i]) ** 2).sum(0) / ((Y[i] - Y[i].mean(0)) ** 2).sum(0)
        return float((r * w).sum() / w.sum())

    def r2(y, p):
        return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    def stats(i, j):   # i: indices into conf; j: indices into the sst subset
        h1, h2, h3 = [], [], []
        for m, (Y, P, w, logit, Ps, Pso) in cache.items():
            h1.append(r2w(Y, P["psi"], w, i) - r2w(Y, P["txt"], w, i))
            h2.append(r2(logit[j], Ps["psi"][j]) - np.mean([r2(logit[j], v[j]) for v in Pso.values()]))
            h3.append(r2(logit[j], Ps["psi"][j]) - r2(logit[j], Ps["txt"][j]))
        return np.mean(h1), (np.mean(h2) if h2 else np.nan), np.mean(h3)

    n, ns = len(conf), int(sst.sum())
    allv = stats(np.arange(n), np.arange(ns))
    for m, (Y, P, w, logit, Ps, Pso) in cache.items():
        per[m] = {k: round(r2w(Y, P[k], w, np.arange(n)), 4) for k in P} | \
                 {"sentiment_probe_R2_psi": round(r2(logit, Ps["psi"]), 4), "sentiment_probe_R2_txt": round(r2(logit, Ps["txt"]), 4),
                  "sentiment_probe_R2_single_others": {o: round(r2(logit, v), 4) for o, v in Pso.items()}}
    rng = np.random.default_rng(0)
    boot = np.array([stats(rng.integers(0, n, n), rng.integers(0, ns, ns)) for _ in range(cfg.n_boot)])
    res = {"n_confirm": int(n), "models": models, "per_model": per, "tests": {}}
    ps = {}
    for k, name in enumerate(("H1", "H2", "H3")):
        b = boot[:, k]
        ps[name] = float(np.mean(b <= 0))
        res["tests"][name] = {"estimate": round(float(allv[k]), 4), "ci95": [round(float(np.nanpercentile(b, 2.5)), 4),
                                                                              round(float(np.nanpercentile(b, 97.5)), 4)],
                              "p_one_sided": round(ps[name], 5)}
    order = sorted(ps, key=ps.get)
    run = 0.0
    for r_, k in enumerate(order):
        run = max(run, min(1.0, ps[k] * (3 - r_)))
        res["tests"][k]["holm_p"] = round(run, 5)
        res["tests"][k]["pass"] = run < 0.05
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "l2_confirm.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    freeze() if cfg.stage == "freeze" else confirm(cfg)


if __name__ == "__main__":
    main()
