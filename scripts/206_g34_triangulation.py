#!/usr/bin/env python
"""G34: how well does a closed model's own answer geometry reconstruct its OWN hidden geometry, which nobody outside
can see? Triangulation (docs/prereg_G34_triangulation.md). Nothing is trained; the method uses the closed model's own
sampled answers only; open models' hidden states are used to validate the estimator, never inside it.

Model (on the vector of all context-pair distances, Spearman correlations):
  hidden geometry of model m      H_m = lambda_m C + e_m
  answer geometry of answerer a   A_a = kappa_a C + f_a          (e, f idiosyncratic; G6: no own excess)
  => rho(H_a, H_b) = lambda_a lambda_b,  rho(A_a, H_b) = kappa_a lambda_b,  and rho(A_m, H_m) = kappa_m lambda_m.
lambda: rank-1 (minres) fit of the off-diagonal hidden-hidden matrix; kappa_a: least squares of rho(A_a, H_b) on
lambda_b over open b != a. The closed model's lambda_c is not identified; it takes the median of the open lambdas,
with the open range as the interval.
Data: G2's 192 confirm contexts and 24 questions (Closed-A: 8 sampled answers each). Open models (G1's 8): answers
SAMPLED to the closed design (8 draws per context and question from the exact P(yes), seed 3400; Theorem 5 debiased
distances), hidden states at G1's dev-chosen layer.
  V  leave one open model out at a time, as if closed: predict rho(A_m, H_m) from the other 7 hidden geometries.
  E  the closed model's triangulated rho(A_c, H_c), bootstrap over contexts.     CPU only.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, SEED = 500, 3400


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G = _mod("g129", "scripts/129_infogeom.py")


def loadings(R, idx):
    """rank-1 minres fit of the off-diagonal of the correlation matrix R restricted to idx"""
    S = R[np.ix_(idx, idx)]
    lam = np.sqrt(np.clip(np.nanmean(np.where(np.eye(len(idx), dtype=bool), np.nan, S), 1), 1e-6, None))
    for _ in range(200):
        new = np.array([sum(S[i, j] * lam[j] for j in range(len(idx)) if j != i) / sum(lam[j] ** 2 for j in range(len(idx)) if j != i)
                        for i in range(len(idx))])
        if np.max(np.abs(new - lam)) < 1e-9:
            break
        lam = new
    return lam


def kappa(r_row, lam):
    return float(np.dot(r_row, lam) / np.dot(lam, lam))


def data():
    C = G.contexts()
    d, cnt = G.claude_counts()
    ids = [r["id"] for r in d["results"]]
    pos = C.reset_index().set_index("id").loc[ids, "index"].values
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    _, items = G.p_yes(G.MODELS[0])
    q = [items.index(i) for i in d["items"]]
    rng = np.random.default_rng(SEED)
    A, H = {}, {}
    for m in G.MODELS:
        k_, _, _ = G.select_layer(m, dev_sel)
        p, _ = G.p_yes(m)
        P = p[np.ix_(pos, q)]
        ks = rng.binomial(8, P)
        A[m] = G.angle_dist(ks, np.full_like(ks, 8))
        Hm, _ = G.hidden(m)
        H[m] = G.cos_dist(Hm[pos, k_])
    A["closed"] = G.angle_dist(cnt[:, :, 0], cnt[:, :, 0] + cnt[:, :, 1])
    return A, H, len(ids)


def matrices(A, H, s=None):
    sub = (lambda M: M) if s is None else (lambda M: M[np.ix_(s, s)])
    ms = G.MODELS
    RHH = np.array([[1.0 if a == b else G.rho(sub(H[a]), sub(H[b])) for b in ms] for a in ms])
    RAH = np.array([[G.rho(sub(A[a]), sub(H[b])) for b in ms] for a in ms + ["closed"]])
    return RHH, RAH


def triangulate(RHH, RAH):
    ms = G.MODELS
    n = len(ms)
    out = {}
    for i, m in enumerate(ms):                                   # leave m out: m plays the closed model
        rest = [j for j in range(n) if j != i]
        lam = loadings(RHH, rest)
        k = kappa(RAH[i, rest], lam)
        out[m] = {"observed": float(RAH[i, i]), "predicted": k * float(np.median(lam)),
                  "interval": [k * float(lam.min()), k * float(lam.max())]}
    lam = loadings(RHH, list(range(n)))
    kc = kappa(RAH[n, :], lam)
    out["closed"] = {"predicted": kc * float(np.median(lam)), "interval": [kc * float(lam.min()), kc * float(lam.max())],
                     "kappa": kc, "lambda_open": [round(float(x), 4) for x in lam]}
    full = {m: float(RAH[i, i] - kappa(RAH[i, [j for j in range(n) if j != i]], lam[[j for j in range(n) if j != i]]) * lam[i])
            for i, m in enumerate(ms)}
    off = RHH[~np.eye(n, dtype=bool)] - np.outer(lam, lam)[~np.eye(n, dtype=bool)]
    out["fit"] = {"own_excess_given_true_lambda": {m: round(v, 4) for m, v in full.items()},
                  "rms_offdiag_residual_hidden": round(float(np.sqrt(np.mean(off ** 2))), 4)}
    return out


def g6_reported():
    """reported: the same leave-one-out on G6's published matrices (1,000 fresh contexts, 88 exact answers)"""
    r = json.loads((ROOT / "results/t1/g6_shared_geometry.json").read_text())
    ms = list(r["rho_answers_A_vs_hidden_B"])
    RAH = np.array([[r["rho_answers_A_vs_hidden_B"][a][b] for b in ms] for a in ms])
    RHH = np.array([[r["reported_hidden_vs_hidden"][a][b] for b in ms] for a in ms])
    out = {}
    for i, m in enumerate(ms):
        rest = [j for j in range(len(ms)) if j != i]
        lam = loadings(RHH, rest)
        k = kappa(RAH[i, rest], lam)
        out[m] = {"observed": round(float(RAH[i, i]), 4), "predicted": round(k * float(np.median(lam)), 4),
                  "interval": [round(k * float(lam.min()), 4), round(k * float(lam.max()), 4)]}
    out["mean_abs_error"] = round(float(np.mean([abs(v["observed"] - v["predicted"]) for v in out.values()])), 4)
    out["inside_interval"] = int(sum(v["interval"][0] <= v["observed"] <= v["interval"][1] for k_, v in out.items() if k_ != "mean_abs_error"))
    return out


def main():
    A, H, n = data()
    RHH, RAH = matrices(A, H)
    T = triangulate(RHH, RAH)
    ms = G.MODELS
    mae = float(np.mean([abs(T[m]["observed"] - T[m]["predicted"]) for m in ms]))
    inside = int(sum(T[m]["interval"][0] <= T[m]["observed"] <= T[m]["interval"][1] for m in ms))
    rng = np.random.default_rng(SEED + 1)
    bc, bm = [], []
    for _ in range(B_N):
        s = np.unique(rng.integers(0, n, n))
        Tb = triangulate(*matrices(A, H, s))
        bc.append(Tb["closed"]["predicted"])
        bm.append(np.mean([abs(Tb[m]["observed"] - Tb[m]["predicted"]) for m in ms]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    res = {"contexts": n, "questions": 24, "samples": 8,
           "V_leave_one_out": {m: {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in T[m].items()} for m in ms},
           "V1_mean_abs_error": [round(mae, 4)] + ci(bm), "V2_inside_interval": inside,
           "E_closed": {"predicted": [round(T["closed"]["predicted"], 4)] + ci(bc),
                        "interval_open_lambda_range": [round(x, 4) for x in T["closed"]["interval"]],
                        "kappa": round(T["closed"]["kappa"], 4), "lambda_open": T["closed"]["lambda_open"]},
           "open_own_rho_sampled_design": {m: round(T[m]["observed"], 4) for m in ms},
           "rho_closed_answers_vs_open_hidden": {m: round(float(RAH[-1, i]), 4) for i, m in enumerate(ms)},
           "fit": T["fit"], "g6_reported": g6_reported()}
    own = [T[m]["observed"] for m in ms]
    res["E2_closed_within_open_own_range"] = bool(min(own) <= T["closed"]["predicted"] <= max(own))
    res["endpoints"] = {"V1_mae_lt_0.10": res["V1_mean_abs_error"][0] < 0.10, "V2_inside_ge_6": inside >= 6,
                        "E1_closed_lower_gt_0": res["E_closed"]["predicted"][1] > 0}
    (ROOT / "results/t1/g34_triangulation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
