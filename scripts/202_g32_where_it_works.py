#!/usr/bin/env python
"""G32: does a label-free quantity, computed from the agent's own answers alone, say where the read-out works?
(docs/prereg_G32_where_it_works.md). Data: G25's targets (each agent's OWN answers, 8 questions x 4 answers):
CAPE-Bench (Gemma-4, gpt-oss-20b, Qwen3.8, Muse-Glimmer, Closed-A), AgentDojo (Gemma-4, gpt-oss-20b), tau2-bench
(Closed-A). Cells: (target, stratum) with >= 5 violations and >= 5 non-violations. Per cell, with no label:
  s_i     = sum_j sigma_j theta_ij / K                    (the fixed read-out; theta: Anscombe angle)
  N_i     = sum_j [m_ij > 0] / (m_ij + 1/2) / K^2          (its sampling-noise variance, from the theory)
  SNR     = Var_cell(s) / mean_cell(N) - 1               (variance beyond sampling noise)
and, with labels, AUROC_cell(s). Ceiling (Gaussian, equal variance; within-class variance >= noise):
  d'^2 <= SNR+ / (p(1-p))  =>  AUROC <= Phi( sqrt(SNR+ / (2 p (1-p))) ).   CPU only.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
B_N, SEED, MIN_EACH = 2000, 3200, 5


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def readout(C, sg):
    k, m = C[..., 0], C[..., 1]
    th = 2 * np.arcsin(np.sqrt((k + 3 / 8) / (m + 3 / 4)))
    K = C.shape[1]
    return th @ sg / K, ((m > 0) / (m + 0.5)).sum(1) / K ** 2


def cells(min_each=MIN_EACH):
    G25 = _mod("g25", "scripts/192_g25_fisher_snr_own.py")
    out = []
    for name, C, sg, y, st, _ in G25.targets():
        s, N = readout(C, sg)
        for u in np.unique(st):
            i = st == u
            if y[i].sum() >= min_each and (1 - y[i]).sum() >= min_each:
                p = y[i].mean()
                snr = s[i].var(ddof=1) / N[i].mean() - 1
                out.append({"target": name, "stratum": str(u), "n": int(i.sum()), "violations": int(y[i].sum()), "base_rate": float(p),
                            "snr": float(snr), "raw_var": float(s[i].var(ddof=1)), "auroc": float(roc_auc_score(y[i], s[i])),
                            "ceiling": float(norm.cdf(np.sqrt(max(snr, 0) / (2 * p * (1 - p)))))})
    return out


def partial_spearman(x, y, Z):
    r = lambda v: rankdata(v)  # noqa: E731
    Zr = np.column_stack([np.ones(len(x))] + [r(z) for z in Z])
    ex = r(x) - Zr @ np.linalg.lstsq(Zr, r(x), rcond=None)[0]
    ey = r(y) - Zr @ np.linalg.lstsq(Zr, r(y), rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def target_level():
    """reported: per target, the read-out's pooled within-stratum SNR and within-stratum AUROC"""
    from graybox.stats import within_auroc
    G25 = _mod("g25", "scripts/192_g25_fisher_snr_own.py")
    res = {}
    for name, C, sg, y, st, _ in G25.targets():
        s, N = readout(C, sg)
        dev = np.concatenate([s[st == u] - s[st == u].mean() for u in np.unique(st)])
        nu = sum((st == u).sum() - 1 for u in np.unique(st))
        res[name] = {"snr": round(float((dev ** 2).sum() / max(nu, 1) / N.mean() - 1), 4),
                     "within_auroc": round(float(within_auroc(y, s, st)), 4)}
    return res


def main():
    X = cells()
    snr, auc = np.array([c["snr"] for c in X]), np.array([c["auroc"] for c in X])
    rng = np.random.default_rng(SEED)
    b1, b2 = [], []
    for _ in range(B_N):
        i = rng.integers(0, len(X), len(X))
        b1.append(spearmanr(snr[i], auc[i]).correlation)
        pos, neg = auc[i][snr[i] > 0], auc[i][snr[i] <= 0]
        b2.append(pos.mean() - neg.mean() if len(pos) and len(neg) else np.nan)
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    pos, neg = auc[snr > 0], auc[snr <= 0]
    res = {"cells": len(X), "C1_spearman_snr_auroc": [round(float(spearmanr(snr, auc).correlation), 4)] + ci(b1),
           "C2_auroc_snr_pos_minus_nonpos": [round(float(pos.mean() - neg.mean()), 4) if len(pos) and len(neg) else None] + ci(b2),
           "cells_snr_pos": int(len(pos)), "cells_snr_nonpos": int(len(neg)),
           "mean_auroc_snr_pos": round(float(pos.mean()), 4) if len(pos) else None,
           "mean_auroc_snr_nonpos": round(float(neg.mean()), 4) if len(neg) else None,
           "reported": {
               "spearman_rawvar_auroc": round(float(spearmanr([c["raw_var"] for c in X], auc).correlation), 4),
               "spearman_n_auroc": round(float(spearmanr([c["n"] for c in X], auc).correlation), 4),
               "spearman_baserate_auroc": round(float(spearmanr([c["base_rate"] for c in X], auc).correlation), 4),
               "partial_spearman_snr_auroc_given_n_baserate": round(partial_spearman(snr, auc, [[c["n"] for c in X], [c["base_rate"] for c in X]]), 4),
               "share_cells_within_ceiling_plus_0.05": round(float(np.mean([c["auroc"] <= c["ceiling"] + 0.05 for c in X])), 4),
               "cells_at_min_3": len(cells(3)),
               "spearman_at_min_3": round(float(spearmanr([c["snr"] for c in cells(3)], [c["auroc"] for c in cells(3)]).correlation), 4)},
           "target_level": target_level(), "per_cell": X}
    res["endpoints"] = {"C1_lower_gt_0": res["C1_spearman_snr_auroc"][1] > 0, "C2_lower_gt_0": res["C2_auroc_snr_pos_minus_nonpos"][1] > 0}
    (ROOT / "results/t1/g32_where_it_works.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_cell"}, indent=1))


if __name__ == "__main__":
    main()
