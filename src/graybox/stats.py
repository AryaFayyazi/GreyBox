"""Metrics shared by all analyses: within-stratum pair-weighted AUROC (CAPE's primary metric), scenario-cluster
bootstrap, grey-box feature maps."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def within_auroc(y, s, strata) -> float:
    """Pair-weighted AUROC over strata that contain both classes: sum_k n+_k n-_k AUC_k / sum_k n+_k n-_k."""
    y, s, strata = np.asarray(y), np.asarray(s, float), np.asarray(strata)
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k
        yk = y[m]
        npos, nneg = yk.sum(), (1 - yk).sum()
        if npos == 0 or nneg == 0:
            continue
        num += npos * nneg * roc_auc_score(yk, s[m])
        den += npos * nneg
    return num / den if den else np.nan


def cluster_boot(fn, clusters, n_boot=1000, seed=0):
    """Bootstrap over clusters. fn(idx) -> statistic (or tuple). Returns array (n_boot, ...)."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uc, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(uc))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uc), len(uc))
        idx = np.concatenate([members[j] for j in pick])
        out.append(fn(idx))
    return np.array(out, dtype=float)


def yes_prob(p_yes, p_no, eps=1e-6):
    """Renormalized P(Yes | answer in {Yes, No})."""
    return (p_yes + eps) / (p_yes + p_no + 2 * eps)


def fisher_rao(q):
    """θ = 2 arcsin sqrt(q): makes the Bernoulli Fisher metric Euclidean; Var(θ̂) ≈ 1/n under n samples."""
    return 2.0 * np.arcsin(np.sqrt(np.clip(q, 0.0, 1.0)))


def sample_tier(q, n, seed=0):
    """Text-only API with n sampled one-word answers per question: q̂ = Binomial(n, q) / n."""
    rng = np.random.default_rng(seed)
    return rng.binomial(n, np.clip(q, 0, 1)) / n
