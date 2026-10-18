"""The observable latent geometry of a model, from its outputs alone (no training; numpy only).

Theory: docs/theory_infogeom.md. A model's hidden coordinates are not identifiable from outputs, but the geometry of its
states as seen through its answers is, and it is measured exactly:

* Ask yes/no questions in side branches after each context. The Fisher-Rao distance between two contexts' answer
  distributions is Euclidean in the angles theta = 2 arcsin sqrt(P(yes)) (Theorem 1), and by Cencov's theorem it is the
  only output geometry invariant under sufficient statistics (Theorem 2).
* From sampled answers (any chat API), `crossfit_distances` is exactly unbiased, for every P(yes), for the geometry
  that n samples identify: the Euclidean geometry of mu_n(P(yes)) = E[Anscombe angle], a known increasing function
  that tends to the angle as n grows (Theorem 5', docs/theory_infogeom.md, Correction 1). Classical MDS gives
  coordinates up to a rigid motion.
* With full log-probabilities, the Gram matrix of the final normalized hidden states is recovered exactly by a linear
  solve: the RMSNorm sphere gives b' S b = d for each context (Theorem 6, `recover_gram`).

    from graybox.geometry import interrogate, fisher_angles_from_counts, observable_distances, mds
    (kA, nA), (kB, nB) = [interrogate(ask, contexts, questions, samples=4) for _ in range(2)]   # two independent halves
    D = crossfit_distances(kA, nA, kB, nB)          # ask(context, question) -> "yes" | "no" | other
    X, eigenvalues = mds(D, 2)
"""
from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------- Theorems 1 and 5: answers
def fisher_angles(p) -> np.ndarray:
    """Exact flat coordinates of Bernoulli answers: theta = 2 arcsin sqrt(p) (Theorem 1)."""
    return 2 * np.arcsin(np.sqrt(np.clip(np.asarray(p, float), 0, 1)))


def fisher_angles_from_counts(k, n) -> np.ndarray:
    """Anscombe angles from k yes answers out of n valid ones. For n p (1 - p) large: mean theta + cot(theta)/(4n) and
    variance 1/(n + 1/2); near p = 0 or 1 both are smaller in size (Correction 1)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    return 2 * np.arcsin(np.sqrt((k + 3 / 8) / (np.maximum(n, 0) + 3 / 4)))


def observable_distances(theta, n=None) -> np.ndarray:
    """Fisher-Rao distances between all rows of theta (contexts x questions). With sample counts n (same shape, or a
    scalar), the large-n sampling variance sum_q (1/(n+1/2)) per context is subtracted from each squared distance; this
    over-corrects for saturated answers (Correction 1): prefer `crossfit_distances` for sampled answers."""
    th = np.asarray(theta, float)
    g = th @ th.T
    nrm = np.diag(g)
    d2 = nrm[:, None] + nrm[None, :] - 2 * g
    if n is not None:
        var = 1 / (np.broadcast_to(np.asarray(n, float), th.shape) + 0.5)
        v = var.sum(1)
        d2 = d2 - (v[:, None] + v[None, :])
    d = np.sqrt(np.maximum(d2, 0))
    np.fill_diagonal(d, 0)
    return d


def mu_n(p, n: int) -> np.ndarray:
    """E[Anscombe angle] for k ~ Binomial(n, p): the coordinate that n samples identify (exact binomial sum)."""
    from math import comb
    p = np.asarray(p, float)
    k = np.arange(n + 1)
    w = np.array([comb(n, int(i)) for i in k], float) * p[..., None] ** k * (1 - p[..., None]) ** (n - k)
    return (w * fisher_angles_from_counts(k, n)).sum(-1)


def crossfit_distances(kA, nA, kB, nB) -> np.ndarray:
    """Distances from two independent halves of the samples (k yes of n valid, contexts x questions, per half):
    D^2(x, x') = sum_q (a_q(x) - a_q(x')) (b_q(x) - b_q(x')) with a, b the halves' Anscombe angles. By independence its
    expectation is exactly sum_q (mu_n(P_q(x)) - mu_n(P_q(x')))^2 for every P (Theorem 5'). Negative values (noise) are
    returned as 0 distance."""
    a = fisher_angles_from_counts(kA, nA)
    b = fisher_angles_from_counts(kB, nB)
    d2 = (a * b).sum(1)
    d2 = d2[:, None] + d2[None, :] - a @ b.T - b @ a.T
    d = np.sqrt(np.maximum(d2, 0))
    np.fill_diagonal(d, 0)
    return d


def mds(D, k: int = 2):
    """Classical multidimensional scaling: coordinates (rows) whose Euclidean distances reproduce D, exact up to a rigid
    motion when D is Euclidean; returns (coordinates, all eigenvalues in decreasing order). Negative eigenvalues measure
    how far D is from Euclidean (sampling noise, debiasing)."""
    D = np.asarray(D, float)
    m = len(D)
    J = np.eye(m) - 1.0 / m
    B = -0.5 * J @ (D ** 2) @ J
    ev, V = np.linalg.eigh(B)
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    return V[:, :k] * np.sqrt(np.maximum(ev[:k], 0)), ev


def _ranks(x):
    """Average ranks (ties share their mean rank), as scipy.stats.rankdata."""
    x = np.asarray(x)
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    r = np.empty(len(x), float)
    start = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1]
    end = np.r_[start[1:], len(x)]
    for s, e in zip(start, end):
        r[o[s:e]] = 0.5 * (s + e - 1) + 1
    return r


def rsa(D1, D2) -> float:
    """Spearman correlation between two distance matrices over all pairs (representational similarity; no fit)."""
    iu = np.triu_indices(len(D1), 1)
    a, b = _ranks(np.asarray(D1)[iu]), _ranks(np.asarray(D2)[iu])
    return float(np.corrcoef(a, b)[0, 1])


def interrogate(ask: Callable[[str, str], str], contexts: Sequence[str], questions: Sequence[str], samples: int = 8):
    """Ask every question after every context, `samples` times, each in its own call (a side branch: the answers are
    independent given the context). `ask(context, question)` returns the model's answer text; answers starting with
    yes / no count, others are dropped. Returns (k yes, n valid), each contexts x questions. Cost: contexts x questions
    x samples calls; with a model that returns token probabilities, use `fisher_angles(P(yes) / (P(yes) + P(no)))`."""
    k = np.zeros((len(contexts), len(questions)), int)
    n = np.zeros_like(k)
    for i, c in enumerate(contexts):
        for j, q in enumerate(questions):
            for _ in range(samples):
                a = str(ask(c, q)).strip().lower()
                if a.startswith("yes"):
                    k[i, j] += 1
                    n[i, j] += 1
                elif a.startswith("no"):
                    n[i, j] += 1
    return k, n


# ---------------------------------------------------------------- Theorem 6: exact recovery from log-probabilities
def logprob_basis(logprobs, d: int):
    """Centered log-probabilities (rows: contexts, columns: the whole vocabulary) and an orthonormal basis Q (V x d) of
    their span. For a pre-norm model with logits W u and no output bias, the span has rank d (the hidden size); the
    singular values after the d-th should be at rounding level, and the returned ratio sigma_d / sigma_{d+1} shows it."""
    C = np.asarray(logprobs, float)
    C = C - C.mean(1, keepdims=True)
    _, s, Vt = np.linalg.svd(C, full_matrices=False)
    gap = float(s[d - 1] / s[d]) if len(s) > d else float("inf")
    return Vt[:d].T, gap


def solve_sphere(B, target: float, max_iter: int = 5000, tol: float = 1e-12):
    """Least squares for symmetric S in b_i' S b_i = target for every row b_i of B, by conjugate gradient on the normal
    equations (from S = 0: the minimum-norm solution when underdetermined). Returns (S, iterations, relative residual)."""
    B = np.asarray(B, float)
    d = B.shape[1]

    def A(Sm):
        q = np.einsum("ij,jk,ik->i", B, Sm, B)
        return (B * q[:, None]).T @ B
    rhs = (B * target).T @ B
    S = np.zeros((d, d))
    r = rhs.copy()
    p = r.copy()
    rs = float((r * r).sum())
    r0 = np.sqrt(rs)
    it, rel = 0, 1.0
    for it in range(1, max_iter + 1):
        Ap = A(p)
        alpha = rs / float((p * Ap).sum())
        S += alpha * p
        r -= alpha * Ap
        rs_new = float((r * r).sum())
        rel = np.sqrt(rs_new) / r0
        if rel < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return (S + S.T) / 2, it, rel


def recover_gram(B_train, B_query, d: int | None = None, max_iter: int = 5000):
    """Theorem 6: the Gram matrix of the final normalized hidden states u = x / rms(x) of the query contexts, from their
    projected centered log-probabilities B (rows; `logprob_basis`). B_train needs at least d(d+1)/2 contexts (twice that
    is safer); fewer leave S undetermined and a warning is raised. The states themselves follow up to an orthogonal map:
    U = B_query @ L with L L' = S. Returns (Gram of the query contexts, S, info)."""
    Bt = np.asarray(B_train, float)
    Bq = np.asarray(B_query, float)
    d = d or Bt.shape[1]
    need = d * (d + 1) // 2
    if len(Bt) < need:
        warnings.warn(f"{len(Bt)} training contexts < d(d+1)/2 = {need}: S is not determined; the Gram is not exact")
    M = Bt.T @ Bt / len(Bt)
    ev, V = np.linalg.eigh(M)
    W = V @ np.diag(ev ** -0.5) @ V.T                    # whitening: any invertible map of B is allowed (U = K B)
    S_w, it, rel = solve_sphere(Bt @ W, float(d), max_iter)
    S = W @ S_w @ W.T
    G = Bq @ S @ Bq.T
    return G, S, {"contexts": len(Bt), "unknowns": need, "cg_iterations": it, "cg_relative_residual": rel,
                  "determined": len(Bt) >= need}


def cosines(G) -> np.ndarray:
    """Cosine matrix of a Gram matrix."""
    G = np.asarray(G, float)
    s = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    return G / s[:, None] / s[None, :]
