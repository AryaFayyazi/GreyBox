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


# ---------------------------------------------------------------- command line: measure a model's observable geometry
CLOSED_24 = ("T02", "T03", "T04", "T05", "T06", "T09", "T10", "T11", "T12", "A01", "A08", "I01", "I02", "I05", "E02",
             "E03", "P01", "P03", "P04", "P06", "S01", "U01", "U09", "U11")   # the 24 questions of the Claude tests (L3, G2)


def _load_contexts(path):
    import json
    from pathlib import Path
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            d = json.loads(line)
            out.append((str(d.get("id", len(out))), d.get("text") or d.get("messages")))
        else:
            out.append((str(len(out)), line))
    return out


def main(argv=None):
    """graybox-geometry: the observable latent geometry of a chat model over your contexts, from its answers alone."""
    import argparse
    import asyncio
    import json
    from pathlib import Path
    try:
        from .adapters import Anthropic, OpenAICompatible, estimate_cost
    except ImportError as e:                           # httpx is not a core dependency
        raise SystemExit(f"graybox-geometry needs the API client: pip install 'greybox-monitor[geometry]' ({e})")
    from .battery_v2 import BATTERY_V2
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("contexts", help="contexts: one per line, or JSON lines with 'text' (or 'messages') and optional 'id'")
    ap.add_argument("--provider", choices=["openai", "anthropic"], required=True,
                    help="openai = any OpenAI-compatible API (with --base-url); anthropic = the Messages API")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--key-env", help="environment variable holding the API key (default: OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    ap.add_argument("--no-logprobs", action="store_true", help="OpenAI-compatible API without top_logprobs: sample instead")
    ap.add_argument("--samples", type=int, default=8, help="sampled answers per question, split into two halves (sample tier)")
    ap.add_argument("--questions", choices=["24", "88"], default="24", help="the 24 questions of the closed-model tests, or all 88")
    ap.add_argument("--out", required=True, help="output JSON: distances, 2-D coordinates, eigenvalues, usage")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    a = ap.parse_args(argv)
    if a.samples < 2 or a.samples % 2:
        raise SystemExit("--samples must be even (two independent halves)")
    ctx = _load_contexts(a.contexts)
    if len(ctx) < 3:
        raise SystemExit("need at least 3 contexts")
    battery = [b for b in BATTERY_V2 if a.questions == "88" or b[0] in CLOSED_24]
    logprob = a.provider == "openai" and not a.no_logprobs
    tier = "logprob" if logprob else "sample"
    calls = len(ctx) * len(battery) * (1 if logprob else a.samples)
    toks = int(np.mean([len(str(t)) for _, t in ctx]) / 4)            # estimate_cost adds the question
    est = estimate_cost(len(ctx), len(battery), toks, tier, a.samples)
    print(f"{len(ctx)} contexts x {len(battery)} questions, tier {tier}: {calls:,} calls; {est}")
    if not a.yes and input("proceed? [y/N] ").strip().lower() != "y":
        return
    texts = [t for _, t in ctx]
    if a.provider == "openai":
        mk = lambda n: OpenAICompatible(a.model, a.base_url, a.key_env or "OPENAI_API_KEY", logprobs=logprob, n_samples=n)  # noqa: E731
    else:
        mk = lambda n: Anthropic(a.model, a.key_env or "ANTHROPIC_API_KEY", n_samples=n)  # noqa: E731
    m = len(battery)

    def run(n):
        ad = mk(n)
        r = asyncio.run(ad.battery(texts, battery))
        q = np.array([x[0] for x in r], float).reshape(len(texts), m)
        k = np.array([x[1] for x in r], float).reshape(len(texts), m)
        return q, k, ad.usage
    if logprob:
        P, _, usage = run(1)
        bad = np.isnan(P)
        P = np.where(bad, 0.5, P)                      # no yes/no token in the top 20: no information (1/2)
        D = observable_distances(fisher_angles(P))
        usage_d = vars(usage)
    else:
        (qA, nA, uA), (qB, nB, uB) = run(a.samples // 2), run(a.samples // 2)
        kA, kB = np.nan_to_num(qA) * nA, np.nan_to_num(qB) * nB
        D = crossfit_distances(np.rint(kA), nA, np.rint(kB), nB)
        bad = (nA == 0) | (nB == 0)
        usage_d = {k_: vars(uA)[k_] + vars(uB)[k_] for k_ in vars(uA)}
    X, ev = mds(D, 2)
    neg = float(-ev[ev < 0].sum() / np.abs(ev).sum())
    out = {"model": a.model, "tier": tier, "questions": [i for i, _ in battery], "ids": [i for i, _ in ctx],
           "distances": D.round(5).tolist(), "coordinates_2d": X.round(5).tolist(), "eigenvalues": ev.round(5).tolist(),
           "non_euclidean_share": round(neg, 4), "cells_without_answer": int(bad.sum()), "usage": usage_d}
    Path(a.out).write_text(json.dumps(out))
    print(f"saved {a.out}: non-Euclidean share {neg:.3f} (0 = exactly Euclidean); cells without a yes/no answer: {int(bad.sum())}")


def cli():
    main()
