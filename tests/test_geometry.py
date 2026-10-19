"""The theorems of docs/theory_infogeom.md, checked numerically on cases with a known answer."""
import warnings

import numpy as np
import pytest

from graybox.geometry import (cosines, fisher_angles, fisher_angles_from_counts, interrogate, logprob_basis, mds,
                              observable_distances, recover_gram, rsa)


def test_theorem1_bhattacharyya():
    """One factor: the Fisher-Rao distance 2 arccos(BC) equals the angle difference."""
    rng = np.random.default_rng(0)
    p, q = rng.random(1000), rng.random(1000)
    bc = np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))
    assert np.allclose(2 * np.arccos(np.clip(bc, -1, 1)), np.abs(fisher_angles(p) - fisher_angles(q)), atol=1e-6)


def test_theorem5_debiased_distance_is_unbiased():
    rng = np.random.default_rng(1)
    P = rng.uniform(0.15, 0.85, (2, 40))
    true = np.sum((fisher_angles(P[0]) - fisher_angles(P[1])) ** 2)
    n = 16
    est = [observable_distances(fisher_angles_from_counts(rng.binomial(n, P), n), n)[0, 1] ** 2 for _ in range(4000)]
    assert abs(np.mean(est) - true) < 0.05 * true + 0.05


def test_mds_exact_up_to_rigid_motion():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 3))
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    Y, ev = mds(D, 3)
    assert np.allclose(np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1)), D, atol=1e-8)
    assert ev[3:].max() < 1e-8


def test_rsa_ties_and_identity():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 4))
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    assert rsa(D, D) == pytest.approx(1.0)
    assert rsa(D, np.round(D)) > 0.8


def test_interrogate_counts():
    k, n = interrogate(lambda c, q: "Yes." if c == "a" else ("no" if q == "x" else "maybe"), ["a", "b"], ["x", "y"], samples=3)
    assert k.tolist() == [[3, 3], [0, 0]] and n.tolist() == [[3, 3], [3, 0]]


def _toy_model(rng, d=10, V=300, N=200):
    """A pre-norm output head: logits = W (gamma * u), u = x / rms(x); returns log-probabilities and the true u."""
    W, gamma = rng.normal(size=(V, d)), rng.uniform(0.5, 2, d)
    x = rng.normal(size=(N, d)) @ np.diag(np.logspace(0, -1, d)) + rng.normal(size=d)
    u = x / np.sqrt((x ** 2).mean(1, keepdims=True))
    z = (u * gamma) @ W.T
    lp = z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
    return lp, u


def test_theorem6_exact_recovery():
    rng = np.random.default_rng(4)
    d = 10
    lp, u = _toy_model(rng, d=d, N=2 * d * (d + 1) // 2 + 60)
    Q, gap = logprob_basis(lp, d)
    assert gap > 1e6                                          # rank d exactly
    B = (lp - lp.mean(1, keepdims=True)) @ Q
    G, S, info = recover_gram(B[:-60], B[-60:])
    assert info["determined"]
    assert np.abs(cosines(G) - cosines(u[-60:] @ u[-60:].T)).max() < 1e-6
    assert np.allclose(np.diag(G), d, rtol=1e-6)              # the sphere ||u||^2 = d holds on new contexts


def test_theorem6_underdetermined_warns_and_fails():
    rng = np.random.default_rng(5)
    d = 10
    lp, u = _toy_model(rng, d=d, N=40 + 60)
    Q, _ = logprob_basis(lp, d)
    B = (lp - lp.mean(1, keepdims=True)) @ Q
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        G, _, info = recover_gram(B[:40], B[-60:])
    assert not info["determined"] and any("not determined" in str(x.message) for x in w)
    assert np.abs(cosines(G) - cosines(u[-60:] @ u[-60:].T)).max() > 1e-3


def test_theorem5prime_crossfit_exactly_unbiased_for_saturated_answers():
    from graybox.geometry import crossfit_distances, mu_n
    rng = np.random.default_rng(6)
    P = np.clip(rng.beta(0.3, 0.3, (2, 24)), 1e-4, 1 - 1e-4)          # saturated, as language models' answers are
    n = 4
    target = ((mu_n(P[0], n) - mu_n(P[1], n)) ** 2).sum()
    est = []
    for _ in range(20000):
        kA, kB = rng.binomial(n, P), rng.binomial(n, P)
        a, b = fisher_angles_from_counts(kA, n), fisher_angles_from_counts(kB, n)
        est.append(((a[0] - a[1]) * (b[0] - b[1])).sum())                 # the unclipped estimator
    assert abs(np.mean(est) - target) < 4 * np.std(est) / np.sqrt(len(est))
    D = crossfit_distances(rng.binomial(n, P), np.full(P.shape, n), rng.binomial(n, P), np.full(P.shape, n))
    assert D.shape == (2, 2) and D[0, 0] == 0 and D[0, 1] == D[1, 0]


def test_mu_n_increasing_and_converges():
    from graybox.geometry import mu_n
    p = np.linspace(0.001, 0.999, 200)
    assert np.all(np.diff(mu_n(p, 8)) > 0)
    assert np.abs(mu_n(p[20:-20], 400) - fisher_angles(p[20:-20])).max() < 0.01


def test_cli_offline(tmp_path, monkeypatch):
    """graybox-geometry end to end, with the provider replaced by a fake model whose P(yes) is known."""
    import json
    pytest.importorskip("httpx")                      # the [geometry] extra
    import graybox.adapters as ad
    from graybox.geometry import main
    rng = np.random.default_rng(7)
    ctx = [f"context {i}" for i in range(12)]
    truth = {c: rng.random() for c in ctx}

    async def fake_battery(self, contexts, battery):
        out = []
        for c in contexts:
            for j, _ in enumerate(battery):
                p = (truth[c] + 0.1 * j) % 1
                if self.logprobs:
                    out.append((p, float("inf"), "logprob"))
                else:
                    k = rng.binomial(self.n_samples, p)
                    out.append((k / self.n_samples, self.n_samples, "sample"))
        return out
    monkeypatch.setattr(ad.OpenAICompatible, "battery", fake_battery)
    f = tmp_path / "c.txt"
    f.write_text("\n".join(ctx))
    for extra in ([], ["--no-logprobs", "--samples", "8"]):
        out = tmp_path / "g.json"
        main([str(f), "--provider", "openai", "--model", "fake", "--out", str(out), "--yes", *extra])
        d = json.loads(out.read_text())
        D = np.array(d["distances"])
        assert D.shape == (12, 12) and np.allclose(D, D.T) and len(d["coordinates_2d"]) == 12
        if not extra:                                   # logprob tier: exact distances
            P = np.array([[(truth[c] + 0.1 * j) % 1 for j in range(24)] for c in ctx])
            assert np.allclose(D, observable_distances(fisher_angles(P)), atol=1e-4)
