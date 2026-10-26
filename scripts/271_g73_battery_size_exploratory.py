#!/usr/bin/env python
"""Exploratory (not preregistered): why G73 failed. On G73's 400 fresh contexts, each open model's own answer geometry
against its own hidden geometry (G73's layer and distances), with all 88 battery questions (exact P(yes)), with only the
24 questions G73 asked Claude (exact), and with those 24 sampled 8 times (binomial from the exact P(yes), Theorem 5's
debiased distances, numpy seed 1); next to Claude's 24-question geometry (G73) and its agreement with each open model's
24-question answer geometry. Run before G78 was locked; this file records it.  -> results/t1/g73_battery_size_exploratory.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def _mod(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "scripts" / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def deb(k, n):
    th = 2 * np.arcsin(np.sqrt((k + 3 / 8) / (n + 3 / 4)))
    var = 1 / (n + 0.5)
    G = th @ th.T
    nr = np.diag(G)
    D = np.sqrt(np.maximum(nr[:, None] + nr[None, :] - 2 * G - (var.sum(1)[:, None] + var.sum(1)[None, :]), 0))
    np.fill_diagonal(D, 0)
    return D


def main():
    G73 = _mod("g73", "265_g73_claude_fresh_geometry.py")
    g, g4 = G73._mod("g", "129_infogeom.py"), G73._mod("g4", "132_g4_shift.py")
    R = sorted((json.loads(x) for x in open(G73.OUT / "claude.jsonl")), key=lambda r: r["i"])
    ix = np.array([r["i"] for r in R])
    cnt = np.array([[[c[0], c[0] + c[1]] for c in r["counts"]] for r in R], float)
    Dc = deb(cnt[..., 0], np.maximum(cnt[..., 1], 1))
    items24 = [q for q, _ in G73.items()]
    C = g.contexts()
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    sub = lambda A: A[np.ix_(ix, ix)]  # noqa: E731
    rng = np.random.default_rng(1)
    out = {"contexts": int(len(ix)), "claude_unanimous_cells": round(float(((cnt[..., 0] == 0) | (cnt[..., 0] == cnt[..., 1])).mean()), 4), "models": {}}
    for m in g.MODELS:
        kk, _, _ = g.select_layer(m, dev_sel)
        H = sub(g.cos_dist(np.load(G73.RUN / m / "hidden_none.npz")["h"][:, kk].astype(np.float64)))
        z = np.load(G73.RUN / m / "psi_none.npz", allow_pickle=True)
        its, p = list(z["items"]), g4.p_of(z)[ix]
        cols = [its.index(q) for q in items24]
        D88 = np.sqrt(g.sq_dist(g.theta(p)))
        D24 = np.sqrt(g.sq_dist(g.theta(p[:, cols])))
        ks = rng.binomial(8, p[:, cols]).astype(float)
        D24s = deb(ks, np.full(ks.shape, 8.0))
        out["models"][m] = {"own_88_exact": round(g.rho(D88, H), 4), "own_24_exact": round(g.rho(D24, H), 4),
                            "own_24_sampled8": round(g.rho(D24s, H), 4), "claude_24_sampled8": round(g.rho(Dc, H), 4),
                            "claude_vs_open_24_answer_geometry": round(g.rho(Dc, D24), 4)}
    (ROOT / "results/t1/g73_battery_size_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
