#!/usr/bin/env python
"""Exploratory (asked by review round 20): G78's 200 fresh contexts mix new kinds (news, code, jailbreak_bench, injection)
with fresh contexts of the six original kinds. G78's locked statistic (Claude's answer geometry - the strongest of the
five text baselines, Spearman with each open model's hidden geometry; the strongest chosen on the subset; mean over the 8
models; bootstrap over contexts, 1,000, seed 7801) on each subset separately.  -> results/t1/g78_by_kind_exploratory.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("g78", ROOT / "scripts/270_g78_claude_full_battery.py")
G78 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(G78)
G73 = G78.G73


def main():
    g = G73._mod("g", "129_infogeom.py")
    C = pd.read_parquet(G73.CTX)
    R = sorted((json.loads(x) for x in open(G78.OUT / "claude.jsonl")), key=lambda r: r["i"])
    ix = np.array([r["i"] for r in R])
    cnt = np.array([[[c[0], c[0] + c[1]] for c in r["counts"]] for r in R], float)
    k, n = cnt[..., 0], np.maximum(cnt[..., 1], 1)
    th = 2 * np.arcsin(np.sqrt((k + 3 / 8) / (n + 3 / 4)))
    var = 1 / (n + 0.5)
    G = th @ th.T
    nr = np.diag(G)
    Dc = np.sqrt(np.maximum(nr[:, None] + nr[None, :] - 2 * G - (var.sum(1)[:, None] + var.sum(1)[None, :]), 0))
    np.fill_diagonal(Dc, 0)
    sub = lambda A: A[np.ix_(ix, ix)]  # noqa: E731
    base = {"input_minilm": sub(g.cos_dist(np.load(G73.RUN / "contexts_minilm.npy").astype(np.float64))),
            "input_qemb06": sub(g.cos_dist(np.load(G73.SE / "fresh_input_qemb06.npy").astype(np.float64))),
            "input_mpnet": sub(g.cos_dist(np.load(G73.SE / "fresh_input_mpnet.npy").astype(np.float64))),
            "claude_replies_qemb06": g.cos_dist(np.load(G78.OUT / "claude_replies_qemb06.npy").astype(np.float64)),
            "claude_replies_mpnet": g.cos_dist(np.load(G78.OUT / "claude_replies_mpnet.npy").astype(np.float64))}
    Cc = g.contexts()
    dev = np.where(Cc.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    H = {m: sub(g.cos_dist(np.load(G73.RUN / m / "hidden_none.npz")["h"][:, g.select_layer(m, dev_sel)[0]].astype(np.float64))) for m in g.MODELS}
    new = C.new_kind.values[ix].astype(bool)
    out = {}
    for name, mask in (("new_kinds", new), ("original_kinds_fresh", ~new)):
        sel = np.flatnonzero(mask)
        q = lambda A, s=sel: A[np.ix_(s, s)]  # noqa: E731
        best = {m: max(base, key=lambda b: g.rho(q(base[b]), q(H[m]))) for m in g.MODELS}
        per = {m: round(float(g.rho(q(Dc), q(H[m])) - g.rho(q(base[best[m]]), q(H[m]))), 4) for m in g.MODELS}
        rng, bs = np.random.default_rng(G78.BOOT_SEED), []
        for _ in range(G78.B_N):
            s = np.unique(rng.choice(sel, len(sel)))
            r = lambda A: A[np.ix_(s, s)]  # noqa: E731
            bs.append(np.mean([g.rho(r(Dc), r(H[m])) - g.rho(r(base[best[m]]), r(H[m])) for m in g.MODELS]))
        out[name] = {"contexts": int(len(sel)), "kinds": sorted(set(C.kind.values[ix][mask])),
                     "claude_agreement": {m: round(float(g.rho(q(Dc), q(H[m]))), 4) for m in g.MODELS},
                     "claude_minus_strongest_per_model": per, "models_claude_ahead": int(sum(v > 0 for v in per.values())),
                     "mean_claude_minus_strongest": [round(float(np.mean(list(per.values()))), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    (ROOT / "results/t1/g78_by_kind_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {x: v[x] for x in ("contexts", "kinds", "models_claude_ahead", "mean_claude_minus_strongest")} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
