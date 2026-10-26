#!/usr/bin/env python
"""G78 (docs/prereg_G78_claude_full_battery.md): G73 asked Claude only 24 of the battery's 88 questions. Diagnosis
(exploratory, on G73's data): open models restricted to the same 24 questions agree with their own hidden geometry at
only 0.10-0.38 (all 88: 0.23-0.68), the range Claude scored. G78 asks Claude ALL 88 battery questions (BATTERY_V2,
the open models' items) on 200 fresh contexts it has never seen (G6's contexts not in G73; numpy seed 7800), 4 samples
each, plus 8 real replies; everything else is G73's code (scripts/265: frame, parser, ledger, baselines, layers).
  stage=collect cap=.. [smoke=N] (paid; smoke uses G73's already-used contexts) ; stage=embed ; stage=analyze
  -> results/t1/g78_claude_full_battery.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
N_CTX, N_SAMP, SEED, B_N, BOOT_SEED = 200, 4, 7800, 1000, 7801
OUT = Path("/data/models/graybox_ws/outputs/g78")

_s = importlib.util.spec_from_file_location("g73", ROOT / "scripts/265_g73_claude_fresh_geometry.py")
G73 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(G73)
_G73_PICKS = G73.picks


def picks():
    C, used = _G73_PICKS()
    rest = np.setdiff1d(np.arange(len(C)), used)
    return C, np.sort(np.random.default_rng(SEED).choice(rest, N_CTX, replace=False))


def items():
    from graybox.battery_v2 import BATTERY_V2
    return list(BATTERY_V2)


def stage_collect(cap, smoke=None):
    G73.N_SAMP, G73.OUT, G73.items = N_SAMP, OUT, items
    if smoke:                                                  # cost check on contexts G73 already used
        G73.picks = lambda: (lambda C, idx: (C, idx[-int(smoke):]))(*_G73_PICKS())
        G73.stage_collect(cap, int(smoke))
    else:
        G73.picks = picks
        G73.stage_collect(cap)


def stage_embed():
    G73.OUT = OUT
    G73.stage_embed()


def stage_analyze():
    g = G73._mod("g", "129_infogeom.py")
    R = sorted((json.loads(x) for x in open(OUT / "claude.jsonl")), key=lambda r: r["i"])
    ix = np.array([r["i"] for r in R])
    assert set(ix) == set(picks()[1]) and len(ix) == N_CTX
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
            "claude_replies_qemb06": g.cos_dist(np.load(OUT / "claude_replies_qemb06.npy").astype(np.float64)),
            "claude_replies_mpnet": g.cos_dist(np.load(OUT / "claude_replies_mpnet.npy").astype(np.float64))}
    C = g.contexts()
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)                 # G73's layer choice
    H, best, res = {}, {}, {"contexts": int(len(ix)), "parsed_share": round(float(cnt[..., 1].sum() / (len(R) * cnt.shape[1] * N_SAMP)), 4), "models": {}}
    for m in g.MODELS:
        kk, _, _ = g.select_layer(m, dev_sel)
        H[m] = sub(g.cos_dist(np.load(G73.RUN / m / "hidden_none.npz")["h"][:, kk].astype(np.float64)))
        r = {"claude_answers": g.rho(Dc, H[m]), **{b: g.rho(D, H[m]) for b, D in base.items()}}
        best[m] = max(base, key=lambda b: r[b])
        res["models"][m] = {**{x: round(float(v), 4) for x, v in r.items()}, "strongest_text_baseline": best[m]}
    rng = np.random.default_rng(BOOT_SEED)
    per, mean = {m: [] for m in g.MODELS}, []
    for _ in range(B_N):
        s = np.unique(rng.integers(0, len(ix), len(ix)))
        q = lambda A: A[np.ix_(s, s)]  # noqa: E731
        d = {m: g.rho(q(Dc), q(H[m])) - g.rho(q(base[best[m]]), q(H[m])) for m in g.MODELS}
        for m in g.MODELS:
            per[m].append(d[m])
        mean.append(np.mean(list(d.values())))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    for m in g.MODELS:
        res["models"][m]["claude_minus_strongest"] = [round(float(np.mean(per[m])), 4)] + ci(per[m])
    res["P1_mean_claude_minus_strongest"] = [round(float(np.mean(mean)), 4)] + ci(mean)
    res["P2_models_claude_beats_strongest"] = int(sum(res["models"][m]["claude_minus_strongest"][1] > 0 for m in g.MODELS))
    res["endpoints"] = {"P1_mean_lower_gt_0": res["P1_mean_claude_minus_strongest"][1] > 0, "P2_at_least_6_of_8": res["P2_models_claude_beats_strongest"] >= 6}
    (ROOT / "results/t1/g78_claude_full_battery.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    st = kw.get("stage", "analyze")
    if st == "collect":
        stage_collect(kw["cap"], kw.get("smoke"))
    elif st == "embed":
        stage_embed()
    else:
        stage_analyze()
