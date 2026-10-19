#!/usr/bin/env python
"""EXPLORATORY (after G1 and G5's locks): does the answer geometry match the hidden geometry WITHIN one kind of context,
or only between kinds? Per model and kind (240 confirm contexts each): Spearman of the answer, text and (if G5 exists)
reply geometries with the hidden geometry at G1's layer; and the pooled-within statistic (pairs of the same kind only).
python scripts/135_within_kind_exploratory.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/129_infogeom.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
G5 = Path("/data/models/graybox_ws/outputs/g5/confirm")


def rho_same_kind(A, B, kind):
    """Spearman over the pairs of contexts of the same kind only."""
    from scipy.stats import rankdata
    iu = np.triu_indices(len(A), 1)
    same = kind[iu[0]] == kind[iu[1]]
    a, b = rankdata(A[iu][same]), rankdata(B[iu][same])
    return float(np.corrcoef(a, b)[0, 1])


def main():
    C = g.contexts()
    test = np.where(C.split.values == "confirm")[0]
    kind = C.source.values[test]
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    emb = np.load(g.L2 / "minilm.npy").astype(np.float64)[test]
    res = {"models": {}}
    wins_text = wins_rep = cells = 0
    for m in g.MODELS:
        k, layer, _ = g.select_layer(m, dev_sel)
        H, _ = g.hidden(m)
        p, _ = g.p_yes(m)
        Dh, Da, Dt = g.cos_dist(H[test, k]), np.sqrt(g.sq_dist(g.theta(p[test]))), g.cos_dist(emb)
        Dr = None
        f = G5 / m / "replies_none.minilm.npy"
        if f.exists():
            Dr = g.cos_dist(np.load(f).reshape(len(test), 8, -1).astype(np.float64).mean(1))
        r = {"pooled_same_kind": {"answers": round(rho_same_kind(Da, Dh, kind), 4), "text": round(rho_same_kind(Dt, Dh, kind), 4)}}
        if Dr is not None:
            r["pooled_same_kind"]["replies"] = round(rho_same_kind(Dr, Dh, kind), 4)
        for s in sorted(set(kind)):
            i = kind == s
            sub = lambda A: A[np.ix_(i, i)]  # noqa: E731
            c = {"answers": round(g.rho(sub(Da), sub(Dh)), 4), "text": round(g.rho(sub(Dt), sub(Dh)), 4)}
            if Dr is not None:
                c["replies"] = round(g.rho(sub(Dr), sub(Dh)), 4)
                wins_rep += c["answers"] > c["replies"]
            wins_text += c["answers"] > c["text"]
            cells += 1
            r[s] = c
        res["models"][m] = r
        print(m, json.dumps(r["pooled_same_kind"]), flush=True)
    allv = [v for r in res["models"].values() for s, v in r.items() if s != "pooled_same_kind"]
    res["cells"] = cells
    res["answers_beat_text_cells"] = int(wins_text)
    res["answers_beat_replies_cells"] = int(wins_rep) if Dr is not None else None
    res["median_within_kind"] = {k_: round(float(np.median([v[k_] for v in allv])), 3) for k_ in allv[0]}
    res["pooled_same_kind_range"] = {k_: [min(r["pooled_same_kind"][k_] for r in res["models"].values()),
                                          max(r["pooled_same_kind"][k_] for r in res["models"].values())] for k_ in res["models"][g.MODELS[0]]["pooled_same_kind"]}
    (g.RES / "within_kind_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: res[k_] for k_ in ("cells", "answers_beat_text_cells", "answers_beat_replies_cells", "median_within_kind", "pooled_same_kind_range")}, indent=1))


if __name__ == "__main__":
    main()
