#!/usr/bin/env python
"""EXPLORATORY (reviewer: "the answers' advantage is mostly the separation of kinds of context"). Per model, on G1's
confirm contexts (1,440, 6 kinds) and G6's fresh contexts (1,000, 10 kinds), with G1's distances, layer and Spearman over
all context pairs (scripts/129, 137):
  1. rho(kind indicator, hidden): kind distance 0 if same kind, else 1 (ties handled by average ranks)
  2. partial Spearman of answers (and text) vs hidden, controlling for the kind indicator (partial Pearson of ranks)
  3. within-kind rho per kind (answers, text), median over kinds; and pooled same-kind pairs
  4. rho(answers) - rho(kind), rho(text) - rho(kind)
Also reproduces the original rho(answers), rho(text) as a check.
python scripts/168_kind_baseline.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = _mod("g", "129_infogeom.py")
g4 = _mod("g4", "132_g4_shift.py")
G6 = Path("/data/models/graybox_ws/outputs/g6")


def partial(rx, ry, rz):
    """Partial correlation of x and y given z, all already rank-transformed."""
    rxy, rxz, ryz = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def analyse(Da, Dt, Dh, kind):
    kind = np.asarray(kind, dtype=object).astype(str)
    iu = np.triu_indices(len(kind), 1)
    Dk = (kind[:, None] != kind[None, :]).astype(float)
    ra, rt, rh, rk = (rankdata(M[iu]) for M in (Da, Dt, Dh, Dk))
    c = lambda x, y: float(np.corrcoef(x, y)[0, 1])  # noqa: E731
    same = Dk[iu] == 0
    r = {"rho_answers": c(ra, rh), "rho_text": c(rt, rh), "rho_kind": c(rk, rh),
         "rho_answers_vs_kind": c(ra, rk), "rho_text_vs_kind": c(rt, rk),
         "partial_answers_given_kind": partial(ra, rh, rk), "partial_text_given_kind": partial(rt, rh, rk)}
    r["answers_minus_kind"] = r["rho_answers"] - r["rho_kind"]
    r["text_minus_kind"] = r["rho_text"] - r["rho_kind"]
    r["pooled_same_kind_answers"] = c(rankdata(Da[iu][same]), rankdata(Dh[iu][same]))
    r["pooled_same_kind_text"] = c(rankdata(Dt[iu][same]), rankdata(Dh[iu][same]))
    pk = {}
    for s in sorted(set(kind)):
        i = np.where(kind == s)[0]
        sub = lambda A: A[np.ix_(i, i)]  # noqa: E731
        pk[s] = {"answers": round(g.rho(sub(Da), sub(Dh)), 4), "text": round(g.rho(sub(Dt), sub(Dh)), 4)}
    r["within_kind_median_answers"] = float(np.median([v["answers"] for v in pk.values()]))
    r["within_kind_median_text"] = float(np.median([v["text"] for v in pk.values()]))
    r = {k: round(v, 4) for k, v in r.items()}
    r["per_kind"] = pk
    return r


def summary(models):
    keys = [k for k in next(iter(models.values())) if k != "per_kind"]
    out = {"median_over_models": {k: round(float(np.median([m[k] for m in models.values()])), 4) for k in keys}}
    cells = [v for m in models.values() for v in m["per_kind"].values()]
    out["median_over_model_kind_cells"] = {k: round(float(np.median([v[k] for v in cells])), 4) for k in ("answers", "text")}
    out["answers_beat_text_partial"] = sum(m["partial_answers_given_kind"] > m["partial_text_given_kind"] for m in models.values())
    return out


def main():
    C = g.contexts()
    test = np.where(C.split.values == "confirm")[0]
    kind1 = C.source.values[test]
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    emb = np.load(g.L2 / "minilm.npy").astype(np.float64)[test]
    Dt1 = g.cos_dist(emb)
    C6 = __import__("pandas").read_parquet(G6 / "contexts.parquet").reset_index(drop=True)
    kind6 = C6["kind"].values
    Dt6 = g.cos_dist(np.load(G6 / "run" / "contexts_minilm.npy").astype(np.float64))
    orig1 = json.loads((g.RES / "g1_open_geometry.json").read_text())["models"]
    orig6 = json.loads((g.RES / "g6_shared_geometry.json").read_text())["reported"]
    res = {"note": "EXPLORATORY, after all locks; scripts/168_kind_baseline.py", "G1": {"models": {}}, "G6": {"models": {}}}
    for m in g.MODELS:
        k, layer, _ = g.select_layer(m, dev_sel)
        H, _ = g.hidden(m)
        p, _ = g.p_yes(m)
        r = analyse(np.sqrt(g.sq_dist(g.theta(p[test]))), Dt1, g.cos_dist(H[test, k]), kind1)
        r["layer"] = layer
        r["reproduces"] = bool(round(r["rho_answers"], 2) == round(orig1[m]["rho_answers"], 2) and round(r["rho_text"], 2) == round(orig1[m]["rho_text"], 2))
        res["G1"]["models"][m] = r
        Dh6 = g.cos_dist(np.load(G6 / "run" / m / "hidden_none.npz")["h"][:, k].astype(np.float64))
        Da6 = np.sqrt(g.sq_dist(g.theta(g4.p_of(np.load(G6 / "run" / m / "psi_none.npz")))))
        r6 = analyse(Da6, Dt6, Dh6, kind6)
        r6["reproduces"] = bool(round(r6["rho_answers"], 2) == round(orig6[m]["answers"], 2) and round(r6["rho_text"], 2) == round(orig6[m]["text"], 2))
        res["G6"]["models"][m] = r6
        print(m, json.dumps({x: {k_: v for k_, v in rr.items() if k_ != "per_kind"} for x, rr in (("G1", r), ("G6", r6))}), flush=True)
    for x in ("G1", "G6"):
        res[x].update(summary(res[x]["models"]))
        print(x, json.dumps({k_: v for k_, v in res[x].items() if k_ != "models"}), flush=True)
    (g.RES / "kind_baseline_exploratory.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
