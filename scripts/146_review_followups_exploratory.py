#!/usr/bin/env python
"""EXPLORATORY follow-ups to the second review (no new model outputs; existing data only):
  claude_reference  open models' 24-question answers (8 samples drawn from their own probabilities, seed 2290) vs own and
                    other open models' hidden geometry, on the 192 Claude contexts: the range Claude is compared with
  g6_new_kinds      G6's own-model excess (two-way residual) on the 400 contexts of the four new kinds only, bootstrap
  g6_per_model      G6's per-model diagonal residual, with bootstrap intervals (heterogeneity behind the pooled null)
python scripts/146_review_followups_exploratory.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = _mod("g", "129_infogeom.py")
g4 = _mod("g4", "132_g4_shift.py")
OUT6 = Path("/data/models/graybox_ws/outputs/g6")


def claude_reference():
    d = json.loads(Path("/data/models/graybox_ws/outputs/l3/claude_haiku55_run.json").read_text())
    ids, items = [r["id"] for r in d["results"]], d["items"]
    C = g.contexts()
    pos = C.reset_index().set_index("id").loc[ids, "index"].values
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    rng = np.random.default_rng(2290)
    Dh, Ds = {}, {}
    for m in g.MODELS:
        k, _, _ = g.select_layer(m, dev_sel)
        H, _ = g.hidden(m)
        Dh[m] = g.cos_dist(H[pos, k])
        p, its = g.p_yes(m)
        P = p[pos][:, [its.index(i) for i in items]]
        Ds[m] = g.angle_dist(rng.binomial(8, P), np.full(P.shape, 8))
    own = [g.rho(Ds[a], Dh[a]) for a in g.MODELS]
    other = [g.rho(Ds[a], Dh[b]) for a in g.MODELS for b in g.MODELS if a != b]
    return {"own": [round(min(own), 4), round(float(np.median(own)), 4), round(max(own), 4)],
            "other": [round(min(other), 4), round(float(np.median(other)), 4), round(max(other), 4)]}


def g6_parts(n_boot=200):
    C = pd.read_parquet(OUT6 / "contexts.parquet").reset_index(drop=True)
    new = np.where(C.new_kind.values.astype(bool))[0]
    Call = g.contexts()
    dev = np.where(Call.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    M = g.MODELS
    Dh, Da = {}, {}
    for m in M:
        k, _, _ = g.select_layer(m, dev_sel)
        Dh[m] = g.cos_dist(np.load(OUT6 / "run" / m / "hidden_none.npz")["h"][:, k].astype(np.float64))
        Da[m] = np.sqrt(g.sq_dist(g.theta(g4.p_of(np.load(OUT6 / "run" / m / "psi_none.npz")))))
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731

    def resid(s):
        R = np.array([[g.rho(sub(Da[a], s), sub(Dh[b], s)) for b in M] for a in M])
        return np.diag(R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean())
    rng = np.random.default_rng(1460)
    out = {}
    for name, idx in (("new_kinds", new), ("all", np.arange(len(C)))):
        b = np.array([resid(idx[np.unique(rng.integers(0, len(idx), len(idx)))]) for _ in range(n_boot)])
        point = resid(idx)
        out[name] = {"pooled": [round(float(point.mean()), 4), round(float(np.percentile(b.mean(1), 2.5)), 4), round(float(np.percentile(b.mean(1), 97.5)), 4)],
                     "per_model": {m: [round(float(point[i]), 4), round(float(np.percentile(b[:, i], 2.5)), 4), round(float(np.percentile(b[:, i], 97.5)), 4)] for i, m in enumerate(M)}}
        print(name, json.dumps(out[name]), flush=True)
    return out


def main():
    res = {"claude_reference": claude_reference()}
    print(json.dumps(res["claude_reference"]), flush=True)
    res["g6"] = g6_parts()
    (g.RES / "review_followups_exploratory.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
