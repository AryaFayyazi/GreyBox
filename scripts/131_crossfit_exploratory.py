#!/usr/bin/env python
"""EXPLORATORY (after the lock of G; Correction 1 of docs/theory_infogeom.md): the cross-fit estimator (Theorem 5')
against the locked debiased-subtraction estimator, with the same 8 samples per cell (cross-fit: 4 + 4).
  open models (confirm split, layer chosen on dev as in G1): rho with the hidden geometry, as a share of the exact rho
  Claude: negative-eigenvalue share of classical MDS (0 = Euclidean), rho with text, and with open models' hidden geometry
python scripts/131_crossfit_exploratory.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.geometry import crossfit_distances, mds, observable_distances, fisher_angles_from_counts  # noqa: E402

spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/129_infogeom.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def neg_share(D):
    _, ev = mds(D, 2)
    return float(-ev[ev < 0].sum() / np.abs(ev).sum())


def main():
    C = g.contexts()
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    test = np.where(C.split.values == "confirm")[0]
    rng = np.random.default_rng(1310)
    res = {"open": {}, "claude": {}}
    layer_k = {}
    for m in g.MODELS:
        k, layer, _ = g.select_layer(m, dev_sel)
        layer_k[m] = k
        p, _ = g.p_yes(m)
        H, _ = g.hidden(m)
        P, Dh = p[test], g.cos_dist(H[test, k])
        exact = g.rho(np.sqrt(g.sq_dist(g.theta(P))), Dh)
        kA, kB = rng.binomial(4, P), rng.binomial(4, P)
        n4 = np.full(P.shape, 4)
        Dsub = observable_distances(fisher_angles_from_counts(kA + kB, 8), 8)
        Dcf = crossfit_distances(kA, n4, kB, n4)
        res["open"][m] = {"rho_exact": round(exact, 4), "subtract_share": round(g.rho(Dsub, Dh) / exact, 4),
                          "crossfit_share": round(g.rho(Dcf, Dh) / exact, 4),
                          "neg_mds_subtract": round(neg_share(Dsub[:400, :400]), 4), "neg_mds_crossfit": round(neg_share(Dcf[:400, :400]), 4)}
        print(m, res["open"][m], flush=True)
    d, cnt = g.claude_counts()
    ids = [r["id"] for r in d["results"]]
    pos = C.reset_index().set_index("id").loc[ids, "index"].values
    r2 = np.random.default_rng(2290)                                   # the same split as G2
    lab = np.zeros(cnt.shape[:2] + (8,), int)
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            s_ = np.repeat([0, 1, 2], cnt[i, j])
            lab[i, j] = r2.permutation(np.pad(s_, (0, 8 - len(s_)), constant_values=2))
    A_, B_ = lab[:, :, :4], lab[:, :, 4:]
    Dcf = crossfit_distances((A_ == 0).sum(2), (A_ < 2).sum(2), (B_ == 0).sum(2), (B_ < 2).sum(2))
    k, n = cnt[:, :, 0], cnt[:, :, 0] + cnt[:, :, 1]
    Dsub = g.angle_dist(k, n)
    Dt = g.cos_dist(np.load(g.L2 / "minilm.npy").astype(np.float64)[pos])
    res["claude"] = {"neg_mds_subtract": round(neg_share(Dsub), 4), "neg_mds_crossfit": round(neg_share(Dcf), 4),
                     "rho_text_subtract": round(g.rho(Dsub, Dt), 4), "rho_text_crossfit": round(g.rho(Dcf, Dt), 4),
                     "rho_open_hidden_crossfit": {m: round(g.rho(Dcf, g.cos_dist(g.hidden(m)[0][pos, layer_k[m]])), 4) for m in g.MODELS}}
    X, ev = mds(Dcf, 2)
    np.savez(g.OUT / "claude_mds_crossfit.npz", X=X, ids=np.array(ids), source=C.source.values[pos], harmful=C.harmful.values[pos])
    print(json.dumps(res["claude"], indent=1))
    (g.RES / "crossfit_exploratory.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
