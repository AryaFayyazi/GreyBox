#!/usr/bin/env python
"""Exploratory (not preregistered): why did G83's 24 new-kind questions not raise agreement (E1 failed)?
On G83's 400 new-kind contexts, per open model: Spearman with the hidden geometry of the 24 new questions alone, the 88
alone, and 88+24; the share of the 24 answered near-deterministically (P(yes) < 0.05 or > 0.95 on >= 90% of contexts);
and agreement within each new kind (pairs inside one kind) for 88 vs 112. -> results/t1/g83_diagnosis_exploratory.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("g83d", ROOT / "scripts/282_g83_newkind_battery.py")
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)
g, g4, G6 = M.g, M.g4, M.G6


def main():
    C, ix = M.new_contexts()
    kinds = C.kind.values[ix]
    Cc = g.contexts()
    dev = np.where(Cc.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    out = {}
    for m in g.MODELS:
        run = G6.OUT / "run" / m
        H = g.cos_dist(np.load(run / "hidden_none.npz")["h"][:, g.select_layer(m, dev_sel)[0]].astype(np.float64))[np.ix_(ix, ix)]
        p88 = g4.p_of(np.load(run / "psi_none.npz", allow_pickle=True))[ix]
        p24 = g4.p_of(np.load(run / "psi_newkinds.npz", allow_pickle=True))
        t88, t24 = g.theta(p88), g.theta(p24)
        D = lambda T: np.sqrt(g.sq_dist(T))  # noqa: E731
        A88, A24, A112 = D(t88), D(t24), D(np.hstack([t88, t24]))
        det = float(np.mean(((p24 < 0.05) | (p24 > 0.95)).mean(0) >= 0.9))
        within = {}
        for k in sorted(set(kinds)):
            s = np.flatnonzero(kinds == k)
            q = lambda A, s=s: A[np.ix_(s, s)]  # noqa: E731
            within[k] = [round(g.rho(q(A88), q(H)), 4), round(g.rho(q(A112), q(H)), 4), round(g.rho(q(A24), q(H)), 4)]
        out[m] = {"rho_24": round(g.rho(A24, H), 4), "rho_88": round(g.rho(A88, H), 4), "rho_112": round(g.rho(A112, H), 4),
                  "share_24_near_deterministic": round(det, 3), "within_kind_88_112_24": within,
                  "mean_sq_dist_share_from_24": round(float((A112 ** 2 - A88 ** 2).mean() / (A112 ** 2).mean()), 3)}
    agg = {k: round(float(np.mean([v[k] for v in out.values()])), 4) for k in ("rho_24", "rho_88", "rho_112", "share_24_near_deterministic", "mean_sq_dist_share_from_24")}
    res = {"mean": agg, "per_model": out}
    (ROOT / "results/t1/g83_diagnosis_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(agg, indent=1))
    for m, v in out.items():
        print(m, v["rho_24"], v["rho_88"], v["rho_112"], v["share_24_near_deterministic"], v["within_kind_88_112_24"])


if __name__ == "__main__":
    main()
