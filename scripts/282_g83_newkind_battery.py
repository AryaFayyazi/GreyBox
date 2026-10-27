#!/usr/bin/env python
"""G83 (docs/prereg_G83_newkind_battery.md): on kinds of context the 88-question battery was not written for (G6's 400
fresh contexts of news, code, jailbreak_bench and injection), answer geometry barely beats text (G78 by kind: Claude +0.02,
open models' own exact answers +0.08; exploratory). If the limit is the battery's coverage, questions written for those
kinds (graybox.battery_newkinds, 24, written before any answer to them existed) should restore it. Per open model (G6's
eight): exact P(yes) on the 24 new questions (scripts/132's generation code, unchanged: G6's frame, no system message,
logprob tier), on the 400 new-kind contexts; hidden geometry from G6's stored states at G1's development-chosen layer.
  stage=gen model=M (GPU) -> outputs/g6/run/<M>/psi_newkinds.npz ; stage=analyze -> results/t1/g83_newkind_battery.json
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
B_N, BOOT_SEED = 1000, 8300


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G6 = _mod("g6_g83", "137_g6_shared_geometry.py")
g, g4 = G6.g, G6.g4


def new_contexts():
    C = G6.ctx_frame()
    return C, np.flatnonzero(C.new_kind.values.astype(bool))


def stage_gen(cfg):
    import graybox.battery_v2 as bv2
    from graybox.battery_newkinds import BATTERY_NEWKINDS
    C, ix = new_contexts()
    bv2.BATTERY_V2 = BATTERY_NEWKINDS                      # scripts/132's stage_gen reads this name at call time
    g4.CONDITIONS = {"newkinds": None}                     # writes psi_newkinds.npz; G6's psi_none.npz is untouched
    g4.contexts = lambda split, per_source: C.iloc[ix].reset_index(drop=True)
    g4.OUT = G6.OUT
    g4.stage_gen(OmegaConf.merge(g4.D, {"model": cfg.model, "split": "run", "replies": 1, "reply_tokens": 1}))


def stage_analyze():
    C, ix = new_contexts()
    SE = Path("/data/models/graybox_ws/outputs/strong_embed")
    sub = lambda A: A[np.ix_(ix, ix)]  # noqa: E731
    base_in = {"input_minilm": sub(g.cos_dist(np.load(G6.OUT / "run/contexts_minilm.npy").astype(np.float64))),
               "input_qemb06": sub(g.cos_dist(np.load(SE / "fresh_input_qemb06.npy").astype(np.float64))),
               "input_mpnet": sub(g.cos_dist(np.load(SE / "fresh_input_mpnet.npy").astype(np.float64)))}
    Cc = g.contexts()
    dev = np.where(Cc.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    D = {}
    for m in g.MODELS:
        run = G6.OUT / "run" / m
        H = sub(g.cos_dist(np.load(run / "hidden_none.npz")["h"][:, g.select_layer(m, dev_sel)[0]].astype(np.float64)))
        th88 = g.theta(g4.p_of(np.load(run / "psi_none.npz", allow_pickle=True)))[ix]
        z = np.load(run / "psi_newkinds.npz", allow_pickle=True)
        assert len(z["ids"]) == len(ix) and (z["ids"] == C.id.values[ix]).all()
        th24 = g.theta(g4.p_of(z))
        dist = lambda T: np.sqrt(g.sq_dist(T))  # noqa: E731
        base = {**base_in, "own_replies_qemb06": sub(g.cos_dist(np.load(SE / f"fresh_replies_{m}_qemb06.npy").astype(np.float64))),
                "own_replies_mpnet": sub(g.cos_dist(np.load(SE / f"fresh_replies_{m}_mpnet.npy").astype(np.float64)))}
        D[m] = {"H": H, "A88": dist(th88), "A112": dist(np.hstack([th88, th24])), "base": base}
    n = len(ix)

    def stats(s):
        q = lambda A: A[np.ix_(s, s)]  # noqa: E731
        e1, e2, per = [], [], {}
        for m, d in D.items():
            r88, r112 = g.rho(q(d["A88"]), q(d["H"])), g.rho(q(d["A112"]), q(d["H"]))
            best = max(g.rho(q(B), q(d["H"])) for B in d["base"].values())
            e1.append(r112 - r88); e2.append(r112 - best)
            per[m] = (r88, r112, best)
        return float(np.mean(e1)), float(np.mean(e2)), per
    a1, a2, per = stats(np.arange(n))
    rng = np.random.default_rng(BOOT_SEED)
    B = [stats(np.unique(rng.integers(0, n, n)))[:2] for _ in range(B_N)]
    ci = lambda k: [round(float(np.percentile([b[k] for b in B], 2.5)), 4), round(float(np.percentile([b[k] for b in B], 97.5)), 4)]  # noqa: E731
    out = {"contexts": n, "per_model": {m: {"rho_88": round(v[0], 4), "rho_112": round(v[1], 4), "strongest_text": round(v[2], 4)} for m, v in per.items()},
           "E1_mean_rho112_minus_rho88": [round(a1, 4)] + ci(0), "E2_mean_rho112_minus_strongest_text": [round(a2, 4)] + ci(1)}
    out["endpoints"] = {"E1_lower_gt_0": out["E1_mean_rho112_minus_rho88"][1] > 0, "E2_lower_gt_0": out["E2_mean_rho112_minus_strongest_text"][1] > 0}
    (ROOT / "results/t1/g83_newkind_battery.json").write_text(__import__("json").dumps(out, indent=1))
    print(__import__("json").dumps(out, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    stage_gen(OmegaConf.create({"model": kw["model"]})) if kw.get("stage") == "gen" else stage_analyze()
