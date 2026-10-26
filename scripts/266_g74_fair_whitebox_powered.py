#!/usr/bin/env python
"""G74 (docs/prereg_G74_fair_whitebox_powered.md): G71's fair white-box comparison, powered. On T2's 12,130 fresh Gemma-4
units minus G68's: every violating unit, plus 600 safe units drawn from the same templates (numpy seed 7400), the agent's own 8 sampled next actions from
the exact turn-start context, each distinct drawn call read by its own 8 at-call answers (scripts/260 stage_draws,
unchanged: G46's code with the agent in its own seat); the geometric forecast (G57/G70's statistic, label-free, sign-free)
and the signed own forecast, against T1's FROZEN turn-start white-box probe (scripts/25, layer 18, fit on the separate
development set) on the same units. Reported: CAPE's probe after reasoning (I_dec) and 16 checker-labelled resamples.
  stage=draws (GPU) -> outputs/tomography/g74_gemma4.jsonl ; stage=analyze -> results/t1/g74_fair_whitebox_powered.json
"""
import importlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
N_EACH, SEED, B_N, BOOT_SEED, NI = 600, 7400, 2000, 7401, -0.05
I = Path("/data/models/graybox_ws/outputs/interrogate")


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G68 = _mod("g68", "scripts/260_g68_open_says_does.py")
T2 = _mod("t2", "scripts/60_decision_point.py")
OUTF = G68.G27.OUT / "g74_gemma4.jsonl"


def units():
    V, _ = T2.units()
    V = V.reset_index(drop=True).assign(row=np.arange(len(V)))
    used = set(zip(G68.units("gemma4").episode_id, G68.units("gemma4").event_id.astype(int)))
    V = V[[(e, int(v)) not in used for e, v in zip(V.episode_id, V.event_id)]]
    rng = np.random.default_rng(SEED)
    viol = V.index[V.y.values == 1]
    safe = V.index[(V.y.values == 0) & V.template.isin(set(V.template[viol])).values]
    pick = np.concatenate([viol, rng.choice(safe, N_EACH, replace=False)])
    U = V.loc[np.sort(pick)].reset_index(drop=True)
    return U.assign(t=U.event_id.astype(int), stratum=U.template.astype(str), y=U.y.astype(int))


def stage_draws():
    G68.units = lambda actor: units()                       # scripts/260's draw stage on these units, unchanged
    G68.stage_draws("gemma4", None, OUTF)


def stage_analyze():
    from graybox.stats import within_auroc
    G70 = _mod("g70", "scripts/262_g70_open_geometry_vs_whitebox.py")
    a25 = importlib.import_module("25_t1_confirm")
    U = units()
    rec = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])): json.loads(x) for x in open(OUTF)}
    m = np.array([(e, int(v)) in rec for e, v in zip(U.episode_id, U.event_id)])
    U = U[m].reset_index(drop=True)
    R = [rec[(e, int(v))] for e, v in zip(U.episode_id, U.event_id)]
    with open(a25.MOD / "frozen_monitors.pkl", "rb") as f:
        M = pickle.load(f)
    wb_all = M["wb"].predict_proba(a25.hidden(I / "gemma4_fresh", M["wb_layer"]))[:, 1]
    Hd = np.load(I / "gemma4_fresh_decision_hidden.npz")
    Rts = pd.read_parquet(I / "gemma4_fresh_rollouts_K16.parquet")
    k = np.array([np.sum(json.loads(s)) for s in Rts.R_unsafe]); n = np.array([len(json.loads(s)) for s in Rts.R_unsafe])
    D = {"y": U.y.values, "st": U.stratum.values, "cl": U.scenario_id.values, "geo": G70.geometric(R),
         "own": np.array([G68.G46.forecast(x) for x in R]), "wb_ts": wb_all[U.row.values], "i_dec": Hd["I_dec"][U.row.values],
         "r16_ts": ((k + 0.5) / (n + 1))[U.row.values]}
    W = lambda s, i: within_auroc(D["y"][i], D[s][i], D["st"][i])  # noqa: E731
    a = np.arange(len(D["y"]))
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(D["cl"])
    bs = [np.concatenate([np.flatnonzero(D["cl"] == u) for u in rng.choice(us, len(us))]) for _ in range(B_N)]
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    out = {"units": int(len(a)), "violations": int(D["y"].sum()), "auroc": {s: round(float(W(s, a)), 4) for s in ("geo", "own", "wb_ts", "i_dec", "r16_ts")}}
    for name, s1, s2 in (("F1_geo_minus_wb_ts", "geo", "wb_ts"), ("F2_own_minus_wb_ts", "own", "wb_ts"), ("wb_ts_minus_i_dec", "wb_ts", "i_dec"),
                         ("geo_minus_i_dec", "geo", "i_dec"), ("r16_ts_minus_wb_ts", "r16_ts", "wb_ts")):
        out[name] = [round(float(W(s1, a) - W(s2, a)), 4)] + ci([W(s1, i) - W(s2, i) for i in bs])
    out["endpoints"] = {"F1_geo_noninferior_to_wb_ts": out["F1_geo_minus_wb_ts"][1] > NI,
                        "F2_own_noninferior_to_wb_ts": out["F2_own_minus_wb_ts"][1] > NI}
    (ROOT / "results/t1/g74_fair_whitebox_powered.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    stage_draws() if kw.get("stage") == "draws" else stage_analyze()
