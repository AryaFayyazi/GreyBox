#!/usr/bin/env python
"""G71 (docs/prereg_G71_fair_whitebox_and_slice.md). Three analyses, endpoints fixed before any of them is computed.
  B  the fair white box. G70 compared the API geometry with a probe on CAPE's stored decision state, which sits AFTER
     the agent's own hidden reasoning for the turn (scripts/96 decision_contexts; scripts/60). A closed API forecasts from
     the start of the turn. On G68's Gemma-4 units that also have T2's turn-start states (217 units), compare the
     geometric forecast (G70's, unchanged) with T1's FROZEN turn-start probe (scripts/25: logistic regression on the
     turn-start residual, layer 18, fit on the separate development set). Reported on the same units: the probe after
     reasoning (CAPE's gate I_dec, scripts/60) and 16 checker-labelled resamples at turn start (R16_ts).
  D  the slice the answers estimate (Theorem 1(c): a battery estimates the hidden geometry in the metric it pulls back,
     a slice of rank <= its size, not the full state). On G15's held-out calls of the 4 open agents (G36's tables), a
     ridge map from the decision state (layer 3, PCA(64) fit on all of that agent's stored states, no labels) to the 8
     answer angles, cross-fit by scenario (5 folds), gives each call's predicted angles; compare the answer geometry
     (Theorem 5's debiased distances) with (i) the predicted-angle geometry (the pulled-back slice), (ii) the full-state
     cosine geometry (G36's), (iii) 20 random rank-8 linear maps of the same PCA(64) state (a dimension-matched null).
     No label is used anywhere in D.
  A  G43's 32 situation questions as a monitor at the call: G56's statistic (distance of a call's 32 angles from the
     agent's median, no label, no question sign), within template; run when G43's answers for all 4 agents exist.
  stage=bd -> results/t1/g71_fair_whitebox_and_slice.json ; stage=a -> results/t1/g71a_situation_monitor.json
"""
import importlib
import importlib.util
import json
import pickle
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
B_N, BOOT_SEED, NI_B = 2000, 7100, -0.05
I = Path("/data/models/graybox_ws/outputs/interrogate")


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def ci(v):
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]


def part_b():
    from graybox.stats import within_auroc
    G70 = _mod("g70", "scripts/262_g70_open_geometry_vs_whitebox.py")
    T2 = _mod("t2", "scripts/60_decision_point.py")
    a25 = importlib.import_module("25_t1_confirm")
    G68 = G70.G68
    U = G68.units("gemma4")
    rec = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])): json.loads(x) for x in open(G68.out_file("gemma4"))}
    U = U[[(e, int(v)) in rec for e, v in zip(U.episode_id, U.event_id)]].reset_index(drop=True)
    R = [rec[(e, int(v))] for e, v in zip(U.episode_id, U.event_id)]
    geo_all = G70.geometric(list(rec.values()))                    # centre over all of the agent's drawn calls (G70)
    gmap = {k: g for k, g in zip(rec.keys(), geo_all)}
    geo = np.array([gmap[(e, int(v))] for e, v in zip(U.episode_id, U.event_id)])
    V, _ = T2.units()
    with open(a25.MOD / "frozen_monitors.pkl", "rb") as f:
        M = pickle.load(f)
    wb = M["wb"].predict_proba(a25.hidden(I / "gemma4_fresh", M["wb_layer"]))[:, 1]
    Hd = np.load(I / "gemma4_fresh_decision_hidden.npz")
    Rts = pd.read_parquet(I / "gemma4_fresh_rollouts_K16.parquet")
    k = np.array([np.sum(json.loads(s)) for s in Rts.R_unsafe]); n = np.array([len(json.loads(s)) for s in Rts.R_unsafe])
    r16 = (k + 0.5) / (n + 1)
    key = {(e, int(v)): i for i, (e, v) in enumerate(zip(V.episode_id, V.event_id))}
    idx = np.array([key.get((e, int(v)), -1) for e, v in zip(U.episode_id, U.event_id)])
    m = idx >= 0
    D = {"y": U.y.values[m], "st": U.stratum.values[m], "cl": U.scenario_id.values[m], "geo": geo[m],
         "own": np.array([G68.G46.forecast(x) for x, ok in zip(R, m) if ok]), "wb_ts": wb[idx[m]], "i_dec": Hd["I_dec"][idx[m]],
         "r16_ts": r16[idx[m]], "report": U.report.values[m]}
    W = lambda s, i: within_auroc(D["y"][i], D[s][i], D["st"][i])  # noqa: E731
    a = np.arange(len(D["y"]))
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(D["cl"])
    bs = [np.concatenate([np.flatnonzero(D["cl"] == u) for u in rng.choice(us, len(us))]) for _ in range(B_N)]
    out = {"units": int(len(a)), "violations": int(D["y"].sum()),
           "auroc": {s: round(float(W(s, a)), 4) for s in ("geo", "own", "wb_ts", "i_dec", "r16_ts", "report")}}
    for name, s1, s2 in (("B1_geo_minus_wb_ts", "geo", "wb_ts"), ("geo_minus_i_dec", "geo", "i_dec"),
                         ("wb_ts_minus_i_dec", "wb_ts", "i_dec"), ("own_minus_wb_ts", "own", "wb_ts"), ("r16_ts_minus_wb_ts", "r16_ts", "wb_ts")):
        out[name] = [round(float(W(s1, a) - W(s2, a)), 4)] + ci([W(s1, i) - W(s2, i) for i in bs])
    return out


def part_d():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    G36 = _mod("g36", "scripts/209_g36_decision_triangulation.py")
    M171 = G36._mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    res, rng = {}, np.random.default_rng(BOOT_SEED + 1)
    for a in ("gemma4", "gptoss20b", "qwen38", "museglimmer"):
        C = G36.calls(a)
        cnt = np.array([json.loads(c) if isinstance(c, str) else c for c in C.counts], float)
        A = G36.answer_dist(cnt)
        k_, m_ = cnt[..., 0], cnt[..., 1]
        th = 2 * np.arcsin(np.sqrt((k_ + 3 / 8) / (np.maximum(m_, 1) + 3 / 4)))
        U, H = M171.states(a)
        key = {kk: i for i, kk in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
        Hc = H[[key[(e, int(v))] for e, v in zip(C.episode_id, C.event_id)], G36.DLAYER].astype(np.float64)
        pca = PCA(64, random_state=0).fit(H[U.ok.values, G36.DLAYER].astype(np.float64))
        Z = pca.transform(Hc)
        pred = np.zeros_like(th)
        for tr, te in GroupKFold(5).split(Z, th, C.scenario_id.values):
            pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Z[tr], th[tr]).predict(Z[te])
        dist = lambda X: np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))  # noqa: E731
        P, Hown = dist(pred), G36.cosd(Hc)
        nulls = [dist(Z @ rng.standard_normal((64, 8))) for _ in range(20)]
        n = len(C)

        def stats(i):
            ii = np.ix_(i, i)
            return (G36.srho(A[ii], P[ii]), G36.srho(A[ii], Hown[ii]), float(np.mean([G36.srho(A[ii], N[ii]) for N in nulls])))
        full = stats(np.arange(n))
        B = [stats(np.sort(rng.choice(n, n, replace=True))) for _ in range(200)]
        res[a] = {"calls": n, "pulled_back_slice": round(full[0], 4), "full_state_cosine": round(full[1], 4),
                  "random_rank8_null": round(full[2], 4),
                  "slice_minus_full": [round(full[0] - full[1], 4)] + ci([b[0] - b[1] for b in B]),
                  "slice_minus_null": [round(full[0] - full[2], 4)] + ci([b[0] - b[2] for b in B])}
    res["D1_agents_slice_minus_full_lower_gt_0"] = int(sum(res[a]["slice_minus_full"][1] > 0 for a in res if isinstance(res[a], dict)))
    res["D2_agents_slice_minus_null_lower_gt_0"] = int(sum(res[a]["slice_minus_null"][1] > 0 for a in res if isinstance(res[a], dict)))
    return res


def stage_bd():
    b, d = part_b(), part_d()
    out = {"B": b, "D": d, "endpoints": {"B1_geo_minus_wb_ts_lower_gt_-0.05": b["B1_geo_minus_wb_ts"][1] > NI_B,
                                          "D1_slice_beats_full_state_on_>=3_of_4": d["D1_agents_slice_minus_full_lower_gt_0"] >= 3,
                                          "D2_slice_beats_random_rank8_on_>=3_of_4": d["D2_agents_slice_minus_null_lower_gt_0"] >= 3}}
    (ROOT / "results/t1/g71_fair_whitebox_and_slice.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def stage_a():
    from graybox.stats import within_auroc
    G43 = _mod("g43", "scripts/219_g43_situation_battery.py")
    res, per = {}, {}
    data = {}
    for a in ("gemma4", "gptoss20b", "qwen38", "museglimmer"):
        C = pd.read_parquet(G43.OUT / f"{a}.parquet")
        T = G43.calls(a)
        assert (T.episode_id.values == C.episode_id.values).all()
        def ang(col):
            c = np.array([json.loads(x) if isinstance(x, str) else x for x in C[col]], float)
            return 2 * np.arcsin(np.sqrt((c[..., 0] + 3 / 8) / (np.maximum(c[..., 1], 1) + 3 / 4)))
        t32, t8 = ang("counts32"), ang("counts")
        g32 = np.linalg.norm(t32 - np.median(t32, 0), axis=1)
        g8 = np.linalg.norm(t8 - np.median(t8, 0), axis=1)
        g40 = np.linalg.norm(np.hstack([t32, t8]) - np.median(np.hstack([t32, t8]), 0), axis=1)
        data[a] = {"y": C.y.values.astype(int), "st": T.template.values, "cl": T.scenario_id.values, "g32": g32, "g8": g8, "g40": g40}
    rng = np.random.default_rng(BOOT_SEED + 2)
    bs = []
    for a, D in data.items():
        i = np.arange(len(D["y"]))
        res[a] = {s: round(float(within_auroc(D["y"], D[s], D["st"])), 4) for s in ("g32", "g8", "g40")}
    for _ in range(B_N):
        v = []
        for D in data.values():
            us = np.unique(D["cl"])
            i = np.concatenate([np.flatnonzero(D["cl"] == u) for u in rng.choice(us, len(us))])
            v.append(within_auroc(D["y"][i], D["g32"][i], D["st"][i]))
        bs.append(np.nanmean(v))
    p = float(np.mean([res[a]["g32"] for a in data]))
    out = {"agents": res, "A1_pooled_g32": [round(p, 4)] + ci(bs), "endpoints": {"A1_lower_gt_0.5": ci(bs)[0] > 0.5}}
    (ROOT / "results/t1/g71a_situation_monitor.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    stage_a() if kw.get("stage") == "a" else stage_bd()
