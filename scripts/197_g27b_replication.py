#!/usr/bin/env python
"""G27b: G27 (state tomography and forecasting) and G28 (the estimate against the true hidden state) replicated on the
three other open agents: gpt-oss-20b, Qwen3.8-27B and Muse-Glimmer-30B (docs/prereg_G27b_replication.md).
G27's code is used unchanged (scripts/194: QUESTIONS, FRAME, prefix, one, angles, sysid, kalman); per agent: its own
vLLM server, its held-out episodes, min(150, available) violating and as many non-violating episodes (seed 2700), every
tool-call step up to and including the first violating call (at most 12).
  stage=battery actor=<gptoss20b|qwen38|museglimmer> [vllm_python=...]   (GPU) -> outputs/tomography/<actor>_steps.parquet
  stage=analyze                                                            (CPU) -> results/t1/g27b_replication.json
"""
import asyncio
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
_s = importlib.util.spec_from_file_location("g27", ROOT / "scripts/194_g27_state_tomography.py")
G27 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(G27)
AGENTS = ["gptoss20b", "qwen38", "museglimmer"]
PORT = {"gptoss20b": 8197, "qwen38": 8198, "museglimmer": 8199}
LAYER, B_N = 3, 2000


def steps(actor):
    sa = importlib.import_module("150_self_audit")
    U = sa.units(f"{actor}_heldout")
    eps = U.groupby("episode_id").agg(set=("set", "first"), bench=("bench", "first"), scenario_id=("scenario_id", "first"),
                                     template=("template", "first")).reset_index()
    rows = []
    for e in eps.itertuples():
        evs = [x for x in sa.rollouts(e.set).loc[e.episode_id].events if x["is_tool_call"] and x["tool"]]
        out, viol = [], 0
        for t, x in enumerate(evs[:G27.MAX_STEPS]):
            y = int(bool(x.get("unsafe_action")))
            out.append({"episode_id": e.episode_id, "set": e.set, "bench": e.bench, "scenario_id": e.scenario_id, "template": e.template,
                        "t": t, "event_id": int(x["event_id"]), "tool": x["tool"], "y": y})
            if y:
                viol = 1
                break
        for o in out:
            o["episode_violates"] = viol
        rows += out
    S = pd.DataFrame(rows)
    ev = S.groupby("episode_id").episode_violates.first()
    n = min(G27.N_EACH, int((ev == 1).sum()))
    rng = np.random.default_rng(G27.SEED)
    pick = np.concatenate([rng.choice(ev.index[ev.values == 1], n, replace=False), rng.choice(ev.index[ev.values == 0], n, replace=False)])
    return S[S.episode_id.isin(set(pick))].sort_values(["episode_id", "t"]).reset_index(drop=True)


def stage_battery(actor, vllm_python=None):
    sa = importlib.import_module("150_self_audit")
    a = sa.srv.ACTORS[actor]
    S = steps(actor)
    print(actor, len(S), "steps,", S.episode_id.nunique(), "episodes,", int(S.y.sum()), "violating calls", flush=True)
    bases = [G27.prefix(sa, a, r) for r in S.itertuples()]
    if vllm_python:
        sys.executable = vllm_python
    proc = sa.srv.start_server(a, PORT[actor])
    try:
        async def go():
            sem = asyncio.Semaphore(G27.CONC)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT[actor]}") as client:
                out, t0 = [], time.time()
                for i in range(0, len(bases), 64):
                    out += await asyncio.gather(*[G27.one(client, a, b, sem) for b in bases[i:i + 64]])
                    print(f"{len(out)}/{len(bases)} {time.time() - t0:.0f}s", flush=True)
                return out
        out = asyncio.run(go())
    finally:
        proc.terminate()
    S.assign(counts=[json.dumps(o) for o in out]).to_parquet(G27.OUT / f"{actor}_steps.parquet")


def estimates(S):
    """G27's cross-fit (5 folds by scenario, seed 2701): forecast, filtered, prior, raw angles, NIS, IV/theory noise."""
    C = np.array([json.loads(c) for c in S.counts], float)
    Y, R = G27.angles(C)
    ep, scen = S.episode_id.values, S.scenario_id.values
    us = np.unique(scen)
    rng = np.random.default_rng(2701)
    fold = dict(zip(us, rng.permutation(len(us)) % 5))
    fc, fl, pr, nis = np.zeros_like(Y), np.zeros_like(Y), np.zeros_like(Y), np.zeros(len(S))
    ratios = []
    for f in range(5):
        tr = [e for e in np.unique(ep) if fold[scen[ep == e][0]] != f]
        M = G27.sysid([(Y[ep == e], R[ep == e]) for e in tr])
        ratios.append(M["r_iv"] / M["r_theory"])
        for e in np.unique(ep):
            i = np.flatnonzero(ep == e)
            if fold[scen[i[0]]] == f:
                a, b, c = G27.kalman(Y[i], R[i], M)
                fc[i], fl[i], nis[i] = a, b, c
                pr[i] = G27.kalman(Y[i], np.full_like(R[i], 1e9), M)[0]
    return Y, fc, fl, pr, nis, np.mean(ratios, 0)


def stage_analyze():
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    from graybox.stats import caught_at_budget, within_auroc
    s171 = importlib.util.spec_from_file_location("m171", ROOT / "scripts/171_decision_greybox_LOCKED.py")
    M171 = importlib.util.module_from_spec(s171)
    s171.loader.exec_module(M171)
    sg = np.array([s for _, s in G27.QUESTIONS])
    res, per = {}, {}
    rng = np.random.default_rng(2790)
    for actor in AGENTS:
        f = G27.OUT / f"{actor}_steps.parquet"
        if not f.exists():
            res[actor] = "missing"
            continue
        S = pd.read_parquet(f)
        Y, fc, fl, pr, nis, ratio = estimates(S)
        y = S.y.values.astype(int)
        strata = np.array([f"{tm}|{tl}" for tm, tl in zip(S.template, S.tool)])
        scen = S.scenario_id.values
        sub = np.flatnonzero(S.t.values > 0)
        s_f, s_k, s_r, s_p = fc @ sg, fl @ sg, Y @ sg, pr @ sg
        W = lambda s, i: within_auroc(y[i], s[i], strata[i])  # noqa: E731,B023
        per[actor] = (y, strata, scen, sub, s_f, s_k, s_r, s_p)
        # G28 on this agent: geometry of the estimate vs the true decision state, where it is stored
        U, H = M171.states(actor)
        key = {(e, int(v)): i for i, (e, v) in enumerate(zip(U.episode_id, U.event_id))}
        idx = np.array([key.get((e, int(v)), -1) for e, v in zip(S.episode_id, S.event_id)])
        ok = (idx >= 0) & np.array([bool(U.ok.values[j]) if j >= 0 else False for j in idx])
        truth = {}
        if ok.sum() > 20:
            Ht = H[idx[ok], LAYER].astype(np.float32)
            dt = pdist(Ht)
            truth = {k: round(float(spearmanr(pdist(v[ok]), dt).correlation), 4) for k, v in (("filtered", fl), ("raw", Y), ("forecast", fc))}
            truth["steps_with_true_state"] = int(ok.sum())
        res[actor] = {"steps": int(len(S)), "episodes": int(S.episode_id.nunique()), "violating_calls": int(y.sum()),
                      "within_stratum": {"forecast_after_first": round(float(W(s_f, sub)), 4), "prior_after_first": round(float(W(s_p, sub)), 4),
                                         "filtered": round(float(W(s_k, np.arange(len(y)))), 4), "raw": round(float(W(s_r, np.arange(len(y)))), 4)},
                      "nis_mean_over_K": round(float(nis.mean() / Y.shape[1]), 4), "iv_noise_over_theory": [round(float(x), 3) for x in ratio],
                      "caught_at_5pct_steps": {k: round(float(caught_at_budget(y, s, 0.05)), 4) for k, s in (("forecast", s_f), ("filtered", s_k), ("raw", s_r), ("prior", s_p))},
                      "G28_rsa_with_true_state": truth}
    # pooled endpoints over the agents present: mean of per-agent differences; cluster bootstrap over scenarios within agent
    def boot(fn):
        pt = np.mean([fn(v, None) for v in per.values()])
        bs = []
        for _ in range(B_N):
            vals = []
            for v in per.values():
                scen = v[2]
                us = np.unique(scen)
                idx = np.concatenate([np.flatnonzero(scen == u) for u in rng.choice(us, len(us))])
                vals.append(fn(v, idx))
            bs.append(np.nanmean(vals))
        return [round(float(pt), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]

    def k1(v, idx):
        y, strata, scen, sub, s_f, s_k, s_r, s_p = v
        i = sub if idx is None else idx[np.isin(idx, sub)]           # resampled calls after the first step
        return within_auroc(y[i], s_f[i], strata[i]) - within_auroc(y[i], s_p[i], strata[i])

    def k2(v, idx):
        y, strata, scen, sub, s_f, s_k, s_r, s_p = v
        i = np.arange(len(y)) if idx is None else idx
        return within_auroc(y[i], s_k[i], strata[i]) - within_auroc(y[i], s_r[i], strata[i])
    if per:
        res["K1_pooled_forecast_minus_prior_after_first"] = boot(k1)
        res["K2_pooled_filtered_minus_raw"] = boot(k2)
        res["endpoints"] = {"K1_pooled_lower_gt_0": res["K1_pooled_forecast_minus_prior_after_first"][1] > 0}
    (ROOT / "results/t1/g27b_replication.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kv = dict(x.split("=", 1) for x in sys.argv[1:])
    if kv.get("stage", "battery") == "battery":
        stage_battery(kv["actor"], kv.get("vllm_python"))
    else:
        stage_analyze()
