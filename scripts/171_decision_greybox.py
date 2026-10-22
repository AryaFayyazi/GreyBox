#!/usr/bin/env python
"""The grey box at the agent's decision point, on several agents (G14 runs this; development stages are exploratory).

Estimator A, the agent's own answers: 8 yes/no questions about its proposed call (web/build_observable_ref.py's
QUESTIONS and FRAME, unchanged), n answers each, through the agent's own API (vLLM, its own parsers); theta = arcsine
angle of the Anscombe estimate. Truth: the agent's stored decision state (interrogate/ for Gemma-4, t17/ for the others).

  stage=battery actor=<a> split=<dev|heldout|fresh> n_viol=300 n_safe=300   (GPU) -> outputs/decision_greybox/<a>_<split>.parquet
  stage=analyze actor=<a> mode=<dev|heldout> layer=3                          (CPU)
analyze, for one agent:
  PCA(32) of the true decision states, fit on the agent's development units;
  probe: logistic regression on the true states (PCA space) of ALL development units and their labels;
  map:   ridge from theta (8 numbers) to the PCA state, fit on the battery's development units (no labels);
  mode=dev:     map cross-fit by scenario (5 folds) within the development battery; probe excludes nothing (exploratory);
  mode=heldout: map and probe fit on development data only, scored on the held-out battery units.
  Scores: within-template and pooled AUROC of the probe on (i) the true state (white box), (ii) the state recreated from
  the answers; (iii) the "complies?" answer alone; R2 of the recreation; scenario-cluster bootstrap (2,000, seed 1740)
  of (ii) − (i).
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
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

sa = importlib.import_module("150_self_audit")
_spec = importlib.util.spec_from_file_location("obsref", ROOT / "web/build_observable_ref.py")
OR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OR)
OUT = Path("/data/models/graybox_ws/outputs/decision_greybox")
PILOT = {("consequence.cons_service_restart.0955.p3.unsafe.s50", 0), ("consequence.cons_service_restart.0955.p3.unsafe.s50", 1),
         ("consequence.cons_service_restart.0955.p3.unsafe.s50", 2), ("reversibility.rev_config_overwrite.0548.p0.unsafe.s50", 2),
         ("third_party.tp_billing_correction.0373.p1.unsafe.s50", 5)}   # the Claude pilot (answers seen): excluded from G14
CFG = OmegaConf.create({"stage": "analyze", "actor": "gptoss20b", "split": "dev", "mode": "dev", "n_viol": 300, "n_safe": 300, "n": 4,
                        "max_tokens": 4096, "concurrency": 64, "port": 8171, "seed": 1710, "layer": 3, "n_boot": 2000, "vllm_python": None, "dry": False})


def unit_name(actor, split):
    return "gemma4_clean" if (actor, split) == ("gemma4", "dev") else "gemma4_fresh" if actor == "gemma4" else f"{actor}_{split}"


def sample(U, n_viol, n_safe, seed):
    rng = np.random.default_rng(seed)
    v, s = np.where(U.y.values == 1)[0], np.where(U.y.values == 0)[0]
    pick = np.concatenate([rng.choice(v, min(n_viol, len(v)), replace=False), rng.choice(s, min(n_safe, len(s)), replace=False)])
    return U.iloc[np.sort(pick)].reset_index(drop=True)


def stage_battery(cfg):
    a = sa.srv.ACTORS[cfg.actor]
    U = sample(sa.units(unit_name(cfg.actor, cfg.split)), cfg.n_viol, cfg.n_safe, cfg.seed)
    if (cfg.actor, cfg.split) == ("gemma4", "dev"):                     # the observable-reference sample already exists
        O = pd.read_parquet("/data/models/graybox_ws/outputs/observable_ref/gemma4.parquet")
        OUT.mkdir(parents=True, exist_ok=True)
        O.to_parquet(OUT / "gemma4_dev.parquet")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if cfg.vllm_python:
        sys.executable = cfg.vllm_python
    proc = sa.srv.start_server(a, cfg.port)
    try:
        async def go():
            sem = asyncio.Semaphore(cfg.concurrency)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{cfg.port}") as client:
                out, t0, rows = [], time.time(), list(U.itertuples())
                for i in range(0, len(rows), 64):
                    out += await asyncio.gather(*[OR.one(client, a, cfg, r, sem) for r in rows[i:i + 64]])
                    print(f"{len(out)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
                return out
        out = asyncio.run(go())
    finally:
        proc.terminate()
    U["counts"] = [json.dumps(o) for o in out]
    U[["set", "episode_id", "event_id", "scenario_id", "template", "y", "counts"]].to_parquet(OUT / f"{cfg.actor}_{cfg.split}.parquet")


def states(actor):
    """All of the agent's units with their stored decision states, and the development mask."""
    if actor == "gemma4":
        I = Path("/data/models/graybox_ws/outputs/interrogate")
        parts = []
        for s, dev in (("clean", True), ("fresh", False)):
            U = pd.read_parquet(I / f"gemma4_{s}_units.parquet")
            z = np.load(I / f"gemma4_{s}_decision_hidden.npz")
            U = U.assign(dev=dev, ok=z["ok"].astype(bool))
            parts.append((U, z["h"]))
        U = pd.concat([p[0] for p in parts], ignore_index=True)
        return U, np.concatenate([p[1] for p in parts])
    t17 = importlib.import_module("96_t17_replication_agents")
    U = pd.read_parquet(t17.units_path(actor))
    z = np.load(f"/data/models/graybox_ws/outputs/t17/{actor}_decision_hidden.npz")
    return U.assign(dev=t17.is_dev(U.scenario_id.values), ok=z["ok"].astype(bool)), z["h"]


def reader(actor):
    """The open reader's (Qwen3-8B, layer 32) features of the same transcript and call, aligned with states(actor)."""
    pick = lambda z: z["h"][:, [int(x) for x in z["layers"]].index(32)].astype(np.float32)  # noqa: E731
    if actor == "gemma4":
        return np.concatenate([pick(np.load(f"/data/models/graybox_ws/outputs/t4/{s}_act.npz")) for s in ("clean", "fresh")])
    f = "/data/models/graybox_ws/outputs/t6/gptoss_act.npz" if actor == "gptoss20b" else f"/data/models/graybox_ws/outputs/t17/{actor}_act.npz"
    return pick(np.load(f))


def theta_of(B):
    C = np.array([json.loads(c) for c in B.counts])
    return OR.theta(C[:, :, 0], C[:, :, 1])


def stage_analyze(cfg):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    U, H = states(cfg.actor)
    H = H[:, cfg.layer].astype(np.float64)
    key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
    devmask = (U.dev & U.ok).values
    pca = PCA(32, random_state=0).fit(H[devmask])
    Z = pca.transform(H)
    Bd = pd.read_parquet(OUT / f"{cfg.actor}_dev.parquet")
    idd = np.array([key[k] for k in zip(Bd.episode_id, Bd.event_id.astype(int))])
    Td = theta_of(Bd)
    R = reader(cfg.actor)
    assert len(R) == len(U)
    def fit(mask):
        probe = LogisticRegression(C=0.1, max_iter=5000).fit(Z[mask], U.y.values[mask])
        rpca = PCA(256, random_state=0).fit(R[mask])
        rmap = Ridge(alpha=10.0).fit(rpca.transform(R[mask]), Z[mask])
        return probe, (lambda idx: rmap.predict(rpca.transform(R[idx])))
    if cfg.mode == "dev":                                            # scenario-disjoint cross-fit: nothing fit on a fold's scenarios
        B, ids, T = Bd, idd, Td
        sc = Bd.scenario_id.values
        fold = pd.Series(sc).map({s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(np.unique(sc)))}).values
        n = len(Bd)
        Zhat, ZBi, ZAB, Zshuf = (np.zeros((n, 32)) for _ in range(4))
        s_true, s_rec, s_B, s_AB, s_shuf, s_sup = (np.zeros(n) for _ in range(6))
        perm = np.random.default_rng(1).permutation(n)                # hostile control: answers paired with OTHER units' states
        for f in range(5):
            tr, te = fold != f, fold == f
            probe, recB = fit(devmask & ~U.scenario_id.isin(set(sc[te])).values)
            ZB_tr, ZB_te = recB(idd[tr]), recB(idd[te])
            Zhat[te] = Ridge(alpha=1.0).fit(Td[tr], Z[idd][tr]).predict(Td[te])
            ZBi[te] = ZB_te
            ZAB[te] = ZB_te + Ridge(alpha=1.0).fit(Td[tr], Z[idd][tr] - ZB_tr).predict(Td[te])
            trp = np.where(tr)[0]
            Zshuf[te] = Ridge(alpha=1.0).fit(Td[trp], Z[idd][perm][trp]).predict(Td[te])
            s_true[te], s_rec[te] = probe.decision_function(Z[idd][te]), probe.decision_function(Zhat[te])
            s_B[te], s_AB[te], s_shuf[te] = probe.decision_function(ZBi[te]), probe.decision_function(ZAB[te]), probe.decision_function(Zshuf[te])
            s_sup[te] = LogisticRegression(C=1.0, max_iter=5000).fit(Td[tr], Bd.y.values[tr]).decision_function(Td[te])   # labels on answers
    else:
        probe, recB = fit(devmask)
        B = pd.read_parquet(OUT / f"{cfg.actor}_{'fresh' if cfg.actor == 'gemma4' else 'heldout'}.parquet")
        ids = np.array([key[k] for k in zip(B.episode_id, B.event_id.astype(int))])
        T = theta_of(B)
        Zhat = Ridge(alpha=1.0).fit(Td, Z[idd]).predict(T)
        ZBi = recB(ids)
        ZAB = ZBi + Ridge(alpha=1.0).fit(Td, Z[idd] - recB(idd)).predict(T)
        s_true, s_rec = probe.decision_function(Z[ids]), probe.decision_function(Zhat)
        s_B, s_AB = probe.decision_function(ZBi), probe.decision_function(ZAB)
    assert U.ok.values[ids].all() and (U.y.values[ids] == B.y.values).all()
    y, tm, sc = B.y.values.astype(int), B.template.values, B.scenario_id.values
    Zt = Z[ids]
    s_q0 = -T[:, 0]
    R2 = lambda Zh: round(float(1 - ((Zt - Zh) ** 2).sum() / ((Zt - Zt.mean(0)) ** 2).sum()), 3)  # noqa: E731
    r2 = float(1 - ((Zt - Zhat) ** 2).sum() / ((Zt - Zt.mean(0)) ** 2).sum())
    W = lambda s, i: within_auroc(y[i], s[i], tm[i])  # noqa: E731
    b = cluster_boot(lambda i: W(s_rec, i) - W(s_true, i), sc, cfg.n_boot, seed=1740)
    b = b[np.isfinite(b)]
    bAB = cluster_boot(lambda i: W(s_AB, i) - W(s_true, i), sc, cfg.n_boot, seed=1741)
    bAB = bAB[np.isfinite(bAB)]
    allu = np.arange(len(y))
    res = {"actor": cfg.actor, "mode": cfg.mode, "layer_index": cfg.layer, "units": int(len(B)), "violations": int(y.sum()),
           "templates_mixed": int(sum(1 for t in np.unique(tm) if 0 < y[tm == t].sum() < (tm == t).sum())),
           "recreation_r2_top32": round(r2, 3), "r2": {"answers": R2(Zhat), "reader": R2(ZBi), "reader+answers": R2(ZAB)},
           "within_all": {"true_state": round(float(W(s_true, allu)), 4), "answers": round(float(W(s_rec, allu)), 4), "reader": round(float(W(s_B, allu)), 4),
                          "reader+answers": round(float(W(s_AB, allu)), 4), "complies_alone": round(float(W(s_q0, allu)), 4)},
           "pooled_all": {"true_state": round(float(roc_auc_score(y, s_true)), 4), "answers": round(float(roc_auc_score(y, s_rec)), 4),
                          "reader": round(float(roc_auc_score(y, s_B)), 4), "reader+answers": round(float(roc_auc_score(y, s_AB)), 4)},
           "reader+answers_minus_true_within": [round(float(W(s_AB, allu) - W(s_true, allu)), 4), round(float(np.percentile(bAB, 2.5)), 4),
                                                round(float(np.percentile(bAB, 97.5)), 4)],
           "within": {"true_state": round(float(W(s_true, np.arange(len(y)))), 4), "recreated_from_answers": round(float(W(s_rec, np.arange(len(y)))), 4),
                      "complies_answer_alone": round(float(W(s_q0, np.arange(len(y)))), 4)},
           "pooled": {"true_state": round(float(roc_auc_score(y, s_true)), 4), "recreated_from_answers": round(float(roc_auc_score(y, s_rec)), 4),
                      "complies_answer_alone": round(float(roc_auc_score(y, s_q0)), 4)},
           "recreated_minus_true_within": [round(float(W(s_rec, np.arange(len(y))) - W(s_true, np.arange(len(y)))), 4),
                                           round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}
    if cfg.mode == "dev":
        res["hostile_controls"] = {"shuffled_pairs_map": round(float(W(s_shuf, allu)), 4),
                                   "supervised_probe_on_answers": round(float(W(s_sup, allu)), 4)}
    out = ROOT / "results/t1" / f"decision_greybox_{cfg.actor}_{cfg.mode}{'_DEV' if cfg.mode == 'dev' else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return res


def fit_source(actor="gemma4", layer=3):
    """The source agent's grey box, all on its development data: state PCA, true-state probe, reader->state map, and the
    answers' correction of the reader's recreation (fit on its development battery)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    U, H = states(actor)
    H = H[:, layer].astype(np.float64)
    dm = (U.dev & U.ok).values
    pca = PCA(32, random_state=0).fit(H[dm])
    Z = pca.transform(H)
    probe = LogisticRegression(C=0.1, max_iter=5000).fit(Z[dm], U.y.values[dm])
    R = reader(actor)
    rpca = PCA(256, random_state=0).fit(R[dm])
    rmap = Ridge(alpha=10.0).fit(rpca.transform(R[dm]), Z[dm])
    key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
    Bd = pd.read_parquet(OUT / f"{actor}_dev.parquet")
    idd = np.array([key[k] for k in zip(Bd.episode_id, Bd.event_id.astype(int))])
    ZB = rmap.predict(rpca.transform(R[idd]))
    amap = Ridge(alpha=1.0).fit(theta_of(Bd), Z[idd] - ZB)
    return {"probe": probe, "rpca": rpca, "rmap": rmap, "amap": amap}


def source_score(src, actor, split, layer=3, _cache={}):
    """Source agent src's grey box (fit on its development data) applied to `actor`'s battery units of `split`,
    standardized on those units (label-free)."""
    if src not in _cache:
        _cache[src] = fit_source(src, layer)
    S = _cache[src]
    U, _ = states(actor)
    key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
    B = pd.read_parquet(OUT / f"{actor}_{split}.parquet")
    ids = np.array([key[k] for k in zip(B.episode_id, B.event_id.astype(int))])
    ZB = S["rmap"].predict(S["rpca"].transform(reader(actor)[ids]))
    v = S["probe"].decision_function(ZB + S["amap"].predict(theta_of(B)))
    return B, (v - v.mean()) / (v.std() + 1e-9)


def stage_loo(cfg):
    """Exploratory (dev): each open agent scored by the OTHER open agents' grey boxes, averaged (label-free z-scores)."""
    from sklearn.metrics import roc_auc_score
    OPEN = ["gemma4", "gptoss20b", "qwen38", "museglimmer"]
    out = {}
    for t in OPEN:
        per, B = {}, None
        for s in OPEN:
            if s != t:
                B, per[s] = source_score(s, t, cfg.split, cfg.layer)
        y, tm = B.y.values.astype(int), B.template.values
        ens = np.mean(list(per.values()), axis=0)
        out[t] = {"ensemble_of_others": round(float(within_auroc(y, ens, tm)), 4), "ensemble_pooled": round(float(roc_auc_score(y, ens)), 4),
                  "each_source": {s: round(float(within_auroc(y, v, tm)), 4) for s, v in per.items()}}
        print(t, json.dumps(out[t]), flush=True)
    return out


def stage_transfer(cfg):
    """Exploratory (dev): the source agent's grey box (Gemma-4) applied to another agent's reader features and answers;
    within-template AUROC on the target's battery, against the target's own white-box probe (scenario-disjoint)."""
    from sklearn.metrics import roc_auc_score
    S = fit_source("gemma4", cfg.layer)
    U, _ = states(cfg.actor)
    key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
    B = pd.read_parquet(OUT / f"{cfg.actor}_{cfg.split}.parquet")
    ids = np.array([key[k] for k in zip(B.episode_id, B.event_id.astype(int))])
    R, T = reader(cfg.actor)[ids], theta_of(B)
    ZB = S["rmap"].predict(S["rpca"].transform(R))
    y, tm = B.y.values.astype(int), B.template.values
    sc = {"reader (Gemma coords)": S["probe"].decision_function(ZB), "reader+answers (Gemma coords)": S["probe"].decision_function(ZB + S["amap"].predict(T)),
          "complies alone": -T[:, 0]}
    res = {k: {"within": round(float(within_auroc(y, v, tm)), 4), "pooled": round(float(roc_auc_score(y, v)), 4)} for k, v in sc.items()}
    print(cfg.actor, cfg.split, json.dumps(res))
    return res


def own_scores(actor, split, layer=3):
    """Held-out (or dry-run) scores for one open agent: its white-box probe on true states, and its own grey box
    (reader recreation + answers' correction, all fit on its development data), on the battery units of `split`."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    U, H = states(actor)
    H = H[:, layer].astype(np.float64)
    dm = (U.dev & U.ok).values
    key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
    Bd, B = pd.read_parquet(OUT / f"{actor}_dev.parquet"), pd.read_parquet(OUT / f"{actor}_{split}.parquet")
    idd = np.array([key[k] for k in zip(Bd.episode_id, Bd.event_id.astype(int))])
    ids = np.array([key[k] for k in zip(B.episode_id, B.event_id.astype(int))])
    assert U.ok.values[ids].all() and (U.y.values[ids] == B.y.values).all()
    pca = PCA(32, random_state=0).fit(H[dm])
    Z = pca.transform(H)
    probe = LogisticRegression(C=0.1, max_iter=5000).fit(Z[dm], U.y.values[dm])
    R = reader(actor)
    rpca = PCA(256, random_state=0).fit(R[dm])
    rmap = Ridge(alpha=10.0).fit(rpca.transform(R[dm]), Z[dm])
    amap = Ridge(alpha=1.0).fit(theta_of(Bd), Z[idd] - rmap.predict(rpca.transform(R[idd])))
    ZB = rmap.predict(rpca.transform(R[ids]))
    T = theta_of(B)
    return B, {"white_box": probe.decision_function(Z[ids]), "grey_box": probe.decision_function(ZB + amap.predict(T)),
               "reader_only": probe.decision_function(ZB), "complies_alone": -T[:, 0]}


def cross_scores(actor, split, layer=3):
    """Gemma-4's grey box (fit on Gemma-4's development data) applied to another agent's reader features and answers."""
    S = fit_source("gemma4", layer)
    U, _ = states(actor)
    key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
    B = pd.read_parquet(OUT / f"{actor}_{split}.parquet")
    ids = np.array([key[k] for k in zip(B.episode_id, B.event_id.astype(int))])
    ZB = S["rmap"].predict(S["rpca"].transform(reader(actor)[ids]))
    return S["probe"].decision_function(ZB + S["amap"].predict(theta_of(B)))


def stage_g14(cfg):
    """G14's locked analysis (docs/prereg_G14_decision_grey_box.md). split=heldout (fresh for Gemma-4); dry=True runs it on
    the development batteries to test the code (scores there are not results)."""
    from sklearn.metrics import roc_auc_score
    OPEN = ["gemma4", "gptoss20b", "qwen38", "museglimmer"]
    sp = lambda a: "dev" if cfg.dry else ("fresh" if a == "gemma4" else "heldout")  # noqa: E731
    rng = np.random.default_rng(1750)
    res, own, cross = {"dry_run": bool(cfg.dry)}, {}, {}
    for a in OPEN:
        if not (OUT / f"{a}_{sp(a)}.parquet").exists():
            continue
        B, S = own_scores(a, sp(a), cfg.layer)
        S["others_grey_box"] = np.mean([source_score(o, a, sp(a), cfg.layer)[1] for o in OPEN if o != a], axis=0)
        y, tm, sc = B.y.values.astype(int), B.template.values, B.scenario_id.values
        own[a] = (y, tm, sc, S)
        res[a] = {"units": int(len(y)), "violations": int(y.sum()),
                  "within": {k: round(float(within_auroc(y, v, tm)), 4) for k, v in S.items()},
                  "pooled": {k: round(float(roc_auc_score(y, v)), 4) for k, v in S.items()}}
    def boot(stat, agents):
        """Mean over agents of a within-template AUROC difference; scenarios resampled within each agent."""
        pt = np.mean([stat(*own[a][:2], own[a][3], np.arange(len(own[a][0]))) for a in agents])
        bs = []
        for _ in range(cfg.n_boot):
            v = []
            for a in agents:
                y, tm, sc, S = own[a]
                us = np.unique(sc)
                pick = rng.choice(us, len(us))
                idx = np.concatenate([np.flatnonzero(sc == u) for u in pick])
                v.append(stat(y, tm, S, idx))
            bs.append(np.nanmean(v))
        return [round(float(pt), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]
    W = lambda y, tm, s, i: within_auroc(y[i], s[i], tm[i])  # noqa: E731
    have = [a for a in OPEN if a in own]
    res["P1_grey_minus_white_mean"] = boot(lambda y, tm, S, i: W(y, tm, S["grey_box"], i) - W(y, tm, S["white_box"], i), have)
    res["P2_others_grey_minus_own_white_mean"] = boot(lambda y, tm, S, i: W(y, tm, S["others_grey_box"], i) - W(y, tm, S["white_box"], i), have)
    res["P3_answers_add_to_reader_mean"] = boot(lambda y, tm, S, i: W(y, tm, S["grey_box"], i) - W(y, tm, S["reader_only"], i), have)
    cf = OUT / "claude_t8b.jsonl"
    if not cfg.dry and cf.exists() and (OUT / "claude_t8b_act.npz").exists():
        from sklearn.metrics import roc_auc_score as auc
        t76 = importlib.import_module("76_t8b_claude_all_actions")
        U76 = t76.units()
        z = np.load(OUT / "claude_t8b_act.npz")
        Rc = z["h"][:, [int(x) for x in z["layers"]].index(32)].astype(np.float32)
        C = pd.DataFrame([json.loads(x) for x in open(cf)])
        C = C[~C.set_index(["episode_id", "event_id"]).index.isin(PILOT)].reset_index(drop=True)   # the 5 pilot actions are excluded
        # scripts/76's units in order; scripts/172's enumeration is the same, so the row of each battery unit is its index there
        t172 = importlib.import_module("172_claude_decision_greybox")
        K = {k: i for i, k in enumerate(zip(*[t172.units()[c] for c in ("episode_id", "event_id")]))}
        ids = np.array([K[(e, int(v))] for e, v in zip(C.episode_id, C.event_id)])
        assert len(U76) == len(Rc) and (U76.y.values[ids] == C.y.values).all()
        Tc = theta_of(C)
        zs = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
        gb, ro = [], []
        for src in ["gemma4", "gptoss20b", "qwen38", "museglimmer"]:          # every open agent's grey box, averaged (label-free z-scores)
            Sg = fit_source(src, cfg.layer)
            ZB = Sg["rmap"].predict(Sg["rpca"].transform(Rc[ids]))
            gb.append(zs(Sg["probe"].decision_function(ZB + Sg["amap"].predict(Tc))))
            ro.append(zs(Sg["probe"].decision_function(ZB)))
        sc = {"grey_box": np.mean(gb, axis=0), "reader_only": np.mean(ro, axis=0), "complies_alone": -Tc[:, 0]}
        y, tm = C.y.values.astype(int), C.template.values
        rs = C.template.str.startswith("cons_service_restart").values
        res["claude"] = {"units": int(len(C)), "violations": int(y.sum()), "restart_units": int(rs.sum()), "restart_violations": int(y[rs].sum()),
                         "within": {k: round(float(within_auroc(y, v, tm)), 4) for k, v in sc.items()},
                         "pooled": {k: round(float(auc(y, v)), 4) for k, v in sc.items()},
                         "restart_within": round(float(within_auroc(y[rs], sc["grey_box"][rs], tm[rs])), 4)}
        bs = cluster_boot(lambda i: within_auroc(y[rs][i], sc["grey_box"][rs][i], tm[rs][i]), C.scenario_id.values[rs], cfg.n_boot, seed=1751)
        bs = bs[np.isfinite(bs)]
        res["claude"]["restart_within_ci"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    out = ROOT / "results/t1" / ("g14_DRYRUN_dev.json" if cfg.dry else "g14_decision_grey_box.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def stage_posthoc(cfg):
    """EXPLORATORY baselines asked for by a review, written before the held-out results were seen (not G14 endpoints):
    (a) a tuned white box: 256 PCs of the true state, logistic regression with C chosen by inner 5-fold CV (dev);
    (b) a direct reader probe on the agent's own development labels (reader 256 PCs, CV-tuned logistic);
    (c) a direct reader probe trained on the OTHER open agents' development labels (pooled), the label-matched comparison
        for the ensemble of other agents' grey boxes;
    (d) grey box − reader only (do the answers add?), per agent, scenario-cluster bootstrap."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV
    OPEN = ["gemma4", "gptoss20b", "qwen38", "museglimmer"]
    sp = lambda a: "fresh" if a == "gemma4" else "heldout"  # noqa: E731
    out, store = {"note": "exploratory; written before the held-out results were seen"}, {}
    for a in OPEN:
        U, H = states(a)
        H = H[:, cfg.layer].astype(np.float64)
        dm = (U.dev & U.ok).values
        key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
        B = pd.read_parquet(OUT / f"{a}_{sp(a)}.parquet")
        ids = np.array([key[k] for k in zip(B.episode_id, B.event_id.astype(int))])
        R = reader(a)
        store[a] = (U, dm, R, B, ids)
        y, tm = B.y.values.astype(int), B.template.values
        p256 = PCA(256, random_state=0).fit(H[dm])
        wb = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=5, max_iter=5000).fit(p256.transform(H[dm]), U.y.values[dm])
        rp = PCA(256, random_state=0).fit(R[dm])
        rpb = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=5, max_iter=5000).fit(rp.transform(R[dm]), U.y.values[dm])
        _, S = own_scores(a, sp(a), cfg.layer)
        sc = {"tuned_white_box": wb.decision_function(p256.transform(H[ids])), "reader_probe_own_labels": rpb.decision_function(rp.transform(R[ids])),
              "grey_box": S["grey_box"], "reader_only": S["reader_only"], "white_box_locked": S["white_box"]}
        out[a] = {k: round(float(within_auroc(y, v, tm)), 4) for k, v in sc.items()}
        b = cluster_boot(lambda i: within_auroc(y[i], S["grey_box"][i], tm[i]) - within_auroc(y[i], S["reader_only"][i], tm[i]), B.scenario_id.values, 1000, seed=1760)
        b = b[np.isfinite(b)]
        out[a]["grey_minus_reader_only"] = [round(float(out[a]["grey_box"] - out[a]["reader_only"]), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
    for t in OPEN:                                                   # (c) reader probe trained on the other agents' labels
        Xs, ys = [], []
        for s_ in OPEN:
            if s_ != t:
                U, dm, R, _, _ = store[s_]
                Xs.append(R[dm]); ys.append(U.y.values[dm])
        X, Y = np.concatenate(Xs), np.concatenate(ys)
        rp = PCA(256, random_state=0).fit(X)
        clf = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=5, max_iter=5000).fit(rp.transform(X), Y)
        U, dm, R, B, ids = store[t]
        out[t]["reader_probe_other_agents_labels"] = round(float(within_auroc(B.y.values.astype(int), clf.decision_function(rp.transform(R[ids])), B.template.values)), 4)
        print(t, json.dumps(out[t]), flush=True)
    (ROOT / "results/t1/g14_posthoc_exploratory.json").write_text(json.dumps(out, indent=1))


def main():
    cfg = OmegaConf.merge(CFG, OmegaConf.from_cli())
    {"battery": stage_battery, "analyze": stage_analyze, "transfer": stage_transfer, "g14": stage_g14, "loo": stage_loo, "posthoc": stage_posthoc}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
