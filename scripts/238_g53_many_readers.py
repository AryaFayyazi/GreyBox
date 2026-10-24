#!/usr/bin/env python
"""G53: does the theory predict, reader by reader, how much another model loses before the call? (docs/prereg_G53_many_readers.md)
On G48's GPT-6-luna units (faithful replay: one tool call per turn), three more open models sample next actions in the
agent's seat, as Gemma-4 does in G48 (scripts/226's other stage unchanged, with the reader's own server settings):
gpt-oss-20b, Qwen3.8-27B, Muse-Glimmer-30B. For each reader, scripts/230's quantities with r known (CAPE's checker on each
drawn call): observed Brier excess over the agent's own draws, the excess the proposition predicts, and the policy gap.
  stage=other actor=<gptoss20b|qwen38|museglimmer> [vllm_python=..] (GPU) ; stage=analyze -> results/t1/g53_many_readers.json
"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
READERS = ("gemma4", "gptoss20b", "qwen38", "museglimmer")
B_N, BOOT_SEED = 2000, 5300


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G48 = _mod("g48", "scripts/229_g48_gpt_rollout.py")
G46, G46B = G48.G46, G48.G46B


def other_file(actor):
    return G48.OTHER if actor == "gemma4" else G48.OUT / f"g48_gpt_other_{actor}.jsonl"


def stage_other(actor, vllm_python=None):
    sa = importlib.import_module("150_self_audit")
    sa.srv.ACTORS["gemma4"] = dict(sa.srv.ACTORS[actor])      # scripts/226's other stage reads the "gemma4" slot
    if vllm_python:
        sys.executable = vllm_python
    G46.units = G48.units
    G46.OTHER = other_file(actor)
    G46.stage_other()


def stage_analyze():
    from graybox.stats import within_auroc
    U, _ = G48.units()
    key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
    own = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G48.OWN))}
    ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    so = [G46B.shares(own[k]) for k in ks]
    fo, ko = np.array([x[0] for x in so]), np.array([x[1] for x in so])
    y, cl = U.y.values.astype(float), U.scenario_id.values
    st3 = (U.stratum + "|" + U.position.astype(str)).values
    R = {}
    for a in READERS:
        f = other_file(a)
        if not f.exists():
            continue
        oth = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(f))}
        sx = [G46B.shares(oth[k]) if k in oth else (np.nan, 0) for k in ks]
        R[a] = {"fx": np.array([x[0] for x in sx]), "kx": np.array([x[1] for x in sx]),
                "F": np.array([G46.forecast(oth[k]) if k in oth else np.nan for k in ks])}

    def q(a, i):
        fx, kx = R[a]["fx"], R[a]["kx"]
        j = i[(ko[i] >= 2) & (kx[i] >= 2)]
        obs = np.mean((y[j] - fx[j]) ** 2) - np.mean((y[j] - fo[j]) ** 2)
        pred = np.mean((fo[j] - fx[j]) ** 2) - 2 * np.mean(fo[j] * (1 - fo[j]) / (ko[j] - 1))
        gap = np.mean((fo[j] - fx[j]) ** 2) - np.mean(fo[j] * (1 - fo[j]) / (ko[j] - 1)) - np.mean(fx[j] * (1 - fx[j]) / (kx[j] - 1))
        return obs, pred, obs - pred, gap
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(cl)
    boots = []
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        boots.append({a: q(a, i) for a in R})
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    full = np.arange(len(y))
    res = {"units": int(len(y)), "violations": int(y.sum()), "readers": {}}
    for a in R:
        o, p, d, g = q(a, full)
        res["readers"][a] = {"observed_excess": [round(float(o), 4)] + ci([b[a][0] for b in boots]),
                             "predicted_excess": [round(float(p), 4)] + ci([b[a][1] for b in boots]),
                             "observed_minus_predicted": [round(float(d), 4)] + ci([b[a][2] for b in boots]),
                             "policy_gap": [round(float(g), 4)] + ci([b[a][3] for b in boots]),
                             "draws_within_position_auroc": round(float(within_auroc(y, np.where(np.isnan(R[a]["F"]), np.nanmedian(R[a]["F"]), R[a]["F"]), st3)), 4)}
    new = [a for a in R if a != "gemma4"]
    if new:
        pooled = [np.mean([b[a][2] for a in new]) for b in boots]
        res["T1_pooled_new_readers_observed_minus_predicted"] = [round(float(np.mean([q(a, full)[2] for a in new])), 4)] + ci(pooled)
        res["T2_readers_inside_pm0.02"] = int(sum(res["readers"][a]["observed_minus_predicted"][1] >= -0.02 and res["readers"][a]["observed_minus_predicted"][2] <= 0.02 for a in R))
        res["endpoints"] = {"T1_interval_inside_pm0.02": res["T1_pooled_new_readers_observed_minus_predicted"][1] >= -0.02 and res["T1_pooled_new_readers_observed_minus_predicted"][2] <= 0.02}
    (ROOT / "results/t1/g53_many_readers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    if kw.get("stage") == "other":
        stage_other(kw["actor"], kw.get("vllm_python"))
    else:
        stage_analyze()
