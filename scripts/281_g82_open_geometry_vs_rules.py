#!/usr/bin/env python
"""G82 (docs/prereg_G82_open_geometry_vs_rules.md): does the label-free geometry read an agent's sampled next actions
better than a task-agnostic rule, across many task families, on decisions never used before? Prospective version of an
exploratory comparison (scripts/272 on G70's draws: geometry 0.75, rule 0.45, checker 0.79).
Units, per open agent (Gemma-4, gpt-oss-20b, Qwen3.8-27B, Muse-Glimmer-30B): CAPE-Bench decision points from stored
episodes that no earlier draw used (episodes outside G68's units; not G74's units): up to N_V violating units and N_S safe
units from the templates that hold a violation (numpy seed 8200 + agent index). Draws: scripts/260's stage, unchanged
(8 sampled next actions from the exact turn-start context, each distinct drawn call read by its 8 at-call answers x 4).
Muse-Glimmer runs under vLLM 0.30 (as G68's recorded deviation).
Scores per unit (within template, clusters = scenarios): geometric (G70's statistic: mean distance of the drawn calls'
angles from the agent's median; no label, no question sign), the task-agnostic rule (share of draws that are not CAPE
read tools), CAPE's checker on the draws (a label per drawn call; reported), the signed forecast (reported).
  stage=draws actor=.. ; stage=analyze -> results/t1/g82_open_geometry_vs_rules.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
AGENTS = ("gemma4", "gptoss20b", "qwen38", "museglimmer")
N_V, N_S, SEED, B_N, BOOT_SEED = 150, 150, 8200, 2000, 8201
I = Path("/data/models/graybox_ws/outputs/interrogate")
POOLS = {"gemma4": ["gemma4_clean_units.parquet", "gemma4_fresh_units.parquet"], "gptoss20b": ["gptoss20b_clean_units.parquet"],
         "qwen38": ["qwen38_t17_units.parquet"], "museglimmer": ["museglimmer_t17_units.parquet"]}


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G68 = _mod("g68_g82", "scripts/260_g68_open_says_does.py")
_G68_UNITS = G68.units


def out_file(actor):
    return G68.G27.OUT / f"g82_{actor}.jsonl"


def units(actor):
    U = pd.concat([pd.read_parquet(I / f) for f in POOLS[actor]], ignore_index=True)
    g68 = _G68_UNITS(actor)
    used_eps = set(g68.episode_id)
    g74 = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])) for x in open(G68.G27.OUT / "g74_gemma4.jsonl")}
    U = U[~U.episode_id.isin(used_eps) & np.array([(e, int(v)) not in g74 for e, v in zip(U.episode_id, U.event_id)])]
    U = U.drop_duplicates(["episode_id", "event_id"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED + AGENTS.index(actor))
    viol = np.flatnonzero(U.y.values == 1)
    viol = rng.choice(viol, min(N_V, len(viol)), replace=False)
    tv = set(U.template.values[viol])
    safe = np.flatnonzero((U.y.values == 0) & U.template.isin(tv).values)
    safe = rng.choice(safe, min(N_S, len(safe)), replace=False)
    V = U.loc[np.sort(np.concatenate([viol, safe]))].reset_index(drop=True)
    return V.assign(t=V.event_id.astype(int), stratum=V.template.astype(str), y=V.y.astype(int))


def stage_draws(actor):
    if actor == "museglimmer":
        sys.executable = "/data/models/cape_ws/envs/v030/bin/python"
    G68.units = lambda a: units(a)
    G68.stage_draws(actor, None, out_file(actor))


def stage_analyze():
    from graybox.stats import within_auroc
    G70 = _mod("g70_g82", "scripts/262_g70_open_geometry_vs_whitebox.py")
    reads = G68.G46.G45.T11.READS
    tool = lambda d: json.loads(d)[0] if d else None  # noqa: E731
    data, res = {}, {"agents": {}}
    for a in AGENTS:
        f = out_file(a)
        if not f.exists():
            continue
        U = units(a)
        rec = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])): json.loads(x) for x in open(f)}
        U = U[np.array([(e, int(v)) in rec for e, v in zip(U.episode_id, U.event_id)])].reset_index(drop=True)
        R = [rec[(e, int(v))] for e, v in zip(U.episode_id, U.event_id)]
        D = {"y": U.y.values, "st": U.stratum.values, "cl": U.scenario_id.values, "geo": G70.geometric(R),
             "rule": np.array([np.mean([bool(d) and tool(d) not in reads for d in r["draws"]]) for r in R]),
             "checker": np.array([np.mean([r["reads"][d]["unsafe"] if d and d in r["reads"] else 0 for d in r["draws"]]) for r in R]),
             "signed": np.array([G68.G46.forecast(r) for r in R])}
        data[a] = D
        W = lambda s: within_auroc(D["y"], D[s], D["st"])  # noqa: E731
        res["agents"][a] = {"units": int(len(D["y"])), "violations": int(D["y"].sum()), "templates": int(len(set(D["st"]))),
                            **{s: round(float(W(s)), 4) for s in ("geo", "rule", "checker", "signed")}}
    rng = np.random.default_rng(BOOT_SEED)
    bd, bg = [], []
    for _ in range(B_N):
        d, gg = [], []
        for D in data.values():
            us = np.unique(D["cl"])
            i = np.concatenate([np.flatnonzero(D["cl"] == u) for u in rng.choice(us, len(us))])
            g_ = within_auroc(D["y"][i], D["geo"][i], D["st"][i])
            d.append(g_ - within_auroc(D["y"][i], D["rule"][i], D["st"][i]))
            gg.append(g_)
        bd.append(np.nanmean(d)); bg.append(np.nanmean(gg))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    A = res["agents"]
    res["P1_mean_geo_minus_rule"] = [round(float(np.mean([A[a]["geo"] - A[a]["rule"] for a in A])), 4)] + ci(bd)
    res["P2_mean_geo"] = [round(float(np.mean([A[a]["geo"] for a in A])), 4)] + ci(bg)
    res["reported_mean_checker"] = round(float(np.mean([A[a]["checker"] for a in A])), 4)
    res["endpoints"] = {"P1_lower_gt_0": res["P1_mean_geo_minus_rule"][1] > 0, "P2_lower_gt_0.5": res["P2_mean_geo"][1] > 0.5,
                        "all_four_agents": len(A) == 4}
    (ROOT / "results/t1/g82_open_geometry_vs_rules.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    stage_draws(kw["actor"]) if kw.get("stage") == "draws" else stage_analyze()
