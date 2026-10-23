#!/usr/bin/env python
"""G44: does an agent's OWN state, read through its own API before its call exists, forecast its next violation better
than another model reading the same conversation? (docs/prereg_G44_self_vs_other_forecast.md)
Every earlier own-vs-other test (G16, G19, G21) asked about a call already in the transcript, which any reader can judge
from the facts. Before the call exists, what the agent will do next depends on its own policy, which the transcript does
not show. The other model (Gemma-4, an open model; a CONTROL, never part of the method) answers G27's 8 forward-looking
questions in G27's pre-call frame, in the agent's seat (the agent's conversation replayed in Gemma-4's chat format),
4 answers each, as the agent itself did. Score: G27's raw read-out (signed sum of angles); nothing fit.
  closed agent: GPT-6-luna's G40 units (fresh episodes, seed 52) in template|tool strata that hold both classes; its own
                answers are G40's (asked before every turn).
  open agents:  G31's steps (G27b batteries) of gpt-oss-20b, Qwen3.8, Muse-Glimmer in template x tool x step strata that
                hold both classes; their own answers are G27b's.
  stage=battery target=<gpt|gptoss20b|qwen38|museglimmer>   (GPU) -> outputs/g44/<target>_by_gemma4.parquet
  stage=analyze                                              (CPU) -> results/t1/g44_self_vs_other_forecast.json
"""
import asyncio
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
OUT = Path("/data/models/graybox_ws/outputs/g44")
OPEN = ["gptoss20b", "qwen38", "museglimmer"]
PORT, B_N, SEED = 8188, 2000, 4400


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G27 = _mod("g27", "scripts/194_g27_state_tomography.py")
SG = np.array([q[1] for q in G27.QUESTIONS])


def risk(C):
    C = np.asarray(C, float)
    return float(2 * np.arcsin(np.sqrt((C[:, 0] + 3 / 8) / (C[:, 1] + 3 / 4))) @ SG)


def gpt_units():
    """G40's units in template|tool strata holding both classes, with the OpenAI messages before each unit's turn"""
    G40 = _mod("g40", "scripts/213_g40_gpt_precall_confirm.py")
    T11 = G40.T11
    specs = {s["instance_id"]: s for s, _ in G40.A80.specs()}
    rows = []
    for e in (json.loads(x) for x in open(G40.G38.GA / "g40.jsonl")):
        sp = specs[e["instance_id"]]
        msgs = [{"role": "system", "content": sp["system_prompt"]}, {"role": "user", "content": sp["task_prompt"]}]
        k = 0
        for x in e["turns"]:
            if x["kind"] != "call":
                break
            if x["tool"] not in T11.READS:
                own = [[sum(a for a in q if a is not None), sum(a is not None for a in q)] for q in x["pre"]]
                rows.append({"instance_id": e["instance_id"], "scenario_id": e["scenario_id"], "stratum": f"{e['template']}|{x['tool']}",
                             "position": k, "y": int(x["unsafe_action"]), "own": own, "msgs": json.dumps(msgs)})
                k += 1
                if x["unsafe_action"]:
                    break
            cid = f"call_{x['t']}"
            msgs = msgs + [{"role": "assistant", "content": x.get("content") or None,
                            "tool_calls": [{"id": cid, "type": "function", "function": {"name": x["tool"], "arguments": x["args_json"]}}]},
                           {"role": "tool", "tool_call_id": cid, "content": x["tool_result"] or ""}]
    U = pd.DataFrame(rows)
    g = U.groupby("stratum").y.agg(["sum", "count"])
    keep = set(g.index[(g["sum"] > 0) & (g["sum"] < g["count"])])
    return U[U.stratum.isin(keep)].reset_index(drop=True)


def open_steps(actor):
    S = pd.read_parquet(G27.OUT / f"{actor}_steps.parquet")
    k = S.template + "|" + S.tool + "|" + S.t.astype(str)
    g = S.groupby(k).y.agg(["sum", "count"])
    mixed = set(g.index[(g["sum"] > 0) & (g["sum"] < g["count"])])
    return S[k.isin(mixed).values].reset_index(drop=True)


def stage_battery(target):
    import httpx
    sa = importlib.import_module("150_self_audit")
    a = sa.srv.ACTORS["gemma4"]
    if target == "gpt":
        U = gpt_units()
        bases = [json.loads(m) for m in U.msgs]
    else:
        U = open_steps(target)
        bases = [G27.prefix(sa, a, r) for r in U.itertuples()]          # the target's conversation in Gemma-4's format
    print(target, len(U), "units,", int(U.y.sum()), "violations", flush=True)
    proc = sa.srv.start_server(a, PORT)
    try:
        async def go():
            sem = asyncio.Semaphore(G27.CONC)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}") as client:
                out, t0 = [], time.time()
                for i in range(0, len(bases), 64):
                    out += await asyncio.gather(*[G27.one(client, a, b, sem) for b in bases[i:i + 64]])
                    print(f"{len(out)}/{len(bases)} {time.time() - t0:.0f}s", flush=True)
                return out
        out = asyncio.run(go())
    finally:
        proc.terminate()
    OUT.mkdir(parents=True, exist_ok=True)
    U = U.drop(columns=[c for c in ("msgs",) if c in U.columns])
    U.assign(other=[json.dumps(o) for o in out]).to_parquet(OUT / f"{target}_by_gemma4.parquet")


def stage_analyze():
    from graybox.stats import within_auroc
    rng = np.random.default_rng(SEED)
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    res = {}
    G = pd.read_parquet(OUT / "gpt_by_gemma4.parquet")
    own = np.array([risk(json.loads(c) if isinstance(c, str) else list(c)) for c in G.own])
    oth = np.array([risk(json.loads(c)) for c in G.other])
    y, cl = G.y.values, G.scenario_id.values
    st2 = G.stratum.values
    st3 = np.array([f"{a}|{b}" for a, b in zip(G.stratum, G.position)])
    W = lambda s, st, i: within_auroc(y[i], s[i], st[i])  # noqa: E731
    i0 = np.arange(len(y))
    us = np.unique(cl)
    b1, b2 = [], []
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        b1.append(W(own, st2, i) - W(oth, st2, i))
        b2.append(W(own, st3, i) - W(oth, st3, i))
    res["closed_gpt"] = {"units": int(len(y)), "violations": int(y.sum()),
                         "own_within_template_tool": round(float(W(own, st2, i0)), 4), "other_within_template_tool": round(float(W(oth, st2, i0)), 4),
                         "own_within_position": round(float(W(own, st3, i0)), 4), "other_within_position": round(float(W(oth, st3, i0)), 4)}
    res["F1_gpt_own_minus_other_template_tool"] = [round(float(W(own, st2, i0) - W(oth, st2, i0)), 4)] + ci(b1)
    res["F1b_gpt_own_minus_other_position"] = [round(float(W(own, st3, i0) - W(oth, st3, i0)), 4)] + ci(b2)
    D = {}
    for a in OPEN:
        f = OUT / f"{a}_by_gemma4.parquet"
        if not f.exists():
            continue
        S = pd.read_parquet(f)
        D[a] = {"own": np.array([risk(json.loads(c)) for c in S.counts]), "oth": np.array([risk(json.loads(c)) for c in S.other]),
                "y": S.y.values, "st": np.array([f"{x}|{z}|{t}" for x, z, t in zip(S.template, S.tool, S.t)]), "cl": S.scenario_id.values}
        d = D[a]
        res[a] = {"steps": int(len(d["y"])), "violations": int(d["y"].sum()),
                  "own": round(float(within_auroc(d["y"], d["own"], d["st"])), 4), "other": round(float(within_auroc(d["y"], d["oth"], d["st"])), 4)}
    if D:
        bo = []
        for _ in range(B_N):
            v = []
            for d in D.values():
                u = np.unique(d["cl"])
                i = np.concatenate([np.flatnonzero(d["cl"] == x) for x in rng.choice(u, len(u))])
                v.append(within_auroc(d["y"][i], d["own"][i], d["st"][i]) - within_auroc(d["y"][i], d["oth"][i], d["st"][i]))
            bo.append(np.nanmean(v))
        res["F2_open_pooled_own_minus_other"] = [round(float(np.mean([res[a]["own"] - res[a]["other"] for a in D])), 4)] + ci(bo)
    res["endpoints"] = {"F1_lower_gt_0": res["F1_gpt_own_minus_other_template_tool"][1] > 0,
                        "F2_lower_gt_0": bool(D) and res["F2_open_pooled_own_minus_other"][1] > 0}
    (ROOT / "results/t1/g44_self_vs_other_forecast.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    stage_battery(kw["target"]) if kw.get("stage") == "battery" else stage_analyze()
