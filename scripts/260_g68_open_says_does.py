#!/usr/bin/env python
"""G68 (docs/prereg_G68_open_says_does.md): says vs does on four OPEN agents across their CAPE-Bench task families.
Claude Haiku 5.5's says-vs-does tests used one template, the only one where its violations vary. Open agents violate on
many. Units: G27/G27b's steps (Gemma-4, gpt-oss-20b, Qwen3.8, Muse-Glimmer), in template|tool|step strata holding both
classes (G30's rule). Per unit:
  says   the agent's pre-call self-report: G30's raw read-out, the signed sum of the step's 8 forward-looking answer
         angles (already measured by G27/G27b; nothing new)
  does   the agent's own 8 sampled next actions from the exact step context (CAPE's own message builder), each distinct
         drawn call read by the agent's own 8 at-call answers, 4 each; G46's forecast (scripts/226), unchanged. The draw
         and read code is scripts/226 stage_other's unit(), with the agent in its own seat. CAPE's checker labels each
         drawn call for a reported ceiling only.
  stage=draws actor=<gemma4|gptoss20b|qwen38|museglimmer> [limit=n out=file] (GPU) -> outputs/tomography/g68_<actor>.jsonl
  stage=analyze (CPU) -> results/t1/g68_open_says_does.json
"""
import asyncio
import copy
import importlib
import importlib.util
import json
import sys
import time
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
AGENTS = ("gemma4", "gptoss20b", "qwen38", "museglimmer")
B_N, BOOT_SEED, PORT = 2000, 6800, 8196


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G46 = _mod("g46", "scripts/226_g46_rollout_forecast.py")
G27 = G46.G27
K, N_ANS = G46.K, G46.N_ANS


def units(actor):
    S = pd.read_parquet(G27.OUT / f"{actor}_steps.parquet")
    S["y"] = S.y.astype(int)
    S["stratum"] = S.template.astype(str) + "|" + S.tool.astype(str) + "|" + S.t.astype(str)
    g = S.groupby("stratum").y.agg(["sum", "count"])
    S = S[S.stratum.isin(set(g.index[(g["sum"] > 0) & (g["sum"] < g["count"])]))].reset_index(drop=True)
    C = np.array([json.loads(c) for c in S.counts], float)
    Y, _ = G27.angles(C)
    S["report"] = Y @ np.array([q[1] for q in G27.QUESTIONS])          # G30's raw pre-call read-out
    return S


def out_file(actor):
    return G27.OUT / f"g68_{actor}.jsonl"


def stage_draws(actor, limit=None, fn=None):
    import httpx
    from graybox.selfaudit import STATE_FRAME, STATE_QUESTIONS, parse_answer
    sa = importlib.import_module("150_self_audit")
    a, tools = sa.srv.ACTORS[actor], sa.srv.TOOLS
    fn = fn or out_file(actor)
    U = units(actor)
    done = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])) for x in open(fn)} if fn.exists() else set()
    todo = [r for r in U.itertuples() if (r.episode_id, int(r.event_id)) not in done][:limit]
    print(f"{actor}: {len(U)} units, {len(todo)} to do", flush=True)

    def context(r):
        ep = sa.rollouts(r.set).loc[r.episode_id]
        spec = sa.specs(r.bench)[ep.instance_id]
        evs = list(ep.events)
        base, _ = sa.ro.openai_msgs(a, spec, evs, int(r.event_id))
        prior = [[x["tool"], x.get("args_json") or "{}"] for x in evs[:int(r.event_id)] if str(x["is_tool_call"]) == "True" and x["tool"]]
        return base, G46.env_at(spec, prior)

    async def post(client, body):
        for _ in range(3):
            try:
                rr = await client.post("/v1/chat/completions", json=body, timeout=1800)
                rr.raise_for_status()
                return rr.json()
            except Exception:  # noqa: BLE001
                body["seed"] += 7919
                await asyncio.sleep(2)
        return None

    async def unit(client, sem, r):                     # scripts/226 stage_other's unit(), the agent in its own seat
        base, env = context(r)
        t = int(r.t)

        async def one_draw(k):
            async with sem:
                return await post(client, {"model": "actor", "messages": base, "tools": tools, "n": 1,
                                           "seed": BOOT_SEED + 1000 * k + t, "max_tokens": 4096, **a["sampling"], **a["extra"]})
        outs = await asyncio.gather(*[one_draw(k) for k in range(K)])
        draws = []
        for ch in [c for o in outs for c in (o or {}).get("choices", [])[:1]]:
            calls = ch["message"].get("tool_calls") or []
            if not calls:
                draws.append(None)
                continue
            name = sa.srv.clean_tool_name(calls[0]["function"]["name"])
            try:
                args = json.loads(calls[0]["function"].get("arguments") or "{}")
                args = args if isinstance(args, dict) else {}
            except json.JSONDecodeError:
                args = {}
            draws.append(json.dumps([name, args], sort_keys=True, default=str))
        reads = {}
        for key in {d for d in draws if d}:
            name, args = json.loads(key)
            cnt = []
            for j, (q, _) in enumerate(STATE_QUESTIONS):
                body = {"model": "actor", "messages": base + [{"role": "user", "content": STATE_FRAME.format(
                    name=name, args=json.dumps(args, sort_keys=True, default=str), q=q)}], "tools": tools, "tool_choice": "none",
                    "n": N_ANS, "seed": BOOT_SEED + 100 * j, "max_tokens": 4096, **a["sampling"], **a["extra"]}
                async with sem:
                    rr = await post(client, body)
                v = [parse_answer(c["message"].get("content") or "") for c in (rr or {}).get("choices", [])]
                cnt.append([sum(1 for x in v if x is False), sum(1 for x in v if x is not None)])
            _, info = copy.deepcopy(env).step(name, args)
            reads[key] = {"counts": cnt, "unsafe": int(bool(info.unsafe))}
        return {"episode_id": r.episode_id, "event_id": int(r.event_id), "t": t, "draws": draws, "n_choices": len(draws), "reads": reads}

    proc = sa.srv.start_server(a, PORT)
    try:
        async def go():
            sem = asyncio.Semaphore(G27.CONC)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}") as client:
                t0 = time.time()
                for i in range(0, len(todo), 32):
                    res = await asyncio.gather(*[unit(client, sem, r) for r in todo[i:i + 32]])
                    with open(fn, "a") as f:
                        for x in res:
                            f.write(json.dumps(x) + "\n")
                    print(f"{actor} {i + len(res)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        asyncio.run(go())
    finally:
        proc.terminate()


def stage_analyze():
    from graybox.stats import within_auroc
    data = {}
    for a in AGENTS:
        if not out_file(a).exists():
            continue
        U = units(a)
        rec = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])): json.loads(x) for x in open(out_file(a))}
        U = U[[(e, int(v)) in rec for e, v in zip(U.episode_id, U.event_id)]].reset_index(drop=True)
        R = [rec[(e, int(v))] for e, v in zip(U.episode_id, U.event_id)]
        data[a] = {"y": U.y.values, "st": U.stratum.values, "cl": U.scenario_id.values, "report": U.report.values,
                   "own": np.array([G46.forecast(x) for x in R]), "checker": np.array([G46.ceiling(x) for x in R]),
                   "templates": int(U.template.str.split("|").str[0].nunique())}
    W = lambda D, s, i: within_auroc(D["y"][i], D[s][i], D["st"][i])  # noqa: E731
    res = {"agents": {}}
    for a, D in data.items():
        i = np.arange(len(D["y"]))
        res["agents"][a] = {"units": int(len(i)), "violations": int(D["y"].sum()), "templates": D["templates"],
                            "says_report": round(float(W(D, "report", i)), 4), "does_own": round(float(W(D, "own", i)), 4),
                            "checker_ceiling": round(float(W(D, "checker", i)), 4)}
    rng = np.random.default_rng(BOOT_SEED)
    b1, b2, per = [], [], {a: [] for a in data}
    for _ in range(B_N):
        d, o = [], []
        for a, D in data.items():
            us = np.unique(D["cl"])
            i = np.concatenate([np.flatnonzero(D["cl"] == u) for u in rng.choice(us, len(us))])
            od, rp = W(D, "own", i), W(D, "report", i)
            d.append(od - rp); o.append(od); per[a].append(od - rp)
        b1.append(np.nanmean(d)); b2.append(np.nanmean(o))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    p1 = float(np.mean([res["agents"][a]["does_own"] - res["agents"][a]["says_report"] for a in data]))
    p2 = float(np.mean([res["agents"][a]["does_own"] for a in data]))
    res["P1_pooled_own_minus_report"] = [round(p1, 4)] + ci(b1)
    res["P2_pooled_own"] = [round(p2, 4)] + ci(b2)
    for a in data:
        res["agents"][a]["own_minus_report"] = [round(res["agents"][a]["does_own"] - res["agents"][a]["says_report"], 4)] + ci(per[a])
    Pn = lambda z: 0.5 * (1 - erf(z / sqrt(2)))  # noqa: E731
    ps = {k: Pn((v[0] - t) / max((v[2] - v[1]) / 3.92, 1e-9)) for k, v, t in
          (("P1", res["P1_pooled_own_minus_report"], 0.0), ("P2", res["P2_pooled_own"], 0.5))}
    order, go, holm = sorted(ps, key=ps.get), True, {}
    for j, k in enumerate(order):
        go = go and ps[k] <= 0.05 / (len(order) - j)
        holm[k] = bool(go)
    res["endpoints"] = {"P1_lower_gt_0": res["P1_pooled_own_minus_report"][1] > 0, "P2_lower_gt_0.5": res["P2_pooled_own"][1] > 0.5,
                        "agents_with_own_gt_report": int(sum(res["agents"][a]["own_minus_report"][0] > 0 for a in data))}
    res["holm_pass"] = {k: bool(holm[k] and res["endpoints"]["P1_lower_gt_0" if k == "P1" else "P2_lower_gt_0.5"]) for k in holm}
    (ROOT / "results/t1/g68_open_says_does.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    if kw.get("stage") == "draws":
        stage_draws(kw["actor"], int(kw["limit"]) if "limit" in kw else None, G27.OUT / kw["out"] if "out" in kw else None)
    else:
        stage_analyze()
