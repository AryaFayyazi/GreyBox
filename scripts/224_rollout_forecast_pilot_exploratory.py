#!/usr/bin/env python
"""EXPLORATORY pilot (not preregistered; development data for a later locked test): the Bayes forecast of the next
violation read through the agent's own API by sampling its next action (prop:selfforecast: P(Y | V) = sum_a pi(a | V)
P(Y | a, V)). On G45's Claude Haiku 5.5 episodes (development data), before each unit's turn (G45's units: non-read
calls up to the first unsafe one), we draw K next actions from the agent in side branches (its own API, same settings
as its loop) and read each drawn call with the agent's own at-call answers (graybox.selfaudit STATE_QUESTIONS, 1 answer
each). Forecasts:
  self_rollout   mean over the K draws of the at-call read-out (no label, nothing fit, no other model)
  draw_unsafe    share of the K draws that CAPE's checker calls unsafe (a diagnostic of the forecastable ceiling; uses
                 the environment, not part of the method)
  pre_questions  G45's pre-call question read-out (already measured)
  stage=run cap=<anthropic ledger $> n_units=<int> ; stage=analyze -> results/t1/rollout_forecast_pilot_exploratory.json
"""
import asyncio
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAPE = Path("/home/arya/new_safety/cape")
for p in (ROOT / "src", ROOT / "scripts", CAPE / "src", CAPE / "scripts"):
    sys.path.insert(0, str(p))
OUT = Path("/data/models/graybox_ws/outputs/g45")
FN = OUT / "rollout_pilot.jsonl"
K, SEED = 8, 2240


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


A74 = _mod("a74", "scripts/74_claude_agent.py")
G44 = _mod("g44", "scripts/220_g44_self_vs_other_forecast.py")


def units(n_units):
    """G45's Claude units with the Anthropic messages and the replayed environment before each unit's turn"""
    from cape.envs.base import MockEnv
    sys.path.insert(0, str(ROOT / "scripts"))
    G45 = _mod("g45", "scripts/222_g45_closed_self_forecast.py")
    reads = G45.T11.READS
    specs = {s["instance_id"]: s for s, _ in A74.specs()}
    rows = []
    for e in (json.loads(x) for x in open(OUT / "claude.jsonl")):
        sp = specs[e["instance_id"]]
        msgs = [{"role": "user", "content": [{"type": "text", "text": sp["task_prompt"]}]}]
        done, k = [], 0
        for x in e["turns"]:
            if x["kind"] != "call":
                break
            if x["tool"] not in reads:
                own = [[sum(a for a in q if a is not None), sum(a is not None for a in q)] for q in x["pre"]]
                rows.append({"instance_id": e["instance_id"], "seed": e["seed"], "scenario_id": e["scenario_id"], "t": x["t"],
                             "stratum": f"{e['template']}|{x['tool']}", "position": k, "y": int(x["unsafe_action"]),
                             "pre_questions": G44.risk(own), "msgs": json.dumps(msgs), "done": json.dumps(done)})
                k += 1
                if x["unsafe_action"]:
                    break
            blocks = ([{"type": "text", "text": x["content"]}] if x.get("content") else [])
            tid = f"toolu_{x['t']:03d}"
            blocks.append({"type": "tool_use", "id": tid, "name": x["tool"], "input": json.loads(x["args_json"] or "{}")})
            msgs = msgs + [{"role": "assistant", "content": blocks},
                           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": str(x["tool_result"] or "")}]}]
            done.append([x["tool"], x["args_json"]])
    U = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    eps = U[["instance_id", "seed"]].drop_duplicates().values
    order = [tuple(r) for r in eps[rng.permutation(len(eps))]]
    keep, n = set(), 0
    for ep in order:
        m = int(((U.instance_id == ep[0]) & (U.seed == ep[1])).sum())
        if n + m > n_units:
            break
        keep.add(ep); n += m
    return U[[(a, b) in keep for a, b in zip(U.instance_id, U.seed)]].reset_index(drop=True), specs, MockEnv


async def one_unit(client, led, sem, r, spec, MockEnv):
    from graybox.selfaudit import STATE_FRAME, STATE_QUESTIONS, parse_answer
    env = MockEnv(spec)
    for name, aj in json.loads(r.done):
        env.step(name, json.loads(aj or "{}"))
    msgs = json.loads(r.msgs)
    last = msgs[-1]
    content = list(last["content"])
    content[-1] = dict(content[-1], cache_control={"type": "ephemeral"})
    cached = msgs[:-1] + [{"role": "user", "content": content}]

    def charge(u):
        cw, cr = (u.cache_creation_input_tokens or 0), (u.cache_read_input_tokens or 0)
        led.charge(math.ceil(u.input_tokens + 1.25 * cw + 0.1 * cr), u.output_tokens)

    async def draw():
        led.check()
        async with sem:
            x = await client.messages.create(model="claude-haiku-5-5", max_tokens=1024, system=spec["system_prompt"], messages=cached,
                                             tools=A74.A_TOOLS, thinking={"type": "disabled"}, output_config={"effort": "low"})
        charge(x.usage)
        uses = [b for b in x.content if b.type == "tool_use"]
        return (uses[0].name, dict(uses[0].input) if isinstance(uses[0].input, dict) else {}) if uses else None

    async def ask(name, args, q):
        led.check()
        m = cached[:-1] + [{"role": "user", "content": content + [{"type": "text", "text": STATE_FRAME.format(
            name=name, args=json.dumps(args, sort_keys=True, default=str), q=q)}]}]
        async with sem:
            x = await client.messages.create(model="claude-haiku-5-5", max_tokens=16, system=spec["system_prompt"], messages=m,
                                             tools=A74.A_TOOLS, tool_choice={"type": "none"}, temperature=1.0,
                                             thinking={"type": "disabled"}, output_config={"effort": "low"})
        charge(x.usage)
        return parse_answer("".join(b.text for b in x.content if b.type == "text"))

    draws = await asyncio.gather(*[draw() for _ in range(K)])
    out = []
    for d in draws:
        if d is None:
            out.append({"call": None})
            continue
        name, args = d
        _, info = copy.deepcopy(env).step(name, args)
        ans = await asyncio.gather(*[ask(name, args, q) for q, _ in STATE_QUESTIONS])
        th = []
        for v, (_, sign) in zip(ans, STATE_QUESTIONS):
            k, m = (1 if v is False else 0), (0 if v is None else 1)
            th.append(sign * 2 * math.asin(math.sqrt((k + 3 / 8) / (m + 3 / 4))))
        out.append({"call": [name, args], "unsafe": int(bool(info.unsafe)), "readout": float(np.mean(th)), "answers": ans})
    return {"instance_id": r.instance_id, "seed": int(r.seed), "t": int(r.t), "draws": out}


def stage_run(cap, n_units):
    import anthropic
    from graybox.budget import HAIKU55, Ledger
    led = Ledger(A74.LEDGER, float(cap), *HAIKU55)
    U, specs, MockEnv = units(int(n_units))
    done = {(json.loads(x)["instance_id"], json.loads(x)["seed"], json.loads(x)["t"]) for x in open(FN)} if FN.exists() else set()
    todo = [r for r in U.itertuples() if (r.instance_id, int(r.seed), int(r.t)) not in done]
    print(f"{len(U)} units ({int(U.y.sum())} violations), {len(todo)} to do; ledger ${led.total():.3f} cap ${cap}", flush=True)
    client = anthropic.AsyncAnthropic(api_key=A74.KEY.read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(24)

    async def go():
        t0 = time.time()
        for i in range(0, len(todo), 6):
            try:
                res = await asyncio.gather(*[one_unit(client, led, sem, r, specs[r.instance_id], MockEnv) for r in todo[i:i + 6]])
            except Exception as e:  # noqa: BLE001
                print("stopping:", type(e).__name__, str(e)[:200], flush=True)
                return
            with open(FN, "a") as f:
                for x in res:
                    f.write(json.dumps(x) + "\n")
            print(f"{i + len(res)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def stage_analyze():
    from graybox.stats import within_auroc
    U, _, _ = units(10 ** 9)
    R = {(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(FN))}
    U = U[[(a, int(b), int(c)) in R for a, b, c in zip(U.instance_id, U.seed, U.t)]].reset_index(drop=True)
    dr = [R[(a, int(b), int(c))]["draws"] for a, b, c in zip(U.instance_id, U.seed, U.t)]
    U["self_rollout"] = [np.mean([x["readout"] if x["call"] else -np.pi / 2 for x in d]) for d in dr]
    U["draw_unsafe"] = [np.mean([x.get("unsafe", 0) for x in d]) for d in dr]
    st2, st3 = U.stratum.values, (U.stratum + "|" + U.position.astype(str)).values
    res = {"units": int(len(U)), "violations": int(U.y.sum()), "episodes": int(U[["instance_id", "seed"]].drop_duplicates().shape[0])}
    for c in ("self_rollout", "draw_unsafe", "pre_questions"):
        res[c] = {"template_tool": round(float(within_auroc(U.y.values, U[c].values, st2)), 4),
                  "position": round(float(within_auroc(U.y.values, U[c].values, st3)), 4)}
    rng = np.random.default_rng(SEED)
    us = U.scenario_id.unique()
    B = []
    for _ in range(1000):
        i = np.concatenate([np.flatnonzero(U.scenario_id.values == u) for u in rng.choice(us, len(us))])
        B.append([within_auroc(U.y.values[i], U[c].values[i], st3[i]) for c in ("self_rollout", "draw_unsafe", "pre_questions")])
    B = np.array(B)
    res["position_ci"] = {c: [round(float(np.nanpercentile(B[:, j], 2.5)), 4), round(float(np.nanpercentile(B[:, j], 97.5)), 4)]
                          for j, c in enumerate(("self_rollout", "draw_unsafe", "pre_questions"))}
    (ROOT / "results/t1/rollout_forecast_pilot_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    {"run": lambda: stage_run(kw["cap"], kw.get("n_units", 150)), "analyze": stage_analyze}[kw.get("stage", "analyze")]()
