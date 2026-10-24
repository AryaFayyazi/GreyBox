#!/usr/bin/env python
"""G50: the policy forecast on an external benchmark (docs/prereg_G50_tau2_policy_forecast.md). On G18's 360 tau2-bench
airline writes of Claude Haiku 5.5 (the exact contexts tau2's LLMAgent sent; 50 NOT_ALLOWED), before each write exists:
K = 8 next actions drawn from the agent through its API with the agent's own settings (scripts/101: thinking disabled,
effort low, max_tokens 2048, the tau2 tools), in side branches of the context before the write; each distinct drawn tool
call read by the agent's own answers to G18's 8 airline questions (scripts/184's frame and operator note, 4 answers each);
forecast = mean over the 8 draws of G18's fixed read-out (a draw that is not a tool call scores the read-out's minimum).
No label, no training, no other model. Label: the actual write's (G18's units).
  stage=run cap=.. (paid) ; stage=analyze -> results/t1/g50_tau2_policy_forecast.json
"""
import asyncio
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K, N_ANS, B_N, BOOT_SEED = 8, 4, 2000, 5000


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G18 = _mod("g18", "scripts/184_tau2_claude_readout.py")
P101 = _mod("p101", "scripts/101_claude_openai_proxy.py")
OUT = G18.OUT
FN = OUT / "g50_draws.jsonl"


def units():
    U = pd.read_json(OUT / "g18_units_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    return G18.sample(U, 360, 1840).reset_index(drop=True)


def readout(counts):
    sg = np.array([s for _, s in G18.QUESTIONS], float)
    c = np.asarray(counts, float)
    return float((sg * 2 * np.arcsin(np.sqrt((c[:, 0] + 3 / 8) / (c[:, 1] + 3 / 4)))).mean())


def s_min():
    return readout([[N_ANS, N_ANS] if s < 0 else [0, N_ANS] for _, s in G18.QUESTIONS])


async def unit(cl, led, sem, r, ctx_tools):
    system, base = P101.to_anthropic(json.loads(r.messages))
    assert base[-1]["role"] == "user"
    last = dict(base[-1]["content"][-1], cache_control={"type": "ephemeral"})
    base = base[:-1] + [{"role": "user", "content": base[-1]["content"][:-1] + [last]}]

    def charge(u):
        cw, cr = (u.cache_creation_input_tokens or 0), (u.cache_read_input_tokens or 0)
        led.charge(math.ceil(u.input_tokens + 1.25 * cw + 0.1 * cr), u.output_tokens)

    async def draw():
        led.check()
        async with sem:
            x = await cl.messages.create(model="claude-haiku-5-5", max_tokens=2048, system=system, messages=base, tools=ctx_tools,
                                         thinking={"type": "disabled"}, output_config={"effort": "low"})
        charge(x.usage)
        uses = [b for b in x.content if b.type == "tool_use"]
        return [uses[0].name, dict(uses[0].input) if isinstance(uses[0].input, dict) else {}] if uses else None

    async def ask(name, args, q):
        led.check()
        frame = {"type": "text", "text": G18.FRAME.format(name=name, args=json.dumps(args, sort_keys=True), q=q)}
        msgs = base[:-1] + [{"role": "user", "content": base[-1]["content"] + [frame]}]
        async with sem:
            x = await cl.messages.create(model="claude-haiku-5-5", max_tokens=16, system=system + G18.SYSTEM_NOTE, messages=msgs,
                                         tools=ctx_tools, tool_choice={"type": "none"}, temperature=1.0,
                                         thinking={"type": "disabled"}, output_config={"effort": "low"})
        charge(x.usage)
        return G18.parse("".join(b.text for b in x.content if b.type == "text"))

    first = await draw()                                  # writes the cache
    draws = [first] + list(await asyncio.gather(*[draw() for _ in range(K - 1)]))
    reads = {}
    for d in draws:
        if d is None:
            continue
        key = json.dumps(d, sort_keys=True, default=str)
        if key in reads:
            continue
        cnt = []
        for q, _ in G18.QUESTIONS:
            v = await asyncio.gather(*[ask(d[0], d[1], q) for _ in range(N_ANS)])
            cnt.append([sum(1 for a in v if a is False), sum(1 for a in v if a is not None)])
        reads[key] = cnt
    return {"sim_id": r.sim_id, "msg_idx": int(r.msg_idx), "tool": r.tool, "y": int(r.y), "task_id": r.task_id,
            "draws": [None if d is None else json.dumps(d, sort_keys=True, default=str) for d in draws], "reads": reads}


def stage_run(cap):
    import anthropic
    from graybox.budget import HAIKU55, Ledger
    led = Ledger(G18.LEDGER, float(cap), *HAIKU55)
    ctx = json.loads((OUT / "g18_context.json").read_text())
    tools = P101.to_tools(ctx["tools"])
    U = units()
    done = {(json.loads(x)["sim_id"], json.loads(x)["msg_idx"]) for x in open(FN)} if FN.exists() else set()
    todo = [r for r in U.itertuples() if (r.sim_id, int(r.msg_idx)) not in done]
    print(f"{len(U)} units ({int(U.y.sum())} violations), {len(todo)} to do; ledger ${led.total():.3f} cap ${cap}", flush=True)
    cl = anthropic.AsyncAnthropic(api_key=Path("/data/models/graybox_ws/.secrets/anthropic_key").read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(16)

    async def go():
        t0 = time.time()
        for i in range(0, len(todo), 6):
            try:
                res = await asyncio.gather(*[unit(cl, led, sem, r, tools) for r in todo[i:i + 6]])
            except Exception as e:  # noqa: BLE001
                print("stopping:", type(e).__name__, str(e)[:200], flush=True)
                return
            with open(FN, "a") as f:
                for x in res:
                    f.write(json.dumps(x) + "\n")
            print(f"{i + len(res)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def forecast(rec):
    lo = s_min()
    sc = [readout(rec["reads"][d]) if d and d in rec["reads"] and sum(c[1] for c in rec["reads"][d]) > 0 else lo for d in rec["draws"]]
    return float(np.mean(sc + [lo] * max(0, K - len(sc))))


def stage_analyze():
    from graybox.stats import within_auroc
    R = [json.loads(x) for x in open(FN)]
    y = np.array([r["y"] for r in R]); st = np.array([r["tool"] for r in R]); cl = np.array([r["task_id"] for r in R])
    F = np.array([forecast(r) for r in R])
    at = pd.read_json(OUT / "g18_conf.jsonl", orient="records", lines=True)
    atk = {(a, int(b)): readout(c) for a, b, c in zip(at.sim_id, at.msg_idx, at.counts)}
    A = np.array([atk.get((r["sim_id"], r["msg_idx"]), np.nan) for r in R])
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(cl)
    B = {"F": [], "D": []}
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        B["F"].append(within_auroc(y[i], F[i], st[i])); B["D"].append(within_auroc(y[i], F[i], st[i]) - within_auroc(y[i], A[i], st[i]))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    res = {"units": int(len(y)), "violations": int(y.sum()),
           "P1_forecast_within_tool": [round(float(within_auroc(y, F, st)), 4)] + ci(B["F"]),
           "at_call_readout_within_tool": round(float(within_auroc(y, A, st)), 4),
           "forecast_minus_at_call": [round(float(within_auroc(y, F, st) - within_auroc(y, A, st)), 4)] + ci(B["D"]),
           "share_draws_tool_calls": round(float(np.mean([d is not None for r in R for d in r["draws"]])), 4)}
    res["endpoints"] = {"P1_lower_gt_0.5": res["P1_forecast_within_tool"][1] > 0.5}
    (ROOT / "results/t1/g50_tau2_policy_forecast.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    {"run": lambda: stage_run(kw["cap"]), "analyze": stage_analyze}[kw.get("stage", "analyze")]()
