#!/usr/bin/env python
"""G38: a second closed agent, GPT-6-luna, read through its own API at its call and before it (docs/prereg_G38_gpt_readout.md).
Calls: T11's units (scripts/81: every executed non-read call of GPT-6-luna's 400 CAPE-Bench core24b episodes, up to and
including the first unsafe one) in template|tool cells that hold both classes (520 calls, 27 violations; chosen from
labels and strata only). Prefix: the agent's own conversation before the call, rebuilt from the stored episode in
OpenAI's format (system, task, then each earlier turn's assistant message with its tool call and the tool result).
  AT   G15's frame (the call shown, on hold) and G15's 8 questions (web/build_observable_ref.py), 4 answers each
  PRE  G27's pre-call frame (the call NOT shown) and G27's 8 forward-looking questions, 4 answers each
Settings as the agent ran: gpt-6-luna, reasoning_effort none, tools listed, tool_choice none, temperature 1, n = 4.
Read-outs (fixed, nothing fit): AT = G15's score (mean signed angle); PRE = G27's raw read-out (signed sum).
  stage=battery cap=<openai ledger $> [limit=N]   (paid)  -> outputs/gpt_agent/g38.jsonl
  stage=analyze                                    (CPU)   -> results/t1/g38_gpt_readout.json
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
CAPE = Path("/home/arya/new_safety/cape")
for p in (ROOT / "src", ROOT / "scripts", CAPE / "src", CAPE / "scripts"):
    sys.path.insert(0, str(p))
GA = Path("/data/models/graybox_ws/outputs/gpt_agent")
B_N, SEED = 2000, 3800


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def units():
    """T11's enumeration (scripts/81 units()), keeping each call's event and the OpenAI messages before it"""
    T11 = _mod("t11", "scripts/81_t11_gpt_monitor.py")
    E = pd.read_parquet(GA / "run.parquet")
    S = {json.loads(l)["instance_id"]: json.loads(l) for l in open("/data/models/cape_ws/outputs/capebench/core24b/scenarios.jsonl")}
    rows = []
    for ep in E.itertuples():
        spec = S[ep.instance_id]
        msgs = [{"role": "system", "content": spec["system_prompt"]}, {"role": "user", "content": spec["task_prompt"]}]
        for e in ep.events:
            if not e["is_tool_call"] or e["tool"] is None:
                if e.get("content_text"):
                    msgs.append({"role": "assistant", "content": e["content_text"]})
                continue
            args = json.loads(e["args_json"] or "{}")
            if e["tool"] not in T11.READS:
                rows.append({"episode_id": ep.episode_id, "scenario_id": ep.scenario_id, "template": f"{ep.template}|{e['tool']}",
                             "tool": e["tool"], "args": json.dumps(args), "event_id": int(e["event_id"]),
                             "y": int(bool(e["unsafe_action"])), "msgs": json.dumps(msgs)})
                if e["unsafe_action"]:
                    break
            cid = f"call_{ep.Index}_{e['event_id']}"
            msgs = msgs + [{"role": "assistant", "content": e.get("content_text") or None,
                            "tool_calls": [{"id": cid, "type": "function", "function": {"name": e["tool"], "arguments": json.dumps(args)}}]},
                           {"role": "tool", "tool_call_id": cid, "content": e["tool_result"] or ""}]
    U = pd.DataFrame(rows)
    ref = T11.units()
    assert len(U) == len(ref) and (U.episode_id.values == ref.episode_id.values).all() and (U.y.values == ref.y.values).all()
    return U


def selected():
    U = units()
    g = U.groupby("template").y.agg(["sum", "count"])
    mixed = set(g.index[(g["sum"] > 0) & (g["sum"] < g["count"])])
    return U[U.template.isin(mixed)].reset_index(drop=True)


def batteries():
    OR = _mod("obsref", "web/build_observable_ref.py")
    G27 = _mod("g27", "scripts/194_g27_state_tomography.py")
    at = [(q, -1 if j == 0 else 1) for j, q in enumerate(OR.QUESTIONS)]
    return {"AT": (OR.FRAME, at), "PRE": (G27.FRAME, list(G27.QUESTIONS))}


def stage_battery(cfg):
    import openai
    from graybox.budget import Ledger
    from graybox.selfaudit import parse_answer
    from cape.envs.base import TOOLS
    A80 = _mod("a80", "scripts/80_gpt_agent.py")
    assert cfg.get("cap") is not None, "set cap (total OpenAI ledger dollars) explicitly"
    led = Ledger(A80.LEDGER, float(cfg["cap"]), *A80.PRICE)
    U = selected()
    if cfg.get("limit"):
        U = U.iloc[: int(cfg["limit"])]
    fn = GA / "g38.jsonl"
    done = {(json.loads(x)["episode_id"], json.loads(x)["event_id"]) for x in open(fn)} if fn.exists() else set()
    todo = [r for r in U.itertuples() if (r.episode_id, r.event_id) not in done]
    print(f"{len(U)} calls ({int(U.y.sum())} violations), {len(todo)} to do; ledger ${led.total():.3f} cap ${cfg['cap']}", flush=True)
    client = openai.AsyncOpenAI(api_key=A80.KEY.read_text().strip(), max_retries=8)
    B = batteries()

    async def ask(sem, msgs, seed):
        led.check()
        async with sem:
            r = await client.chat.completions.create(model=A80.MODEL, messages=msgs, tools=TOOLS, tool_choice="none", n=4, seed=seed,
                                                     temperature=1.0, max_completion_tokens=16, reasoning_effort="none")
        led.charge(r.usage.prompt_tokens, r.usage.completion_tokens)
        v = [parse_answer(c.message.content or "") for c in r.choices]
        return [None if x is None else int(not x) for x in v]                # 1 = yes

    async def battery(sem, r):
        base = json.loads(r.msgs)
        out = {}
        for arm, (frame, qs) in B.items():
            txt = [frame.format(name=r.tool, args=r.args, q=q) if arm == "AT" else frame.format(q=q) for q, _ in qs]
            out[arm] = await asyncio.gather(*[ask(sem, base + [{"role": "user", "content": t}], SEED + j) for j, t in enumerate(txt)])
        return out

    async def go():
        sem, t0 = asyncio.Semaphore(int(cfg.get("concurrency", 12))), time.time()
        wave = int(cfg.get("wave", 8))
        for i in range(0, len(todo), wave):
            batch = todo[i:i + wave]
            try:
                res = await asyncio.gather(*[battery(sem, r) for r in batch])
            except Exception as e:  # noqa: BLE001  (BudgetExceeded or API failure: keep what is written)
                print("stopping:", type(e).__name__, str(e)[:200], flush=True)
                return
            with open(fn, "a") as f:
                for r, o in zip(batch, res):
                    f.write(json.dumps({"episode_id": r.episode_id, "event_id": int(r.event_id), "scenario_id": r.scenario_id,
                                        "template": r.template, "y": int(r.y), "answers": o}) + "\n")
            print(f"{i + len(batch)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def stage_analyze():
    from graybox.stats import within_auroc
    S = pd.read_json(GA / "g38.jsonl", orient="records", lines=True)
    B = batteries()
    y, st, cl = S.y.values.astype(int), S.template.values, S.scenario_id.values
    th = lambda k, m: 2 * np.arcsin(np.sqrt((k + 3 / 8) / (m + 3 / 4)))  # noqa: E731
    s, unan = {}, {}
    for arm, (_, qs) in B.items():
        sg = np.array([g for _, g in qs])
        C = np.array([[[sum(x for x in a if x is not None), sum(x is not None for x in a)] for a in o[arm]] for o in S.answers], float)
        T = th(C[..., 0], C[..., 1])
        s[arm] = (T * sg).mean(1) if arm == "AT" else T @ sg
        ok = C[..., 1] > 0
        unan[arm] = round(float(((C[..., 0][ok] == 0) | (C[..., 0][ok] == C[..., 1][ok])).mean()), 4)
    W = lambda a, i: within_auroc(y[i], s[a][i], st[i])  # noqa: E731
    rng = np.random.default_rng(SEED)
    us = np.unique(cl)
    keys = {"A1_at_call": lambda i: W("AT", i), "P1_pre_call": lambda i: W("PRE", i), "pre_minus_at": lambda i: W("PRE", i) - W("AT", i)}
    Bs = {k: [] for k in keys}
    for _ in range(B_N):
        idx = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        for k, f in keys.items():
            Bs[k].append(f(idx))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    i0 = np.arange(len(y))
    res = {"calls": int(len(y)), "violations": int(y.sum()), "scenarios": int(len(us)), "unanimity": unan}
    for k, f in keys.items():
        res[k] = [round(float(f(i0)), 4)] + ci(Bs[k])
    res["endpoints"] = {"A1_lower_gt_0.5": res["A1_at_call"][1] > 0.5, "P1_lower_gt_0.5": res["P1_pre_call"][1] > 0.5}
    (ROOT / "results/t1/g38_gpt_readout.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    stage_battery(kw) if kw.get("stage") == "battery" else stage_analyze()
