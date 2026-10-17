#!/usr/bin/env python
"""GPT (gpt-6-luna) as a tool-using agent on CAPE-Bench (core24b), through OpenAI's Chat Completions tool calling.
A port of scripts/74 (Claude): same scenarios, MockEnv labels, partner-world reveal and episode format; one tool call
per turn (parallel_tool_calls=False; reasoning_effort="none", as Claude ran with thinking disabled; any extra call still gets a "not executed" result). Own ledger and cap
(outputs/openai/spend.json, $5.00; all input tokens charged at the uncached price, so the ledger over-estimates).
Original scripts/74 docstring follows.

Reproduces the semantics of CAPE's harness (cape/scripts/20_server_rollouts.py episode()): the same scenarios and
MockEnv, one tool call executed per turn (any extra tool_use blocks get a "not executed" result, as the API requires a
result for each), labels from MockEnv.step (unsafe_action, critical), and the reveal from the partner world
(obs_differs_from_partner). Max 20 turns; thinking disabled; effort low. Every call goes to the shared ledger (cap
$9.90); checkpoint every wave of 8 episodes.
  stage=pilot  10 x- episodes (format, cost); stage=pilot2: 40 shuffled x- episodes, seed 48 (violation rate; never analysed)
  stage=run    N x- episodes: scenarios in a shuffled order (rng seed 0; non-nuisance, cf=unsafe), seed 50
Output: outputs/claude_agent/{pilot,run}.parquet in CAPE's rollout format (one row per episode, events list)
python scripts/80_gpt_agent.py stage=pilot | stage=pilot2 | stage=run n=1000
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CAPE = Path("/home/arya/new_safety/cape")
for p in (ROOT / "src", CAPE / "src"):
    sys.path.insert(0, str(p))
from cape.envs.base import TOOLS, MockEnv  # noqa: E402
from graybox.budget import Ledger  # noqa: E402

MODEL = "gpt-6-luna"
PRICE = (0.10 / 1e6, 0.50 / 1e6)          # $ per input / output token (developers.openai.com/api/docs/pricing, 2026-10-17)
OUT = Path("/data/models/graybox_ws/outputs/gpt_agent")
KEY = Path("/data/models/graybox_ws/.secrets/openai_key")
LEDGER = Path("/data/models/graybox_ws/outputs/openai/spend.json")
CAP = 5.00
D = OmegaConf.create({"stage": "pilot", "n": 400, "max_turns": 20, "concurrency": 6})


def specs():
    S = [json.loads(line) for line in open("/data/models/cape_ws/outputs/capebench/core24b/scenarios.jsonl")]
    by = {s["instance_id"]: s for s in S}
    xs = sorted((s for s in S if s["cf_value"] == "unsafe" and not s["nuisance"]), key=lambda s: s["instance_id"])
    out = []
    for s in xs:
        pid = f"{s['base_id']}.p{s['paraphrase_id']}.safe" + (".aware" if s["monitor_aware"] else "")
        if pid in by:
            out.append((s, by[pid]))
    return out


async def episode(client, led, sem, spec, partner, seed, max_turns):
    env, part = MockEnv(spec), MockEnv(partner)
    msgs = [{"role": "system", "content": spec["system_prompt"]}, {"role": "user", "content": spec["task_prompt"]}]
    turns, status = [], "max_turns"
    for t in range(max_turns):
        led.check()
        async with sem:
            r = await client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS, parallel_tool_calls=False,
                                                     max_completion_tokens=2048, seed=seed, reasoning_effort="none")
        led.charge(r.usage.prompt_tokens, r.usage.completion_tokens)
        ch = r.choices[0]
        text = ch.message.content or ""
        uses = list(ch.message.tool_calls or [])
        turn = {"event_id": t, "finish_reason": ch.finish_reason, "reasoning_text": "", "content_text": text, "is_tool_call": bool(uses),
                "tool": None, "args_json": None, "parse_error": None, "tool_result": None, "unsafe_action": False, "critical": False,
                "goal": False, "escalation": False, "violations": "", "obs_differs_from_partner": False, "n_tool_calls": len(uses), "tool_raw": None}
        turns.append(turn)
        if ch.finish_reason == "length":
            status = "truncated"
            break
        if not uses:
            status = "final_answer"
            break
        u = uses[0]
        try:
            args = json.loads(u.function.arguments or "{}")
            args = args if isinstance(args, dict) else {}
        except json.JSONDecodeError as e:
            args, turn["parse_error"] = {}, str(e)[:200]
        obs, info = env.step(u.function.name, args)
        pobs, _ = part.step(u.function.name, args)
        turn.update(tool=u.function.name, tool_raw=u.function.name, args_json=json.dumps(args), tool_result=obs, unsafe_action=bool(info.unsafe),
                    critical=bool(info.critical), goal=bool(info.goal), escalation=bool(info.escalation),
                    violations=",".join(info.violations), obs_differs_from_partner=pobs != obs)
        msgs.append(ch.message.model_dump(exclude_none=True))
        msgs.append({"role": "tool", "tool_call_id": u.id, "content": obs})
        msgs += [{"role": "tool", "tool_call_id": x.id, "content": "Not executed: one tool call per turn."} for x in uses[1:]]
    unsafe = [x["event_id"] for x in turns if x["unsafe_action"]]
    crit = [x["event_id"] for x in turns if x["critical"]]
    reveal = [x["event_id"] for x in turns if x["obs_differs_from_partner"]]
    s = spec
    return {"episode_id": f"{s['instance_id']}.s{seed}", "scenario_id": s["scenario_id"], "scenario_family": s["scenario_family"],
            "template": s["template"], "instance_id": s["instance_id"], "cf_value": s["cf_value"], "seed": seed,
            "model": MODEL, "status": status, "n_events": len(turns), "unsafe_episode": bool(unsafe),
            "first_unsafe_event": unsafe[0] if unsafe else None, "critical_done": bool(crit),
            "reveal_event": reveal[0] if reveal else None, "events": turns}


async def run(cfg, jobs):
    import openai
    led = Ledger(LEDGER, CAP, *PRICE)
    client = openai.AsyncOpenAI(api_key=KEY.read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(cfg.concurrency)
    part = OUT / f"{cfg.stage}.partial.json"
    res = json.loads(part.read_text()) if part.exists() else []
    done = {r["episode_id"] for r in res}
    todo = [j for j in jobs if f"{j[0]['instance_id']}.s{j[2]}" not in done]
    t0 = time.time()
    for i in range(0, len(todo), 8):
        res += await asyncio.gather(*[episode(client, led, sem, s, p, sd, cfg.max_turns) for s, p, sd in todo[i:i + 8]])
        OUT.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(res))
        print(f"{len(res)}/{len(jobs)} episodes, unsafe {sum(r['unsafe_episode'] for r in res)}, informed "
              f"{sum(r['reveal_event'] is not None for r in res)}, spend ${led.total():.3f}, {time.time() - t0:.0f}s", flush=True)
    return res


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    OUT.mkdir(parents=True, exist_ok=True)
    S = specs()
    import numpy as np
    order = np.random.default_rng({"pilot": 49, "pilot2": 48}.get(cfg.stage, 0)).permutation(len(S))   # shuffled across families
    n = {"pilot": 10, "pilot2": 40}.get(cfg.stage, int(cfg.n))
    seed = {"pilot": 49, "pilot2": 48}.get(cfg.stage, 50)
    jobs = [(S[i][0], S[i][1], seed) for i in order[:n]] if cfg.stage != "pilot" else [(s, p, 49) for s, p in S[:n]]
    print(f"{cfg.stage}: {len(jobs)} episodes from {len(S)} x- scenarios; spend ${Ledger(LEDGER, CAP, *PRICE).total():.3f} (cap ${CAP})", flush=True)
    try:
        res = asyncio.run(run(cfg, jobs))
    except Exception as e:
        print("STOPPED:", type(e).__name__, str(e)[:200], flush=True)
        p = OUT / f"{cfg.stage}.partial.json"
        res = json.loads(p.read_text()) if p.exists() else []
    df = pd.DataFrame(res)
    df.to_parquet(OUT / f"{cfg.stage}.parquet")
    print("saved", len(df), "episodes; unsafe", int(df.unsafe_episode.sum()), "; status", df.status.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
