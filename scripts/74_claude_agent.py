#!/usr/bin/env python
"""Claude Haiku 5.5 as a tool-using agent on CAPE-Bench (core24b), through Anthropic's own tool-use API.

Reproduces the semantics of CAPE's harness (cape/scripts/20_server_rollouts.py episode()): the same scenarios and
MockEnv, one tool call executed per turn (any extra tool_use blocks get a "not executed" result, as the API requires a
result for each), labels from MockEnv.step (unsafe_action, critical), and the reveal from the partner world
(obs_differs_from_partner). Max 20 turns; thinking disabled; effort low. Every call goes to the shared ledger (cap
$9.90); checkpoint every wave of 8 episodes.
  stage=pilot  10 x- episodes (format, cost); stage=pilot2: 40 shuffled x- episodes, seed 48 (violation rate; never analysed)
  stage=run    N x- episodes: scenarios in a shuffled order (rng seed 0; non-nuisance, cf=unsafe), seed 50
Output: outputs/claude_agent/{pilot,run}.parquet in CAPE's rollout format (one row per episode, events list)
python scripts/74_claude_agent.py stage=pilot | stage=run n=400
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
from graybox.budget import HAIKU55, Ledger  # noqa: E402

OUT = Path("/data/models/graybox_ws/outputs/claude_agent")
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
CAP = 9.90
D = OmegaConf.create({"stage": "pilot", "n": 400, "max_turns": 20, "concurrency": 6})
A_TOOLS = [{"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]} for t in TOOLS]


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
    msgs = [{"role": "user", "content": spec["task_prompt"]}]
    turns, status = [], "max_turns"
    for t in range(max_turns):
        led.check()
        async with sem:
            r = await client.messages.create(model="claude-haiku-5-5", max_tokens=1024, system=spec["system_prompt"], messages=msgs,
                                             tools=A_TOOLS, thinking={"type": "disabled"}, output_config={"effort": "low"})
        led.charge(r.usage.input_tokens, r.usage.output_tokens)
        text = "".join(b.text for b in r.content if b.type == "text")
        uses = [b for b in r.content if b.type == "tool_use"]
        turn = {"event_id": t, "finish_reason": r.stop_reason, "reasoning_text": "", "content_text": text, "is_tool_call": bool(uses),
                "tool": None, "args_json": None, "parse_error": None, "tool_result": None, "unsafe_action": False, "critical": False,
                "goal": False, "escalation": False, "violations": "", "obs_differs_from_partner": False, "n_tool_calls": len(uses), "tool_raw": None}
        turns.append(turn)
        if r.stop_reason == "max_tokens":
            status = "truncated"
            break
        if not uses:
            status = "final_answer"
            break
        u = uses[0]
        args = dict(u.input) if isinstance(u.input, dict) else {}
        obs, info = env.step(u.name, args)
        pobs, _ = part.step(u.name, args)
        turn.update(tool=u.name, tool_raw=u.name, args_json=json.dumps(args), tool_result=obs, unsafe_action=bool(info.unsafe),
                    critical=bool(info.critical), goal=bool(info.goal), escalation=bool(info.escalation),
                    violations=",".join(info.violations), obs_differs_from_partner=pobs != obs)
        msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
        res = [{"type": "tool_result", "tool_use_id": u.id, "content": obs}]
        res += [{"type": "tool_result", "tool_use_id": x.id, "content": "Not executed: one tool call per turn.", "is_error": True} for x in uses[1:]]
        msgs.append({"role": "user", "content": res})
    unsafe = [x["event_id"] for x in turns if x["unsafe_action"]]
    crit = [x["event_id"] for x in turns if x["critical"]]
    reveal = [x["event_id"] for x in turns if x["obs_differs_from_partner"]]
    s = spec
    return {"episode_id": f"{s['instance_id']}.s{seed}", "scenario_id": s["scenario_id"], "scenario_family": s["scenario_family"],
            "template": s["template"], "instance_id": s["instance_id"], "cf_value": s["cf_value"], "seed": seed,
            "model": "claude-haiku-5-5", "status": status, "n_events": len(turns), "unsafe_episode": bool(unsafe),
            "first_unsafe_event": unsafe[0] if unsafe else None, "critical_done": bool(crit),
            "reveal_event": reveal[0] if reveal else None, "events": turns}


async def run(cfg, jobs):
    import anthropic
    led = Ledger(LEDGER, CAP, *HAIKU55)
    client = anthropic.AsyncAnthropic(api_key=KEY.read_text().strip(), max_retries=8)
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
    S = specs()
    import numpy as np
    order = np.random.default_rng({"pilot": 49, "pilot2": 48}.get(cfg.stage, 0)).permutation(len(S))   # shuffled across families
    n = {"pilot": 10, "pilot2": 40}.get(cfg.stage, int(cfg.n))
    seed = {"pilot": 49, "pilot2": 48}.get(cfg.stage, 50)
    jobs = [(S[i][0], S[i][1], seed) for i in order[:n]] if cfg.stage != "pilot" else [(s, p, 49) for s, p in S[:n]]
    print(f"{cfg.stage}: {len(jobs)} episodes from {len(S)} x- scenarios; spend ${Ledger(LEDGER, CAP, *HAIKU55).total():.3f} (cap ${CAP})", flush=True)
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
