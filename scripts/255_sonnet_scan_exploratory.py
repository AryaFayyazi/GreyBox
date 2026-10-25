#!/usr/bin/env python
"""Exploratory scan (no claim, not preregistered): on which CAPE-Bench templates does Claude Sonnet 5.5 violate, and how
often? G46c's episode loop (exact contexts, one call per turn) with the model changed and thinking off as Sonnet 5.5
requires (thinking between_tools, effort low). 24 templates x 2 scenarios x seed 400. Ledger: Anthropic, Sonnet 5.5 list
prices ($2 / $10 per MTok; cache writes charged as input at 1.25x).
  -> /data/models/graybox_ws/outputs/l3/sonnet_scan.jsonl ; summary printed
"""
import asyncio
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL, SONNET55 = "claude-sonnet-5-5", (2.0 / 1e6, 10.0 / 1e6)


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G46C = _mod("g46c_s", "scripts/232_g46c_exact_context.py")
A74 = G46C.A74
OUTF = G46C.OUT / "sonnet_scan.jsonl"


async def episode(client, led, sem, spec, seed):
    from cape.envs.base import MockEnv
    env = MockEnv(spec)
    msgs = [{"role": "user", "content": [{"type": "text", "text": spec["task_prompt"]}]}]
    turns, status = [], "max_turns"
    for t in range(A74.D.max_turns):
        led.check()
        async with sem:
            r = await client.messages.create(model=MODEL, max_tokens=1024, system=spec["system_prompt"], messages=msgs,
                                             tools=A74.A_TOOLS, thinking={"type": "between_tools"}, output_config={"effort": "low"})
        led.charge(r.usage.input_tokens + int(1.25 * (r.usage.cache_creation_input_tokens or 0)), r.usage.output_tokens)
        uses = [b for b in r.content if b.type == "tool_use"]
        if r.stop_reason == "max_tokens" or not uses:
            status = "truncated" if r.stop_reason == "max_tokens" else "final_answer"
            break
        u = uses[0]
        args = dict(u.input) if isinstance(u.input, dict) else {}
        obs, info = env.step(u.name, args)
        turns.append({"t": t, "tool": u.name, "unsafe_action": bool(info.unsafe), "n_tool_use": len(uses)})
        msgs = msgs + [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]},
                       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": u.id, "content": obs}] +
                        [{"type": "tool_result", "tool_use_id": x.id, "content": "Not executed: one tool call per turn.", "is_error": True}
                         for x in uses[1:]]}]
    return {"instance_id": spec["instance_id"], "template": spec["template"], "seed": seed, "status": status, "turns": turns,
            "violated": any(x["unsafe_action"] for x in turns)}


def main(cap, per_template=2, smoke=False):
    import anthropic
    from graybox.budget import Ledger
    led = Ledger(A74.LEDGER, float(cap), *SONNET55)
    by = defaultdict(list)
    for s, _ in A74.specs():
        by[s["template"]].append(s)
    jobs = [s for t in sorted(by) for s in by[t][:per_template]]
    if smoke:
        jobs = jobs[:1]
    client = anthropic.AsyncAnthropic(api_key=A74.KEY.read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(16)

    async def go():
        t0 = time.time()
        res = []
        for i in range(0, len(jobs), 8):
            res += await asyncio.gather(*[episode(client, led, sem, s, 400) for s in jobs[i:i + 8]])
            print(f"{len(res)}/{len(jobs)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
        return res
    res = asyncio.run(go())
    if not smoke:
        with open(OUTF, "a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
    agg = defaultdict(lambda: [0, 0])
    for r in res:
        agg[r["template"]][0] += r["violated"]; agg[r["template"]][1] += 1
    for t, (v, n) in sorted(agg.items()):
        print(f"{t:28s} {v}/{n}")
    print("turns of first:", res[0]["turns"][:3], res[0]["status"])


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    main(kw["cap"], int(kw.get("per", 2)), kw.get("smoke") == "1")
