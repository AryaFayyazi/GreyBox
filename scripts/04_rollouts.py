#!/usr/bin/env python
"""Rollout forecaster R (the behavioural-sufficiency ceiling, theory T1) at the turn start of each unit.

The actor is served exactly as in CAPE's rollouts (vLLM OpenAI server, the model's own tool-call and reasoning parsers,
CAPE sampling settings). For each unit: replay the earlier actions through a fresh MockEnv, send the turn-start
conversation with n=K, and label every sampled turn with CAPE's own checker (MockEnv.step on a copy).
R(x) = fraction of the K sampled turns whose action is unsafe.

python scripts/04_rollouts.py actor=gemma4 tag=pilot K=8
"""
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.cape_bridge import CAPE, rollouts, specs  # noqa: E402

sys.path.insert(0, str(CAPE / "scripts"))
import importlib  # noqa: E402

srv = importlib.import_module("20_server_rollouts")
from cape.agents.rollout import clean_tool_name, stable_seed  # noqa: E402
from cape.envs.base import TOOLS, MockEnv  # noqa: E402

IN = Path("/data/models/graybox_ws/outputs/interrogate")
D = OmegaConf.create({"actor": "gemma4", "tag": "pilot", "K": 8, "max_tokens": 4096, "concurrency": 64,
                      "port": 8131, "limit": None})


def openai_msgs(a, spec, events, t):
    msgs = [{"role": "system", "content": spec["system_prompt"] + a.get("system_suffix", "")}] + \
        srv.prior_openai(spec.get("prior_messages")) + [{"role": "user", "content": spec["task_prompt"]}]
    env = MockEnv(spec)
    for e in events:
        if int(e["event_id"]) >= t:
            break
        if not e["is_tool_call"] or e["tool"] is None:
            continue
        args = json.loads(e["args_json"]) if e["args_json"] else None
        if args is not None:
            env.step(e["tool"], args)
        r = e["reasoning_text"] or ""
        c = {"id": f"call_{e['event_id']}", "type": "function",
             "function": {"name": e["tool"], "arguments": e["args_json"] or "{}"}}
        msgs.append({"role": "assistant", "content": e["content_text"] or "", "tool_calls": [c], "thinking": r,
                     "reasoning": r, "reasoning_content": r})
        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": e["tool_result"] or ""})
    return msgs, env


def label(env, choice):
    m = choice["message"]
    calls = m.get("tool_calls") or []
    if choice.get("finish_reason") == "length":
        return {"kind": "truncated", "unsafe": 0, "critical": 0, "tool": None}
    if not calls:
        return {"kind": "final", "unsafe": 0, "critical": 0, "tool": None}
    c = calls[0]
    name = clean_tool_name(c["function"]["name"])
    try:
        args = json.loads(c["function"].get("arguments") or "{}")
        assert isinstance(args, dict)
    except Exception:
        return {"kind": "parse_error", "unsafe": 0, "critical": 0, "tool": name}
    _, info = copy.deepcopy(env).step(name, args)
    return {"kind": "call", "unsafe": int(info.unsafe), "critical": int(info.critical), "tool": name}


async def one(client, a, cfg, r, sem):
    ep = rollouts(r.set).loc[r.episode_id]
    spec = specs(r.bench)[ep.instance_id]
    msgs, env = openai_msgs(a, spec, list(ep.events), int(r.event_id))
    body = {"model": "actor", "messages": msgs, "tools": TOOLS, "max_tokens": cfg.max_tokens, "n": cfg.K,
            "seed": stable_seed("graybox_R", r.episode_id, int(r.event_id)), **a["sampling"], **a["extra"]}
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.post("/v1/chat/completions", json=body, timeout=900)
                resp.raise_for_status()
                labs = [label(env, ch) for ch in resp.json()["choices"]]
                return {"unsafe": [x["unsafe"] for x in labs], "critical": [x["critical"] for x in labs],
                        "kind": [x["kind"] for x in labs], "tool": [x["tool"] for x in labs]}
            except Exception as ex:  # noqa: BLE001
                err = str(ex)[:200]
                body["seed"] += 7919
                await asyncio.sleep(2)
    return {"unsafe": [], "critical": [], "kind": [f"server_error:{err}"], "tool": []}


async def run(cfg, a, units):
    sem = asyncio.Semaphore(cfg.concurrency)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{cfg.port}") as client:
        tasks = [one(client, a, cfg, r, sem) for r in units.itertuples()]
        out, t0 = [], time.time()
        for i in range(0, len(tasks), 256):
            out += await asyncio.gather(*tasks[i:i + 256])
            print(f"{len(out)}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
        return out


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    a = srv.ACTORS[cfg.actor]
    stem = IN / f"{cfg.actor}_{cfg.tag}"
    units = pd.read_parquet(f"{stem}_units.parquet")
    if cfg.limit:
        units = units.head(int(cfg.limit))
    proc = srv.start_server(a, cfg.port)
    try:
        out = asyncio.run(run(cfg, a, units))
    finally:
        proc.terminate()
    res = units[["set", "episode_id", "event_id", "cls"]].copy()
    res["R_unsafe"] = [json.dumps(o["unsafe"]) for o in out]
    res["R_kind"] = [json.dumps(o["kind"]) for o in out]
    res["R_tool"] = [json.dumps(o["tool"]) for o in out]
    res["R"] = [np.mean(o["unsafe"]) if o["unsafe"] else np.nan for o in out]
    res.to_parquet(f"{stem}_rollouts_K{cfg.K}.parquet")
    print(res.groupby("cls").R.describe().to_string())


if __name__ == "__main__":
    main()
