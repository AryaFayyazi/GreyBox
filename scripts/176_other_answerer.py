#!/usr/bin/env python
"""G16: are the AGENT'S OWN answers needed, or does another model answering the same 8 questions about the same call
do as well? (docs/prereg_G16_own_vs_other.md). Same held-out calls as G15 (G14's battery files), same questions,
frame, 4 answers each, same read-out; the answerer is a different open model served by vLLM:
  Gemma-4 answers about gpt-oss's, Qwen3.8's, Muse-Glimmer's and Claude's calls; Muse-Glimmer answers about Gemma-4's.
The answerer reads the agent's conversation up to the call (in its own chat format), then the question.
  python scripts/176_other_answerer.py answerer=gemma4 target=gptoss20b      (GPU) -> outputs/decision_greybox/<target>_heldout_by_<answerer>.parquet
  python scripts/176_other_answerer.py answerer=gemma4 target=claude
"""
import asyncio
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.selfaudit import parse_answer  # noqa: E402

sa = importlib.import_module("150_self_audit")
_s = importlib.util.spec_from_file_location("obsref", ROOT / "web/build_observable_ref.py")
OR = importlib.util.module_from_spec(_s)
_s.loader.exec_module(OR)
OUT = Path("/data/models/graybox_ws/outputs/decision_greybox")
CFG = OmegaConf.create({"answerer": "gemma4", "target": "gptoss20b", "n": 4, "max_tokens": 4096, "concurrency": 64, "port": 8176,
                        "seed": 1710, "vllm_python": None})


def claude_msgs(row, units172):
    """Claude's stored conversation (Anthropic blocks) as OpenAI-style messages, with the system prompt first."""
    r = units172[(row.episode_id, int(row.event_id))]
    out = [{"role": "system", "content": r["system"]}]
    for m in json.loads(r["msgs"]):
        if m["role"] == "user":
            for b in m["content"]:
                if b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
                elif b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(b["content"])})
        else:
            text = "".join(b.get("text", "") for b in m["content"] if b["type"] == "text")
            calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in m["content"] if b["type"] == "tool_use"]
            out.append({"role": "assistant", "content": text or None, **({"tool_calls": calls} if calls else {})})
    return out, r["tool"], r["args"]


async def one(client, a, cfg, base, name, args, sem):
    yes = []
    for j, q in enumerate(OR.QUESTIONS):
        msgs = base + [{"role": "user", "content": OR.FRAME.format(name=name, args=args, q=q)}]
        body = {"model": "actor", "messages": msgs, "tools": sa.TOOLS, "tool_choice": "none", "n": cfg.n, "seed": cfg.seed + j,
                "max_tokens": cfg.max_tokens, **a["sampling"], **a["extra"]}
        k = m = 0
        async with sem:
            for _ in range(3):
                try:
                    resp = await client.post("/v1/chat/completions", json=body, timeout=900)
                    resp.raise_for_status()
                    for ch in resp.json()["choices"]:
                        v = parse_answer(ch["message"].get("content") or "")
                        if v is not None:
                            m += 1
                            k += int(not v)
                    break
                except Exception:  # noqa: BLE001
                    body["seed"] += 7919
                    await asyncio.sleep(2)
        yes.append([k, m])
    return yes


def main():
    cfg = OmegaConf.merge(CFG, OmegaConf.from_cli())
    a = sa.srv.ACTORS[cfg.answerer]
    if cfg.target == "claude":
        t172 = importlib.import_module("172_claude_decision_greybox")
        U = t172.units()
        units172 = {(r.episode_id, int(r.event_id)): {"system": r.system, "msgs": r.msgs, "tool": r.tool, "args": r.args} for r in U.itertuples()}
        B = pd.DataFrame([json.loads(x) for x in open(OUT / "claude_t8b.jsonl")])
        items = [claude_msgs(r, units172) for r in B.itertuples()]
    else:
        split = "fresh" if cfg.target == "gemma4" else "heldout"
        B = pd.read_parquet(OUT / f"{cfg.target}_{split}.parquet")
        Uall = sa.units("gemma4_fresh" if cfg.target == "gemma4" else f"{cfg.target}_heldout")
        full = {(e, int(v)): r for e, v, r in zip(Uall.episode_id, Uall.event_id, Uall.itertuples())}
        items = []
        for r0 in B.itertuples():
            r = full[(r0.episode_id, int(r0.event_id))]                # the unit row (with bench) of this battery call
            base = sa.audit_msgs(a, r, "ask")[:-1]            # the target's conversation before the call, in the answerer's format
            e = {int(x["event_id"]): x for x in sa.rollouts(r.set).loc[r.episode_id].events}[int(r.event_id)]
            items.append((base, e["tool"], e["args_json"] or "{}"))
    if cfg.vllm_python:
        sys.executable = cfg.vllm_python
    proc = sa.srv.start_server(a, cfg.port)
    try:
        async def go():
            sem = asyncio.Semaphore(cfg.concurrency)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{cfg.port}") as client:
                out, t0 = [], time.time()
                for i in range(0, len(items), 64):
                    out += await asyncio.gather(*[one(client, a, cfg, b, nm, ar, sem) for b, nm, ar in items[i:i + 64]])
                    print(f"{len(out)}/{len(items)} {time.time() - t0:.0f}s", flush=True)
                return out
        out = asyncio.run(go())
    finally:
        proc.terminate()
    B = B.copy()
    B["counts"] = [json.dumps(o) for o in out]
    tag = "claude_t8b" if cfg.target == "claude" else f"{cfg.target}_{'fresh' if cfg.target == 'gemma4' else 'heldout'}"
    B[[c for c in ("set", "episode_id", "event_id", "scenario_id", "template", "y", "counts") if c in B.columns]].to_parquet(OUT / f"{tag}_by_{cfg.answerer}.parquet")


if __name__ == "__main__":
    main()
