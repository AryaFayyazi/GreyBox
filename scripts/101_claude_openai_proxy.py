#!/usr/bin/env python
"""A local OpenAI-compatible Chat Completions endpoint backed by Claude Haiku 5.5, for benchmark harnesses that speak
the OpenAI format (tau2-bench via litellm). Every call charges the shared Anthropic ledger and stops at the cap.

Settings as in T8b/T16b: thinking disabled, effort "low". Haiku 5.5 rejects sampling parameters (temperature, top_p,
top_k), so they are dropped and the dropped names are logged. Automatic prompt caching (top-level cache_control); the
ledger charges effective input tokens = input + 1.25 x cache writes + 0.1 x cache reads. Every call is logged
(tokens only, no content) to <log>. The Anthropic key is read from the secrets file and never logged.
python scripts/101_claude_openai_proxy.py port=8801 log=/data/models/graybox_ws/outputs/tau2_claude/proxy.jsonl
"""
import json
import math
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.budget import HAIKU55, BudgetExceeded, Ledger  # noqa: E402

KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
CAP = float(os.environ.get("GB_CAP", "14.80"))  # default: $9.90 + the $5.00 added on 2026-10-17, less $0.10 margin; GB_CAP overrides
MODEL = "claude-haiku-5-5"
D = OmegaConf.create({"port": 8801, "log": "/data/models/graybox_ws/outputs/tau2_claude/proxy.jsonl", "max_tokens": 2048})
START = "(The conversation starts.)"          # Anthropic needs a user turn first; tau2's agent speaks first


def to_anthropic(messages):
    system, out = [], []

    def add(role, blocks):
        if out and out[-1]["role"] == role:
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": list(blocks)})
    for m in messages:
        r, c = m["role"], m.get("content")
        text = c if isinstance(c, str) else "".join(p.get("text", "") for p in (c or []) if isinstance(p, dict))
        if r in ("system", "developer"):
            system.append(text)
        elif r == "user":
            add("user", [{"type": "text", "text": text or "(empty)"}])
        elif r == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": text or "(empty)"}])
        elif r == "assistant":
            blocks = [{"type": "text", "text": text}] if text and text.strip() else []
            for tc in m.get("tool_calls") or []:
                f = tc["function"]
                try:
                    args = json.loads(f.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": f.get("arguments")}
                blocks.append({"type": "tool_use", "id": tc["id"], "name": f["name"], "input": args})
            add("assistant", blocks or [{"type": "text", "text": "(no content)"}])
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": START}]})
    return "\n\n".join(system), out


def to_tools(tools):
    return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}} for t in tools or []]


def make_app(cfg):
    import anthropic
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    app = FastAPI()
    client = anthropic.AsyncAnthropic(api_key=KEY.read_text().strip(), max_retries=8)
    led = Ledger(LEDGER, CAP, *HAIKU55)
    log = Path(cfg.log)
    log.parent.mkdir(parents=True, exist_ok=True)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        try:
            led.check()
        except BudgetExceeded as e:
            return JSONResponse({"error": {"message": str(e), "type": "budget_exceeded"}}, status_code=402)
        system, msgs = to_anthropic(body["messages"])
        kw = {"model": MODEL, "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or cfg.max_tokens),
              "system": system, "messages": msgs, "thinking": {"type": "disabled"}, "output_config": {"effort": "low"},
              "cache_control": {"type": "ephemeral"}}
        if body.get("tools"):
            kw["tools"] = to_tools(body["tools"])
        dropped = [k for k in ("temperature", "top_p", "top_k") if body.get(k) is not None]   # Haiku 5.5 rejects sampling parameters
        t0 = time.time()
        r = await client.messages.create(**kw)
        u = r.usage
        cw, cr = (u.cache_creation_input_tokens or 0), (u.cache_read_input_tokens or 0)
        eff = math.ceil(u.input_tokens + 1.25 * cw + 0.1 * cr)
        total = led.charge(eff, u.output_tokens)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [{"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}} for b in r.content if b.type == "tool_use"]
        finish = "tool_calls" if calls else ("length" if r.stop_reason == "max_tokens" else "stop")
        with open(log, "a") as f:
            f.write(json.dumps({"t": round(t0, 2), "s": round(time.time() - t0, 2), "tools": bool(body.get("tools")), "in": u.input_tokens,
                                "cache_write": cw, "cache_read": cr, "out": u.output_tokens, "stop": r.stop_reason, "n_calls": len(calls), "dropped": dropped,
                                "ledger_usd": round(total, 4)}) + "\n")
        msg = {"role": "assistant", "content": text or None}
        if calls:
            msg["tool_calls"] = calls
        return {"id": r.id, "object": "chat.completion", "created": int(t0), "model": MODEL,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": u.input_tokens + cw + cr, "completion_tokens": u.output_tokens,
                          "total_tokens": u.input_tokens + cw + cr + u.output_tokens}}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}
    return app


def main():
    import uvicorn
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    uvicorn.run(make_app(cfg), host="127.0.0.1", port=int(cfg.port), log_level="warning")


if __name__ == "__main__":
    main()
