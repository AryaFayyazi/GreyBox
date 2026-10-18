#!/usr/bin/env python
"""A local Chat Completions endpoint that forwards to OpenAI's gpt-6-luna with T11's settings (reasoning_effort "none",
no parallel tool calls), for tau2-bench via litellm. Every call charges the OpenAI ledger (outputs/openai/spend.json,
cap $5.00; the full prompt is charged, cached or not). Sampling parameters are dropped (as for Claude) and logged.
Calls are logged with token counts only. The key is read from the secrets file and never logged.
python scripts/101b_gpt_openai_proxy.py port=8802 log=/data/models/graybox_ws/outputs/tau2_gpt/proxy.jsonl
"""
import json
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.budget import BudgetExceeded, Ledger  # noqa: E402

KEY = Path("/data/models/graybox_ws/.secrets/openai_key")
LEDGER, CAP = Path("/data/models/graybox_ws/outputs/openai/spend.json"), 5.00
PRICE = (0.10 / 1e6, 0.50 / 1e6)                        # gpt-6-luna, $ per input / output token (scripts/83)
MODEL = "gpt-6-luna"
D = OmegaConf.create({"port": 8802, "log": "/data/models/graybox_ws/outputs/tau2_gpt/proxy.jsonl"})
KEEP = {"messages", "tools", "tool_choice", "max_tokens", "max_completion_tokens"}


def make_app(cfg):
    import openai
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    app = FastAPI()
    client = openai.AsyncOpenAI(api_key=KEY.read_text().strip(), max_retries=8)
    led = Ledger(LEDGER, CAP, *PRICE)
    log = Path(cfg.log)
    log.parent.mkdir(parents=True, exist_ok=True)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        try:
            led.check()
        except BudgetExceeded as e:
            return JSONResponse({"error": {"message": str(e), "type": "budget_exceeded"}}, status_code=402)
        dropped = sorted(k for k in body if k not in KEEP | {"model", "stream"})
        kw = {k: v for k, v in body.items() if k in KEEP}
        if "max_tokens" in kw:
            kw["max_completion_tokens"] = kw.pop("max_tokens")
        kw.update(model=MODEL, reasoning_effort="none")
        if kw.get("tools"):
            kw["parallel_tool_calls"] = False
        t0 = time.time()
        r = await client.chat.completions.create(**kw)
        u = r.usage
        total = led.charge(u.prompt_tokens, u.completion_tokens)
        with open(log, "a") as f:
            f.write(json.dumps({"t": round(t0, 2), "s": round(time.time() - t0, 2), "tools": bool(kw.get("tools")), "in": u.prompt_tokens,
                                "out": u.completion_tokens, "finish": r.choices[0].finish_reason, "dropped": dropped, "ledger_usd": round(total, 4)}) + "\n")
        return r.model_dump(exclude_none=True)

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
