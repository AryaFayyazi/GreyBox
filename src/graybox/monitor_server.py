"""Minimal self-hosted monitoring API (FastAPI). Starts without torch; transcripts never leave the machine.

    GB_TEXT_MONITOR=greybox-text-monitor.json uvicorn graybox.monitor_server:app      # CPU, real time
    GB_MONITOR_PROBE=reader_probe.joblib     uvicorn graybox.monitor_server:app       # + GPU reader

POST /v1/monitor/text    the text monitor trained in the browser trainer (escalation cut from its file)
POST /v1/monitor/action  the reader monitor (Qwen3-8B + probe), with a frozen QuantileGate per stream
GET  /v1/health
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .monitor import QuantileGate, Step, TextMonitor

app = FastAPI(title="Grey Box monitor", version="0.2")
_STATE: dict = {"gates": {}}


class StepIn(BaseModel):
    tool: str
    args: dict | None = Field(default_factory=dict)
    result: str = ""


class ActionRequest(BaseModel):
    instructions: str = Field(..., description="the agent's operating instructions (system prompt)")
    task: str = Field(..., description="the user's task")
    history: list[StepIn] = Field(default_factory=list, description="earlier tool calls with their results")
    action: StepIn = Field(..., description="the proposed action (tool '__final__' for a final answer)")
    stream: str = Field("default", description="deployment id; each stream calibrates its own frozen cut (reader only)")
    budget: float = Field(0.05, gt=0, lt=1, description="fraction of this stream's actions to escalate (reader only)")


def _hist(req):
    return [Step(h.tool, h.args or {}, h.result) for h in req.history]


@app.get("/v1/health")
def health():
    return {"text_monitor": bool(os.environ.get("GB_TEXT_MONITOR")), "reader_monitor": bool(os.environ.get("GB_MONITOR_PROBE")),
            "reader_loaded": "reader" in _STATE}


@app.post("/v1/monitor/text")
def monitor_text(req: ActionRequest):
    path = os.environ.get("GB_TEXT_MONITOR")
    if not path:
        raise HTTPException(503, "set GB_TEXT_MONITOR to a model file from the browser trainer")
    if "text" not in _STATE:
        _STATE["text"] = TextMonitor.load_json(path)
    tm, t0 = _STATE["text"], time.perf_counter()
    z = tm.decision(req.instructions, req.task, _hist(req), (req.action.tool, req.action.args))
    return {"score": round(z, 4), "escalate": z >= tm._json["threshold_logit"], "threshold": round(tm._json["threshold_logit"], 4),
            "budget": tm._json.get("budget"), "ms": round(1000 * (time.perf_counter() - t0), 3),
            "note": "text monitor: reliable on the tools and policies it was trained on; retrain for new ones"}


@app.post("/v1/monitor/action")
def monitor_action(req: ActionRequest):
    path = os.environ.get("GB_MONITOR_PROBE")
    if not path:
        raise HTTPException(503, "set GB_MONITOR_PROBE to reader_probe.joblib (needs a GPU and torch)")
    if "reader" not in _STATE:
        from .monitor import ReaderMonitor
        _STATE["reader"] = ReaderMonitor.load(path)
    t0 = time.perf_counter()
    p = _STATE["reader"].score(req.instructions, req.task, _hist(req), (req.action.tool, req.action.args))
    g = _STATE["gates"].setdefault((req.stream, req.budget), QuantileGate(req.budget))
    t = g.threshold
    return {"p_unsafe": round(p, 6), "escalate": g(p), "threshold": t, "calibrated": t is not None, "frozen": g.frozen,
            "ms": round(1000 * (time.perf_counter() - t0), 1),
            "note": "p_unsafe ranks actions but is not calibrated on a new agent; act on 'escalate'"}
