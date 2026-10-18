"""Minimal self-hosted monitoring API (FastAPI). Starts without torch; transcripts never leave the machine.

    GB_TEXT_MONITOR=greybox-text-monitor.json uvicorn graybox.monitor_server:app      # CPU, real time
    GB_MONITOR_PROBE=reader_probe.joblib     uvicorn graybox.monitor_server:app       # + GPU reader

POST /v1/monitor/text    the text monitor (browser trainer or python -m graybox.train_text; escalation cut from its file).
                         Also a browser-reader file (graybox-embed-monitor/1; graybox.embed_monitor).
POST /v1/monitor/action  the reader monitor (Qwen3-8B + probe), with a frozen QuantileGate per stream
GET  /v1/health
"""
from __future__ import annotations

import json
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .monitor import Q, QuantileGate, Step, TextMonitor, act_text

from . import __version__  # noqa: E402
app = FastAPI(title="Grey Box monitor", version=__version__)
_STATE: dict = {"gates": {}}
_READER_LOCK = threading.Lock()      # one forward pass at a time on the GPU model


class StepIn(BaseModel):
    model_config = ConfigDict(extra="forbid")      # a misspelt field is an error, not silently ignored
    tool: str
    args: dict | str | None = Field(default_factory=dict, description="a dict, or a JSON string (OpenAI's function.arguments)")
    result: str | None = ""

    @field_validator("args")
    @classmethod
    def _parse_args(cls, v):
        if isinstance(v, str):
            try:
                v = json.loads(v or "{}")
            except json.JSONDecodeError:
                return {"_raw": v}
            return v if isinstance(v, dict) else {"_raw": v}
        return v or {}

    @field_validator("result")
    @classmethod
    def _result(cls, v):
        return v or ""


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str = Field("", description="the agent's operating instructions (system prompt)")
    task: str = Field("", description="the user's task")
    history: list[StepIn] = Field(default_factory=list, description="earlier tool calls with their results")
    action: StepIn = Field(..., description="the proposed action (tool '__final__' for a final answer)")
    stream: str = Field("default", description="deployment id; each stream calibrates its own frozen cut (reader only)")
    budget: float = Field(0.05, gt=0, lt=1, description="fraction of this stream's actions to escalate (reader only)")
    gate: bool = Field(False, description="text monitor: escalate the top `budget` of THIS stream's own traffic (a frozen "
                       "QuantileGate) instead of the file's cut. Use it on a new kind of task, where the file's cut can escalate almost nothing")
    context: str | None = Field(None, description="optional prebuilt context (graybox.chat.chat_ctx, for conversations with "
                                "a user); replaces instructions, task and history. Use it only with a monitor trained on such contexts")


def _hist(req):
    return [Step(h.tool, h.args or {}, h.result) for h in req.history]


def _load_text():
    """Load the text (or browser-reader) monitor named by GB_TEXT_MONITOR once; remember a load error for /v1/health."""
    path = os.environ.get("GB_TEXT_MONITOR")
    if not path or "text" in _STATE or "text_error" in _STATE:
        return
    try:
        d = json.loads(open(path).read())
        if d.get("type") == "embed":
            from .embed_monitor import EmbedMonitor
            _STATE["text"] = EmbedMonitor(d)
        else:
            _STATE["text"] = TextMonitor.load_json(path)
    except Exception as e:  # noqa: BLE001
        _STATE["text_error"] = f"{type(e).__name__}: {e}"


@app.on_event("startup")
def _startup():
    _load_text()                      # fail fast: a bad GB_TEXT_MONITOR shows in /v1/health, not at the first request


@app.get("/v1/health")
def health():
    _load_text()
    return {"text_monitor": "text" in _STATE, "text_monitor_error": _STATE.get("text_error"),
            "reader_monitor": bool(os.environ.get("GB_MONITOR_PROBE")), "reader_loaded": "reader" in _STATE}


@app.post("/v1/monitor/text")
def monitor_text(req: ActionRequest):
    if not os.environ.get("GB_TEXT_MONITOR"):
        raise HTTPException(503, "set GB_TEXT_MONITOR to a monitor file (browser trainer or python -m graybox.train_text)")
    _load_text()
    if "text" not in _STATE:
        raise HTTPException(503, f"the monitor file could not be loaded: {_STATE.get('text_error')}")
    tm, t0 = _STATE["text"], time.perf_counter()
    if req.context is None and tm._json.get("chat"):
        raise HTTPException(422, "this monitor was trained with conversation-aware context (--chat): send 'context' (graybox.chat.chat_ctx)")
    if req.context is not None:
        z = tm.decision(None, None, None, None, rendered=(req.context, act_text(req.action.tool, req.action.args)))
    else:
        z = tm.decision(req.instructions, req.task, _hist(req), (req.action.tool, req.action.args))
    if req.gate:
        g = _STATE["gates"].setdefault(("text:" + req.stream, req.budget), QuantileGate(req.budget))
        esc, thr, budget = bool(g(z)), g.threshold, req.budget
    else:
        esc, thr, budget = z >= tm._json["threshold_logit"], tm._json["threshold_logit"], tm._json.get("budget")
    return {"score": round(z, 4), "unit": "logit", "escalate": esc, "threshold": None if thr is None else round(thr, 4),
            "cut": "live gate on this stream" if req.gate else "the file's cut",
            "budget": budget, "ms": round(1000 * (time.perf_counter() - t0), 3),
            "note": "text monitor: reliable on the tools and policies it was trained on; retrain for new ones"}


class CalibrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    monitor: str = Field("reader", pattern="^(reader|text)$", description="which endpoint's gate to seed: reader (/action) or text (/text with gate=true)")
    stream: str = "default"
    budget: float = Field(0.05, gt=0, lt=1)
    scores: list[float] = Field(..., description="this agent's past scores, no labels: p_unsafe for the reader, the logit 'score' for the text monitor")


@app.post("/v1/monitor/calibrate")
def calibrate(req: CalibrateRequest):
    """Seed a stream's escalation cut from past unlabelled traffic: the cut is the (1 - budget) quantile of all given
    scores, frozen at once. Replaces the stream's gate."""
    if len(req.scores) < 100:
        raise HTTPException(422, "need at least 100 past scores")
    if req.monitor == "text" and all(0.0 <= x <= 1.0 for x in req.scores):
        raise HTTPException(422, "text monitors gate on logits: send the 'score' values of /v1/monitor/text (or Guard's v.score), not probabilities")
    g = QuantileGate(req.budget, warmup=len(req.scores), min_seen=100).fit(req.scores)
    _STATE["gates"][(("text:" if req.monitor == "text" else "") + req.stream, req.budget)] = g
    return {"stream": req.stream, "budget": req.budget, "threshold": g.threshold, "frozen": g.frozen, "n": len(req.scores)}


@app.post("/v1/monitor/action")
def monitor_action(req: ActionRequest):
    path = os.environ.get("GB_MONITOR_PROBE")
    if not path:
        raise HTTPException(503, "set GB_MONITOR_PROBE to reader_probe.joblib (needs a GPU and torch)")
    t0 = time.perf_counter()
    with _READER_LOCK:            # the first load is inside the lock: concurrent first requests raced on the lazy imports
        if "reader" not in _STATE:
            from .reader_load import load_reader
            _STATE["reader"] = load_reader(path)
            t0 = time.perf_counter()
        rm = _STATE["reader"]
        if req.context is None and getattr(rm, "chat", False):
            raise HTTPException(422, "this probe was trained with conversation-aware context (--chat): send 'context' (graybox.chat.chat_ctx)")
        if req.context is not None:
            p = float(rm.probe.predict_proba(rm.features(Q.format(ctx=req.context, cot="", act=act_text(req.action.tool, req.action.args))))[0, 1])
        else:
            p = rm.score(req.instructions, req.task, _hist(req), (req.action.tool, req.action.args))
        g = _STATE["gates"].setdefault((req.stream, req.budget), QuantileGate(req.budget))
        t = g.threshold
        esc = g(p)
    return {"p_unsafe": round(p, 6), "escalate": esc, "threshold": t, "calibrated": t is not None, "frozen": g.frozen,
            "ms": round(1000 * (time.perf_counter() - t0), 1),
            "note": "p_unsafe ranks actions but is not calibrated on a new agent; act on 'escalate'"}
