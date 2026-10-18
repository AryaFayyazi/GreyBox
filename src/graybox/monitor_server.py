"""Minimal self-hosted monitoring API (FastAPI). Starts without torch; transcripts never leave the machine.

    GB_TEXT_MONITOR=greybox-text-monitor.json uvicorn graybox.monitor_server:app      # CPU, real time
    GB_MONITOR_PROBE=reader_probe.joblib     uvicorn graybox.monitor_server:app       # + GPU reader

POST /v1/monitor/text    the text monitor (browser trainer or python -m graybox.train_text; escalation cut from its file).
                         Also a browser-reader file (graybox-embed-monitor/1; graybox.embed_monitor).
POST /v1/monitor/action  the reader monitor (Qwen3-8B + probe), with a frozen QuantileGate per stream
GET  /v1/health

Security: bind to localhost unless you set GB_API_KEY; with it, every endpoint but /v1/health needs the header
"Authorization: Bearer <GB_API_KEY>". Requests carry whole transcripts, which can hold credentials.
Gates: GB_GATE_STATE=/path/gates.json keeps each stream's gate (warm-up scores, frozen cut) across restarts; without it,
a restarted server escalates every action of a stream until it has seen 100 again (fail closed). A gate belongs to a
(stream, budget) pair: a request with another budget starts a new, cold gate, and /calibrate seeds one budget.
"""
from __future__ import annotations

import contextlib
import hmac
import json
import os
import threading
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .monitor import Q, QuantileGate, Step, TextMonitor, act_text

from . import __version__  # noqa: E402

_STATE: dict = {"gates": {}}
_READER_LOCK = threading.Lock()      # one forward pass at a time on the GPU model
_GATE_LOCK = threading.Lock()


def _auth(request: Request):
    """With GB_API_KEY set, every endpoint but /v1/health needs "Authorization: Bearer <key>"."""
    key = os.environ.get("GB_API_KEY")
    if not key or request.url.path == "/v1/health":
        return
    got = request.headers.get("authorization", "")
    if not hmac.compare_digest(got.encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "missing or wrong API key: send 'Authorization: Bearer <GB_API_KEY>'")


def _gate_key(stream: str, budget: float) -> str:
    return f"{stream}|{budget!r}"


def _load_gates():
    path = os.environ.get("GB_GATE_STATE")
    if path and os.path.exists(path):
        d = json.loads(open(path).read())
        _STATE["gates"] = {k: QuantileGate.from_dict(v) for k, v in d.items()}


def _save_gates():
    """Write every gate atomically (GB_GATE_STATE); called when a gate is created, freezes or is calibrated, every 50
    gated requests, and at shutdown."""
    path = os.environ.get("GB_GATE_STATE")
    if not path:
        return
    with _GATE_LOCK:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({k: g.to_dict() for k, g in _STATE["gates"].items()}, f)
        os.replace(tmp, path)


def _gate(stream: str, budget: float) -> QuantileGate:
    k = _gate_key(stream, budget)
    if k not in _STATE["gates"]:
        _STATE["gates"][k] = QuantileGate(budget)
        _save_gates()
    return _STATE["gates"][k]


def _observe(g: QuantileGate, score: float) -> bool:
    was = g.frozen
    esc = bool(g(score))
    _STATE["gated"] = _STATE.get("gated", 0) + 1
    if g.frozen != was or _STATE["gated"] % 50 == 0:
        _save_gates()
    return esc


@contextlib.asynccontextmanager
async def _lifespan(app):
    _load_text()                      # fail fast: a bad GB_TEXT_MONITOR shows in /v1/health, not at the first request
    _load_gates()
    yield
    _save_gates()


app = FastAPI(title="Grey Box monitor", version=__version__, lifespan=_lifespan, dependencies=[Depends(_auth)])


@app.middleware("http")
async def _auth_docs(request: Request, call_next):
    """The API key also guards the generated docs (/docs, /redoc, /openapi.json), which route dependencies do not cover."""
    if request.url.path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
        try:
            _auth(request)
        except HTTPException as e:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": e.detail}, status_code=e.status_code)
    return await call_next(request)


class StepIn(BaseModel):
    model_config = ConfigDict(extra="forbid")      # a misspelt field is an error, not silently ignored
    tool: str = Field(..., min_length=1, description="the tool name (an empty name is refused: Guard blocks such calls too)")
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
    budget: float | None = Field(None, gt=0, lt=1, description="fraction of this stream's actions to escalate: the reader's gate "
                                 "(default 0.05), or the text monitor's live gate (needs gate=true; default 0.05)")
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


@app.get("/v1/health")
def health():
    _load_text()
    return {"version": __version__, "text_monitor": "text" in _STATE, "text_monitor_error": _STATE.get("text_error"),
            "reader_monitor": bool(os.environ.get("GB_MONITOR_PROBE")), "reader_loaded": "reader" in _STATE,
            "auth_required": bool(os.environ.get("GB_API_KEY")), "gate_state_persisted": bool(os.environ.get("GB_GATE_STATE")),
            "gates": len(_STATE["gates"])}


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
    if req.budget is not None and not req.gate:
        raise HTTPException(422, "'budget' applies only with gate=true; without it the file's own cut is used")
    warn = ("this monitor was not trained on conversation-aware contexts: 'context' changes its input"
            if req.context is not None and not tm._json.get("chat") else None)
    if req.context is not None:
        z = tm.decision(None, None, None, None, rendered=(req.context, act_text(req.action.tool, req.action.args)))
    else:
        z = tm.decision(req.instructions, req.task, _hist(req), (req.action.tool, req.action.args))
    if req.gate:
        budget = req.budget or 0.05
        g = _gate("text:" + req.stream, budget)
        thr = g.threshold
        esc = _observe(g, z)
    else:
        esc, thr, budget = z >= tm._json["threshold_logit"], tm._json["threshold_logit"], tm._json.get("budget")
    return {"score": round(z, 4), "unit": "logit", "escalate": esc, "threshold": None if thr is None else round(thr, 4),
            "cut": "live gate on this stream" if req.gate else "the file's cut",
            "budget": budget, "ms": round(1000 * (time.perf_counter() - t0), 3), "warning": warn,
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
    _STATE["gates"][_gate_key(("text:" if req.monitor == "text" else "") + req.stream, req.budget)] = g
    _save_gates()
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
        g = _gate(req.stream, req.budget or 0.05)
        t = g.threshold
        esc = _observe(g, p)
    return {"p_unsafe": round(p, 6), "escalate": esc, "threshold": t, "calibrated": t is not None, "frozen": g.frozen,
            "ms": round(1000 * (time.perf_counter() - t0), 1),
            "note": "p_unsafe ranks actions but is not calibrated on a new agent; act on 'escalate'"}
