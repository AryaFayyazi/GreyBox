"""Grey Box API: estimate and stream the latent trajectory of a black-box model from queries alone.

The server only ever calls the black box through its query interface (`BlackBox.query`). Ground truth (true hidden
states, true board) is attached ONLY for open models, as an overlay that lets users check the estimate.

Endpoints
  GET  /v1/models                  available black boxes, their batteries and held-out validation metrics
  POST /v1/runs                    run a sequence through a black box; returns run_id
  GET  /v1/runs/{id}               full run (frame metadata + per-step events)
  GET  /v1/runs/{id}/stream        per-step SSE replay (Last-Event-ID resumable)
  GET  /                           3D viewer
Run:  uvicorn graybox.server:app --port 8765   (research viewer; needs the [research] dependencies: torch, transformers)
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .othello import SQ, TOK, Board, OthelloGPT, board_to_relative, boards_along, random_game

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "web"
MODELS = Path("/data/models/graybox_ws/models")
CHI2_3_95 = 7.815
app = FastAPI(title="Grey Box", version="0.2.0")
RUNS: dict[str, dict] = {}


def mlp(n_in, n_out):
    return torch.nn.Sequential(torch.nn.Linear(n_in, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(512, n_out))


class OthelloBlackBox:
    """Othello-GPT behind a query-only interface. `query(prefix, suffixes)` returns next-move distributions."""
    name = "othello-gpt"
    battery_desc = "next-move distribution, plus the 60 single-move suffixes (61 queries per step)"

    def __init__(self):
        self.model = OthelloGPT.load()
        ck = torch.load(MODELS / "othello/demo.pt", weights_only=False)
        self.ck = ck
        self.psi2h, self.psi2b = mlp(ck["n_in"], ck["npc"]), mlp(ck["n_in"], 192)
        self.psi2h.load_state_dict(ck["psi2h"])
        self.psi2b.load_state_dict(ck["psi2b"])
        self.wb = torch.nn.Linear(512, 192)
        self.wb.load_state_dict(ck["wb"])
        for m in (self.psi2h, self.psi2b, self.wb):
            m.eval()
        self.n_queries = 0

    # ---- the ONLY access path the estimator uses -------------------------------------------------------------
    @torch.no_grad()
    def query(self, moves: list[int]) -> np.ndarray:
        """Black-box call: for the game prefix `moves`, return 61 next-move distributions (each over 60 squares):
        row 0 = no suffix, row k = after appending square-token k."""
        ids = torch.tensor([[TOK[s] for s in moves]])
        lg, _, kv = self.model.forward_cache(ids)
        t = len(moves) - 1
        p0 = torch.softmax(lg[0, t, 1:], -1)
        p1 = torch.softmax(self.model.extend(kv, [t] * 60, list(range(1, 61)))[:, 1:], -1)
        self.n_queries += 61
        return torch.cat([p0[None], p1]).numpy()

    # ---- ground truth, open models only (overlay) ---------------------------------------------------------------
    @torch.no_grad()
    def truth_hidden(self, moves: list[int]) -> np.ndarray:
        _, rs, _ = self.model.forward_cache(torch.tensor([[TOK[s] for s in moves]]))
        return rs[self.ck["layer"]][0].numpy()          # (T, 512): residual after every prefix


class RunRequest(BaseModel):
    model: str = Field("othello-gpt")
    moves: list[int] | None = Field(None, description="board squares 0..63; omitted = a random legal game")
    seed: int = Field(0, ge=0, le=1_000_000)
    n_samples: int = Field(0, ge=0, le=256, description="0 = logprob tier; n>0 = text-only tier with n samples/query")
    uncertainty_reps: int = Field(12, ge=0, le=64, description="sampled replicas used for the uncertainty ellipsoid")
    gauge_rotation: bool = Field(False, description="rotate the model's hidden space by a random orthogonal matrix")


BOXES: dict[str, OthelloBlackBox] = {}


def box(name):
    if name not in BOXES:
        if name != "othello-gpt":
            raise HTTPException(404, f"unknown model {name}")
        BOXES[name] = OthelloBlackBox()
    return BOXES[name]


def features(P: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.clip(P, 0, 1)).reshape(-1)        # categorical Fisher-Rao embedding


def build_run(req: RunRequest) -> dict:
    bb = box(req.model)
    ck = bb.ck
    rng = np.random.default_rng(req.seed)
    moves = (req.moves or random_game(rng))[:58]       # a 1-token query after move 58 still fits the 59-token context
    boards = boards_along(moves)
    H = bb.truth_hidden(moves)
    G = np.linalg.qr(rng.normal(size=(512, 512)))[0] if req.gauge_rotation else np.eye(512)
    comps, hmu = ck["comps"], ck["hmu"]
    dmu, D, scale = ck["disp_mu"], ck["disp_dirs"], ck["disp_scale"]
    to3 = lambda pc64: ((pc64 - dmu) @ D) * scale  # noqa: E731  same 64->3 map for estimate and truth
    events, q0 = [], bb.n_queries
    for t in range(len(moves)):
        P = bb.query(moves[: t + 1])
        X = features(P)

        def decode(x):
            xs = torch.tensor(((x - ck["xmu"]) / ck["xsd"])[None], dtype=torch.float32)
            with torch.no_grad():
                return bb.psi2h(xs)[0].numpy(), torch.softmax(bb.psi2b(xs).view(64, 3), -1).numpy()

        if req.n_samples > 0:
            samp = lambda: features(np.stack([rng.multinomial(req.n_samples, p / p.sum()) / req.n_samples  # noqa: E731
                                              for p in P.astype(np.float64)]))
            X = samp()
        h_est, board_p = decode(X)
        reps = []
        for _ in range(req.uncertainty_reps if req.n_samples > 0 else 0):
            reps.append(to3(decode(samp())[0]))
        cov = np.cov(np.array(reps).T) if len(reps) > 3 else np.zeros((3, 3))
        w, U = np.linalg.eigh(cov)
        h_true = H[t]
        pc_true = (h_true - hmu) @ comps.T
        # white-box readout uses the KNOWN basis; under a gauge rotation the model computes the same function, but
        # the residual a white-box tool reads is h G, so a probe fit in the original basis reads the wrong directions
        h_read = h_true @ G
        with torch.no_grad():
            wb_p = torch.softmax(bb.wb(torch.tensor(((h_read - hmu) / ck["hsd"])[None], dtype=torch.float32))
                                 .view(64, 3), -1).numpy()
        b, player, legal = boards[t]
        rel = board_to_relative(b, player)
        events.append({
            "seq": t, "t": t, "move": int(moves[t]), "player_to_move": int(player),
            "point_est": to3(h_est).round(5).tolist(),
            "point_true": to3(pc_true).round(5).tolist(),
            "ellipsoid": {"radii": np.sqrt(np.clip(w, 0, None) * CHI2_3_95).round(5).tolist(),
                          "axes": U.T.round(5).tolist()},
            "board_true": rel.tolist(),
            "board_est": board_p.argmax(-1).tolist(),
            "board_est_conf": board_p.max(-1).round(3).tolist(),
            "board_whitebox": wb_p.argmax(-1).tolist(),
            "legal": [int(s) for s in legal],
            "acc_est_occupied": round(float((board_p.argmax(-1) == rel)[rel > 0].mean()) if (rel > 0).any() else 1.0, 4),
            "acc_whitebox_occupied": round(float((wb_p.argmax(-1) == rel)[rel > 0].mean()) if (rel > 0).any() else 1.0, 4),
            "queries_so_far": int(bb.n_queries - q0),
        })
    run_id = uuid.uuid4().hex[:12]
    frame = {
        "run_id": run_id, "model": req.model, "battery": bb.battery_desc,
        "tier": "logprobs" if req.n_samples == 0 else f"text only, {req.n_samples} samples",
        "frame": f"the three main directions of the true layer-{ck['layer']} hidden state (they hold "
                 f"{100 * ck['disp_var_explained']:.0f}% of its variance after the opening move). The estimate is "
                 "placed in the same frame by a decoder checked on held-out games",
        "validation": ck["metrics"], "gauge_rotation": req.gauge_rotation,
        "cloud": ck["disp_cloud"][:2500].round(4).tolist(),
        "cloud_t": ck["disp_cloud_t"][:2500].tolist(),
        "n_steps": len(moves), "seed": req.seed,
        "identified": ["behaviourally visible state (61-query battery)", "its decoded board",
                       "hidden-vector projection via a validated decoder"],
        "not_identified": ["raw hidden coordinates without the reference frame (GL(d) gauge)",
                           "state that no query in the battery can reveal"],
    }
    return {"run_id": run_id, "frame": frame, "events": events}


# ------------------------------------------------------------------------------------------- Grey Box Live
from .adapters import Anthropic, OpenAICompatible  # noqa: E402
from .live import LiveSession, Readouts  # noqa: E402

LIVE: dict[str, LiveSession] = {}


class LiveRequest(BaseModel):
    provider: str = Field("openai_compat", description="openai_compat | anthropic")
    model: str
    base_url: str | None = Field(None, description="OpenAI-compatible base URL (OpenAI, DeepSeek, vLLM, ...)")
    key_env: str | None = Field(None, description="NAME of the environment variable holding the key (never the key)")
    logprobs: bool = True
    n_samples: int = Field(8, ge=1, le=64)
    extra: dict = Field(default_factory=dict)
    open_model: str | None = Field(None, description="L2 name of an OPEN model (e.g. qwen3_8b) to overlay its true hidden state")


class TurnRequest(BaseModel):
    message: str


@app.post("/v1/live/sessions")
def live_session(req: LiveRequest):
    if req.provider == "anthropic":
        led = None
        if os.environ.get("GB_SPEND_LEDGER"):                # shared hard cap across all processes (graybox.budget)
            from .budget import Ledger
            led = Ledger(os.environ["GB_SPEND_LEDGER"], float(os.environ.get("GB_SPEND_CAP", "4.0")),
                         float(os.environ.get("GB_PRICE_IN", "1e-7")), float(os.environ.get("GB_PRICE_OUT", "5e-7")))
        ad = Anthropic(model=req.model, key_env=req.key_env or "ANTHROPIC_API_KEY", n_samples=req.n_samples, ledger=led)
    else:
        ad = OpenAICompatible(model=req.model, base_url=req.base_url or "https://api.openai.com/v1",
                              key_env=req.key_env, logprobs=req.logprobs, n_samples=req.n_samples, extra=req.extra)
    truth = None
    if req.open_model:
        from .live import OpenTruth
        r0 = Readouts.load()
        truth = OpenTruth(req.open_model, battery=r0.battery if r0 else "v2")
    s = LiveSession(adapter=ad, readouts=Readouts.load(), truth=truth, model_key=req.open_model or "default")
    LIVE[s.id] = s
    r = s.readouts
    return {"session_id": s.id, "model_key": s.model_key, "battery_size": len(s.battery()), "questions": dict(s.battery()),
            "concepts": [{"name": c["name"], "kind": c["kind"], "validation": c["validation"],
                          "threshold": r.threshold(c, s.model_key)} for c in (r.concepts if r else [])],
            "cloud": r.cloud.round(4).tolist() if r else [], "cloud_label": r.cloud_label if r else []}


@app.post("/v1/live/sessions/{sid}/turns")
async def live_turn(sid: str, req: TurnRequest):
    if sid not in LIVE:
        raise HTTPException(404, "unknown session")
    return await LIVE[sid].turn(req.message)


# ------------------------------------------------------------------------------------------- reader action monitor
class MonitorStep(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    result: str = ""


class MonitorRequest(BaseModel):
    instructions: str = Field(..., description="the agent's operating instructions (system prompt)")
    task: str = Field(..., description="the user's task")
    history: list[MonitorStep] = Field(default_factory=list, description="earlier tool calls with their results")
    action: MonitorStep = Field(..., description="the proposed action (tool '__final__' for a final answer)")
    stream: str = Field("default", description="deployment id; each stream calibrates its own escalation threshold")
    budget: float = Field(0.05, gt=0, lt=1, description="fraction of this stream's actions to escalate")


_MONITOR, _GATES = {}, {}


@app.post("/v1/monitor/action")
def monitor_action(req: MonitorRequest):
    """P(unsafe) of executing the proposed action, from the transcript alone (reader monitor; preregistered T4/T5)."""
    from .monitor import ReaderMonitor, Step
    if "m" not in _MONITOR:
        path = os.environ.get("GB_MONITOR_PROBE", "/data/models/graybox_ws/models/monitor/reader_probe.joblib")
        _MONITOR["m"] = ReaderMonitor.load(path)
    m = _MONITOR["m"]
    from .monitor import QuantileGate
    p = m.score(req.instructions, req.task, [Step(h.tool, h.args, h.result) for h in req.history], (req.action.tool, req.action.args))
    g = _GATES.setdefault((req.stream, req.budget), QuantileGate(req.budget))
    t = g.threshold
    return {"p_unsafe": round(p, 6), "escalate": g(p), "threshold": t, "calibrated": t is not None,
            "note": "p_unsafe ranks actions but is not calibrated on a new agent; act on 'escalate' (label-free quantile gate)",
            "validated": {"gemma_same_scenarios_auroc": 0.994, "gemma_unseen_families_auroc": 0.969,
                          "closed_a_auroc": 0.991, "closed_a_zero_shot_judge_auroc": 0.633,
                          "source": "preregistered T4, T5, T8b"}}


@app.get("/v1/models")
def models():
    out = {}
    p = MODELS / "othello/demo.pt"
    if p.exists():
        out["othello-gpt"] = {"battery": OthelloBlackBox.battery_desc,
                              "validation": torch.load(p, weights_only=False)["metrics"], "ground_truth_overlay": True}
    return out


@app.post("/v1/runs")
def create_run(req: RunRequest):
    run = build_run(req)
    RUNS[run["run_id"]] = run
    while len(RUNS) > 50:
        RUNS.pop(next(iter(RUNS)))
    return {"run_id": run["run_id"], "n_steps": len(run["events"]),
            "links": {"run": f"/v1/runs/{run['run_id']}", "stream": f"/v1/runs/{run['run_id']}/stream"}}


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")
    return RUNS[run_id]


def _sse(event, data, id_=None):
    return (f"id: {id_}\n" if id_ is not None else "") + f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/v1/runs/{run_id}/stream")
async def stream(run_id: str, request: Request, start: int = 0, pace_ms: int = 350):
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")
    run = RUNS[run_id]
    last = request.headers.get("last-event-id")
    if last and last.isdigit():
        start = int(last) + 1
    pace = max(0, min(pace_ms, 3000)) / 1000

    async def gen():
        yield _sse("frame", run["frame"])
        for ev in run["events"][start:]:
            if await request.is_disconnected():
                return
            yield _sse("step", ev, ev["seq"])
            await asyncio.sleep(pace)
        yield _sse("done", {"run_id": run_id})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


SKELETON = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>')


@app.get("/live")
def live_page():
    return HTMLResponse(SKELETON + (STATIC / "live.html").read_text() + "</body></html>")


@app.get("/")
def index():
    """The viewer page is authored without a document skeleton (so it can also be published as an artifact)."""
    return HTMLResponse(SKELETON + (STATIC / "index.html").read_text() + "</body></html>")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
