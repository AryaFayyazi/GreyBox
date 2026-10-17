"""Grey Box Live: chat with a black-box model and watch its estimated state, turn by turn.

Per user turn:
  1. fork: the conversation + the new user message + each battery question (the model's state as it is about to
     answer; the format the L2 decoders were trained and validated on)
  2. reply: the model's real answer to the unmodified conversation
  3. readout: θ(Ψ) -> a 3D point in a shared, model-agnostic frame; concept scores from decoders validated
     leave-one-family-out on open models; safety flags at fixed, documented thresholds
The readout bundle (scripts/18_live_readouts.py) holds the live battery, the frame and the concept decoders together
with their held-out validation. Without a bundle, only the raw battery answers are returned.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .battery_v2 import BATTERY_V2
from .battery_v2 import PREAMBLE as PREAMBLE_V2
from .battery_v2m import BATTERY_V2M
from .battery_v2m import PREAMBLE as PREAMBLE_V2M

BUNDLE_V1 = Path("/data/models/graybox_ws/models/live/readouts.npz")
BUNDLE_V2 = Path("/data/models/graybox_ws/models/live/readouts_v2.npz")     # scripts/39 (Live benchmark, v2m)
BUNDLE = BUNDLE_V2 if BUNDLE_V2.exists() else BUNDLE_V1


def theta(q):
    return 2.0 * np.arcsin(np.sqrt(np.clip(np.asarray(q, float), 0.0, 1.0)))


@dataclass
class Readouts:
    items: list            # item ids (subset of battery v2) in bundle order
    mu: np.ndarray         # (m,) frame centre in θ space
    dirs: np.ndarray       # (m, 3) frame directions
    scale: float
    concepts: list         # dicts: name, w (m,), b, threshold, validation (dict), kind
    cloud: np.ndarray      # (k, 3) reference points from open models (for the background)
    cloud_label: list
    battery: str = "v2"    # v2 (L2 format) | v2m (Live v2: questions about the LATEST user message)

    @classmethod
    def load(cls, path=BUNDLE):
        if not Path(path).exists():
            return None
        z = np.load(path, allow_pickle=True)
        return cls(items=list(z["items"]), mu=z["mu"], dirs=z["dirs"], scale=float(z["scale"]),
                   concepts=list(z["concepts"]), cloud=z["cloud"], cloud_label=list(z["cloud_label"]),
                   battery=str(z["battery"]) if "battery" in z.files else "v2")

    def preamble(self):
        return PREAMBLE_V2M if self.battery == "v2m" else PREAMBLE_V2

    def threshold(self, c, model_key):
        """Per-model calibrated threshold (v2 bundles) or one fixed threshold (v1)."""
        t = c["threshold"]
        return float(t.get(model_key, t["default"])) if isinstance(t, dict) else float(t)


class OpenTruth:
    """Ground-truth overlay for an OPEN model: its true hidden state at the token the battery forks from, plus the
    decoder that maps the live battery answers into the same frame (scripts/22). Needs the weights and a GPU."""

    def __init__(self, l2_name: str, battery: str = "v2"):
        import sys
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        import importlib
        run12 = importlib.import_module("12_l2_run")
        self.cfg = run12.MODELS[l2_name]
        path = run12.path_of(self.cfg["hf"])
        self.tok = AutoTokenizer.from_pretrained(path)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        except Exception:
            self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        tag = "open_v2m_" if battery == "v2m" else "open_"                # scripts/40 (v2m) or scripts/22 (v2)
        self.d = np.load(Path("/data/models/graybox_ws/models/live") / f"{tag}{l2_name}.npz")
        self.torch = torch

    def points(self, convo: list, theta_items: np.ndarray):
        d = self.d
        s = self.tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True, **self.cfg["kw"])
        ids = self.tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        with self.torch.no_grad():
            hs = self.model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        h = hs[int(d["layer"])][0, -1].float().cpu().numpy()
        true_pc = (h - d["mu"]) @ d["comps"].T
        est_pc = ((theta_items - d["xm"]) / d["xs"]) @ d["coef"].T + d["intercept"]
        k = float(d["scale"])
        return (est_pc[:3] * k).round(4).tolist(), (true_pc[:3] * k).round(4).tolist(), json.loads(str(d["check"]))


@dataclass
class LiveSession:
    adapter: object
    readouts: Readouts | None
    truth: OpenTruth | None = None
    model_key: str = "default"     # which calibrated threshold to use (an open model's L2 name, or 'default')
    messages: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        if self.readouts is not None and hasattr(self.adapter, "preamble"):
            self.adapter.preamble = self.readouts.preamble()

    def battery(self):
        if self.readouts is None:
            return list(BATTERY_V2)
        keep = set(self.readouts.items)
        base = BATTERY_V2M if self.readouts.battery == "v2m" else BATTERY_V2
        return [(k, q) for k, q in base if k in keep]

    async def turn(self, user_message: str) -> dict:
        t0 = time.time()
        convo = self.messages + [{"role": "user", "content": user_message}]
        items = self.battery()
        answers = await self.adapter.battery([convo], items)
        q = np.array([a[0] for a in answers], float)
        nobs = np.array([a[1] for a in answers], float)
        reply = await self.adapter.reply(convo)
        self.messages = convo + [{"role": "assistant", "content": reply}]
        th = theta(np.nan_to_num(q, nan=0.5))
        ev = {"turn": len(self.turns), "user": user_message, "reply": reply,
              "answers": {k: (None if not np.isfinite(v) else round(float(v), 4)) for (k, _), v in zip(items, q)},
              "tier": answers[0][2] if answers else None, "seconds": round(time.time() - t0, 2),
              "usage": vars(self.adapter.usage).copy()}
        if self.readouts is not None:
            r = self.readouts
            order = [r.items.index(k) for k, _ in items]
            x = np.zeros(len(r.items))
            x[order] = th
            ev["point"] = (((x - r.mu) @ r.dirs) * r.scale).round(4).tolist()
            n_eff = np.where(np.isfinite(nobs), nobs, np.inf)
            var = np.where(np.isinf(n_eff), 0.0, 1.0 / np.maximum(n_eff, 1))      # Var θ̂_j ≈ 1/n (T4)
            cov = (r.dirs.T * var) @ r.dirs * r.scale ** 2
            w, U = np.linalg.eigh(cov)
            ev["ellipsoid"] = {"radii": np.sqrt(np.clip(w, 0, None) * 7.815).round(4).tolist(), "axes": U.T.round(4).tolist()}
            ev["concepts"] = []
            for c in r.concepts:
                z = float(x @ c["w"] + c["b"])
                p = 1 / (1 + np.exp(-z))
                thr = r.threshold(c, self.model_key)
                ev["concepts"].append({"name": c["name"], "score": round(p, 4), "flag": bool(p >= thr),
                                       "threshold": thr, "validation": c["validation"], "kind": c["kind"]})
            if self.truth is not None:
                order_items = [k for k, _ in items]
                th_items = np.array([th[order_items.index(k)] for k in self.truth.d["items"]])
                est3, true3, check = self.truth.points(convo, th_items)
                ev["hidden_est"], ev["hidden_true"], ev["hidden_check"] = est3, true3, check
        self.turns.append(ev)
        return ev
