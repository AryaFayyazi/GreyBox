"""A monitor on a browser-size reader (graybox.embed), with the same interface as TextMonitor, so Guard and the server
use it unchanged. The model file is JSON, the same file the browser trainer writes:

  {"type": "embed", "encoder": "minilm", "mean": [...], "scale": [...], "w": [...], "b": ..., "threshold_logit": ...,
   "budget": 0.05, "chat": false}

score = b + w . (x - mean) / scale, x = graybox.embed.Encoder(encoder).encode(ctx, act). Act on threshold_logit (the
label-free cut: the (1 - budget) quantile of the scores of the agent's own actions).
"""
from __future__ import annotations

import json

import numpy as np

from .embed import ENCODERS, Encoder
from .monitor import act_text, ctx_text


class EmbedMonitor:
    def __init__(self, d: dict, device: str = "cpu", cache_dir: str | None = None):
        assert d.get("type") == "embed" and d["encoder"] in ENCODERS, "not an embed-monitor file"
        self._json, self.device, self.cache_dir = d, device, cache_dir
        self.mean, self.scale, self.w = (np.asarray(d[k], np.float64) for k in ("mean", "scale", "w"))
        self.b = float(d["b"])
        self._enc = None

    @classmethod
    def load_json(cls, path, **kw) -> "EmbedMonitor":
        return cls(json.loads(open(path).read()), **kw)

    @property
    def encoder(self) -> Encoder:
        if self._enc is None:
            self._enc = Encoder(self._json["encoder"], self.device, self.cache_dir, batch=1)
        return self._enc

    def logits(self, ctxs, acts) -> np.ndarray:
        X = self.encoder.encode(list(ctxs), list(acts)).astype(np.float64)
        return self.b + ((X - self.mean) / self.scale) @ self.w

    def decision(self, instructions, task, history, action, *, rendered=None) -> float:
        ctx, act = rendered or (ctx_text(instructions, task, history), act_text(*action))
        return float(self.logits([ctx], [act])[0])

    def escalate(self, instructions, task, history, action) -> bool:
        return self.decision(instructions, task, history, action) >= self._json["threshold_logit"]


def to_json(encoder: str, probe, threshold_logit: float, budget: float = 0.05, **meta) -> dict:
    """A fitted sklearn pipeline StandardScaler -> LogisticRegression (graybox.embed.fit_probe) as an embed-monitor file."""
    sc, lr = probe.steps[0][1], probe.steps[-1][1]
    return {"type": "embed", "encoder": encoder, "web": ENCODERS[encoder]["web"], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "w": lr.coef_[0].tolist(), "b": float(lr.intercept_[0]), "threshold_logit": float(threshold_logit), "budget": budget, **meta}
