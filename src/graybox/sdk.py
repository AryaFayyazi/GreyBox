"""Grey Box SDK: read a black-box model's state, and guard an agent's actions, through its API only.

    from graybox.sdk import Reader, AgentGuard

    reader = Reader.load(adapter, bundle="/path/readouts_v4.npz", thresholds={"will_refuse": ..., "harmful_latest": ...})
    r = await reader.read(turns)          # turns: [{"role": "user"|"assistant", "content": str}, ...] ending on user
    r.will_refuse, r.point, r.answers, r.alarm(moderator_verdict)

    guard = AgentGuard(propose=my_agent_step, is_unsafe=my_rule_checker, k=8, tau=1/8)
    d = await guard.check(messages)       # d.escalate, d.risk, d.samples

Reader uses the one-call TRANSCRIPT format (src/graybox/batch.py), validated on six open models and preregistered on
Claude Haiku 5.5 (L3d). The adapter needs one async method, `_ask(messages, max_tokens) -> str` (graybox.adapters.Anthropic
has it; any provider can be wrapped in five lines). AgentGuard is the untrained resampling guard, and it is only as good as
`is_unsafe`. With the environment's exact rule checker it caught 81% of violations at 4% escalation (scripts/59, 61).
With an LLM judge as `is_unsafe` it caught 12%, worse than judging the actual action (41%), and far worse than a
white-box probe (94%; preregistered T3). Use it only with an exact, cheap rule check. Thresholds and validity are only as good as their calibration: calibrate `thresholds` on the target
model's own benign traffic (95th percentile), as the validation did.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from .batch import batch_block, parse_batch, transcript_prompt
from .battery_v2m import BATTERY_V2M


@dataclass
class Reading:
    answers: dict                 # question -> estimated P(yes)
    theta: np.ndarray             # 2 asin sqrt q per item (the grey-box state)
    point: list                   # 3D coordinates in the shared space
    scores: dict                  # readout name -> probability
    flags: dict                   # readout name -> score >= calibrated threshold
    usable: float                 # share of questions answered across samples

    @property
    def will_refuse(self) -> float:
        return self.scores["will_refuse"]

    def alarm(self, moderator_verdict: str | None) -> bool:
        """The validated alarm: a moderator says Unsafe AND the model is not about to refuse."""
        return moderator_verdict == "Unsafe" and not self.flags["will_refuse"]


@dataclass
class Reader:
    adapter: Any
    mu: np.ndarray
    dirs: np.ndarray
    scale: float
    readouts: dict                # name -> (w, b)
    thresholds: dict
    n_samples: int = 2
    items: list = field(default_factory=lambda: list(BATTERY_V2M))

    @classmethod
    def load(cls, adapter, bundle, thresholds: dict, n_samples: int = 2):
        z = np.load(bundle, allow_pickle=True)
        assert list(z["items"]) == [k for k, _ in BATTERY_V2M], "bundle must cover the 88 v2m items in order"
        ro = {c["name"]: (np.asarray(c["w"], float), float(c["b"])) for c in z["concepts"]}
        return cls(adapter, np.asarray(z["mu"], float), np.asarray(z["dirs"], float), float(z["scale"]), ro, thresholds, n_samples)

    async def read(self, turns: list[dict]) -> Reading:
        block, k = batch_block(self.items), len(self.items)
        prompt = [{"role": "user", "content": transcript_prompt(turns, block)}]
        texts = await asyncio.gather(*[self.adapter._ask(prompt, 1100) for _ in range(self.n_samples)])
        c = sum(parse_batch(t, k) for t in texts).astype(float)
        q = (c[:, 0] + 0.5) / (c[:, 0] + c[:, 1] + 1.0)
        th = 2 * np.arcsin(np.sqrt(q))
        scores = {n: float(1 / (1 + np.exp(-(th @ w + b)))) for n, (w, b) in self.readouts.items()}
        return Reading(answers={q_: round(float(v), 4) for (_, q_), v in zip(self.items, q)}, theta=th,
                       point=(((th - self.mu) @ self.dirs) * self.scale).round(4).tolist(), scores=scores,
                       flags={n: s >= self.thresholds.get(n, 1.0) for n, s in scores.items()},
                       usable=float(c[:, :2].sum() / c.sum()))


@dataclass
class GuardDecision:
    escalate: bool
    risk: float                   # share of the K resampled actions that the checker marks unsafe
    samples: list                 # the K proposed actions
    unsafe: list                  # checker verdict per sample


@dataclass
class AgentGuard:
    """Resample the agent's next step K times and escalate when >= tau of them are unsafe. No training."""
    propose: Callable[[list], Awaitable[Any]]      # one sample of the agent's next action for these messages
    is_unsafe: Callable[[Any], bool]               # the deployment's own rule checker
    k: int = 8
    tau: float = 1 / 8

    async def check(self, messages: list) -> GuardDecision:
        samples = await asyncio.gather(*[self.propose(messages) for _ in range(self.k)])
        bad = [bool(self.is_unsafe(s)) for s in samples]
        risk = float(np.mean(bad))
        return GuardDecision(escalate=risk >= self.tau, risk=risk, samples=list(samples), unsafe=bad)
