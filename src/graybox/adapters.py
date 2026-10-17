"""Black-box adapters: how the grey box talks to a model it cannot open (L3).

Every adapter answers one call: yes_prob(context, question) -> (q_hat, n_obs, tier), the estimated
P(Yes | context + question). Tiers:
  logprob   the API returns top-k logprobs of the first answer token: q is read in one call (n_obs = inf)
  sample    text only: n sampled one-word answers; q_hat = (#yes)/(#yes + #no); Var(2 arcsin sqrt q_hat) ~ 1/n
Keys come from environment variables only and are never logged or stored.
"""
from __future__ import annotations

import asyncio
import math
import os
import random
from dataclasses import dataclass, field

import httpx

from .battery_v2 import PREAMBLE


def fork(context, question: str, preamble: str = PREAMBLE) -> list[dict]:
    """The interrogation fork: the conversation up to the latest user message, with the question appended to that
    message (the same format the decoders were trained on). `context` is a string (one user message) or a list of
    chat messages whose last element is the user's message. `preamble`: battery_v2.PREAMBLE (L2) or
    battery_v2m.PREAMBLE (Live v2: the question is about the user's LATEST message only)."""
    if isinstance(context, str):
        return [{"role": "user", "content": context + preamble + question}]
    msgs = [dict(m) for m in context]
    assert msgs and msgs[-1]["role"] == "user", "fork expects the conversation to end with the user's message"
    msgs[-1]["content"] = msgs[-1]["content"] + preamble + question
    return msgs


def _classify(text: str) -> str | None:
    t = (text or "").strip().strip(".!*\"'").lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return None


BLOCKED = "[output blocked by the provider's content filter]"


@dataclass
class Usage:
    blocked: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    unparsed: int = 0


@dataclass
class OpenAICompatible:
    """OpenAI Chat Completions schema: OpenAI, DeepSeek, xAI, Together, OpenRouter, vLLM servers."""
    model: str
    base_url: str = "https://api.openai.com/v1"
    key_env: str | None = "OPENAI_API_KEY"
    logprobs: bool = True             # provider returns top_logprobs for the answer token
    n_samples: int = 16               # used when logprobs is False (or as a check)
    concurrency: int = 16
    extra: dict = field(default_factory=dict)
    preamble: str = PREAMBLE
    usage: Usage = field(default_factory=Usage)

    def _headers(self):
        key = os.environ.get(self.key_env) if self.key_env else None
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def _call(self, client, body):
        for attempt in range(5):
            try:
                r = await client.post("/chat/completions", json=body, timeout=120)
                if r.status_code == 429 or r.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                j = r.json()
                u = j.get("usage") or {}
                self.usage.calls += 1
                self.usage.input_tokens += u.get("prompt_tokens", 0)
                self.usage.output_tokens += u.get("completion_tokens", 0)
                return j
            except httpx.HTTPError:
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("provider call failed after retries")

    async def yes_prob(self, client, sem, context, question: str):
        msgs = fork(context, question, self.preamble)
        async with sem:
            if self.logprobs:
                j = await self._call(client, {"model": self.model, "messages": msgs, "max_tokens": 1, "temperature": 1.0,
                                              "logprobs": True, "top_logprobs": 20, **self.extra})
                top = j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                py = sum(math.exp(t["logprob"]) for t in top if _classify(t["token"]) == "yes")
                pn = sum(math.exp(t["logprob"]) for t in top if _classify(t["token"]) == "no")
                if py + pn == 0:
                    self.usage.unparsed += 1
                    return float("nan"), 0, "logprob"
                return py / (py + pn), math.inf, "logprob"
            j = await self._call(client, {"model": self.model, "messages": msgs, "max_tokens": 2, "temperature": 1.0,
                                          "n": self.n_samples, **self.extra})
            lab = [_classify(c["message"].get("content", "")) for c in j["choices"]]
            y, n = lab.count("yes"), lab.count("no")
            self.usage.unparsed += len(lab) - y - n
            return (y / (y + n) if y + n else float("nan")), y + n, "sample"

    async def reply(self, messages: list[dict], max_tokens: int = 600) -> str:
        """The model's actual answer to the conversation (never altered by the interrogation forks)."""
        async with httpx.AsyncClient(base_url=self.base_url, headers=self._headers()) as client:
            j = await self._call(client, {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                          "temperature": 0.7, **self.extra})
        return j["choices"][0]["message"].get("content") or ""

    async def battery(self, contexts: list, battery: list[tuple[str, str]]):
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(base_url=self.base_url, headers=self._headers()) as client:
            tasks = [self.yes_prob(client, sem, c, q) for c in contexts for _, q in battery]
            return await asyncio.gather(*tasks)


@dataclass
class Anthropic:
    """Anthropic Messages API through the official SDK. No logprobs, so the text-only tier: n sampled one-word answers
    per battery question (thinking disabled, low effort, max 3 tokens). Every call is charged to an optional shared
    spend ledger (graybox.budget.Ledger), which raises once the hard cap is reached. The key comes from the
    environment variable `key_env` and is passed to the SDK only."""
    model: str
    key_env: str = "ANTHROPIC_API_KEY"
    n_samples: int = 8
    concurrency: int = 12
    preamble: str = PREAMBLE
    ledger: object = None
    usage: Usage = field(default_factory=Usage)
    _client: object = None

    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=os.environ.get(self.key_env, ""), max_retries=5)
        return self._client

    async def _ask(self, messages, max_tokens):
        if self.ledger is not None:
            self.ledger.check()
        import anthropic
        for attempt in range(12):                          # rate limits / overload: back off and retry (SDK retries
            try:                                           # only briefly)
                r = await self.client().messages.create(model=self.model, max_tokens=max_tokens, messages=messages,
                                                        thinking={"type": "disabled"}, output_config={"effort": "low"})
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError):
                if attempt == 11:
                    raise
                await asyncio.sleep(min(90.0, 5.0 * 2 ** attempt) * (0.5 + random.random()))
            except anthropic.BadRequestError as e:          # the provider's output filter: an observable outcome
                if "content filtering" not in str(e):
                    raise
                self.usage.blocked += 1
                if self.ledger is not None:                # no usage is returned: charge a conservative estimate
                    self.ledger.charge(len(str(messages)) // 4 + 1, max_tokens)
                return BLOCKED
        self.usage.calls += 1
        self.usage.input_tokens += r.usage.input_tokens
        self.usage.output_tokens += r.usage.output_tokens
        if self.ledger is not None:
            self.ledger.charge(r.usage.input_tokens, r.usage.output_tokens)
        return "".join(b.text for b in r.content if b.type == "text")

    async def yes_prob(self, sem, context, question: str):
        msgs = fork(context, question, self.preamble)
        y = n = 0
        for _ in range(self.n_samples):
            async with sem:
                lab = _classify(await self._ask(msgs, 3))
            y += lab == "yes"
            n += lab == "no"
            self.usage.unparsed += lab is None
        return (y / (y + n) if y + n else float("nan")), y + n, "sample"

    async def reply(self, messages: list[dict], max_tokens: int = 600) -> str:
        return await self._ask(messages, max_tokens)

    async def battery(self, contexts, battery):
        sem = asyncio.Semaphore(self.concurrency)
        return await asyncio.gather(*[self.yes_prob(sem, c, q) for c in contexts for _, q in battery])


def estimate_cost(n_contexts: int, n_items: int, mean_context_tokens: int, tier: str, n_samples: int = 16,
                  usd_per_mtok_in: float = 1.0, usd_per_mtok_out: float = 4.0, cached_fraction: float = 0.0) -> dict:
    """Rough budget before running a battery on a paid API (prices are inputs; check the provider's page)."""
    calls = n_contexts * n_items * (1 if tier == "logprob" else n_samples)
    tin = calls * (mean_context_tokens + 60)
    tout = calls * 2
    usd = tin * (1 - 0.9 * cached_fraction) / 1e6 * usd_per_mtok_in + tout / 1e6 * usd_per_mtok_out
    return {"calls": calls, "input_tokens": tin, "output_tokens": tout, "usd_estimate": round(usd, 2)}
