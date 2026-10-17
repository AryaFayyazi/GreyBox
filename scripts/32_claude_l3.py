#!/usr/bin/env python
"""L3: do the Live readouts (trained only on open models) work on a CLOSED model? Claude Haiku 5.5 via the Anthropic API.

For held-out L2 contexts (confirmation split, never used to build the readouts): ask the 24 live battery items in the
same format as the open models (question appended to the user's message), n sampled one-word answers each (the
Anthropic API returns no logprobs: text-only tier), plus Claude's real reply to the unmodified message.
Hard budget: a persistent spend ledger is checked before every call; the run stops at CAP_USD across all runs.
Key: read from /data/models/graybox_ws/.secrets/anthropic_key (owner-only), never printed or logged.

python scripts/32_claude_l3.py stage=pilot          # 3 contexts x 24 items x 2 samples (~$0.005)
python scripts/32_claude_l3.py stage=run n=8        # 192 contexts x 24 items x 8 samples (~$1.2)
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.adapters import fork  # noqa: E402
from graybox.battery_v2 import BATTERY_V2  # noqa: E402

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6          # $/token, prompts < 100K tokens (skill pricing table, 2026-10-06)
CAP_USD = 4.00                                        # hard cap over ALL runs (the user's key holds $5)
L2 = Path("/data/models/graybox_ws/outputs/l2")
OUT = Path("/data/models/graybox_ws/outputs/l3")
LEDGER = OUT / "spend.json"
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
D = OmegaConf.create({"stage": "pilot", "n": 8, "n_ctx_per_source": 64, "concurrency": 12, "seed": 0})


class Budget:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
        self.lock = asyncio.Lock()

    async def charge(self, usage):
        async with self.lock:
            self.d["input_tokens"] += usage.input_tokens
            self.d["output_tokens"] += usage.output_tokens
            self.d["calls"] += 1
            self.d["usd"] = self.d["input_tokens"] * PRICE_IN + self.d["output_tokens"] * PRICE_OUT
            if self.d["calls"] % 200 == 0:
                LEDGER.write_text(json.dumps(self.d))

    def ok(self):
        return self.d["usd"] < CAP_USD

    def save(self):
        LEDGER.write_text(json.dumps(self.d))


def classify(text):
    t = (text or "").strip().strip(".!*\"'").lower()
    return "yes" if t.startswith("yes") else "no" if t.startswith("no") else None


async def ask(client, sem, budget, msgs, max_tokens):
    async with sem:
        if not budget.ok():
            raise RuntimeError("budget cap reached")
        r = await client.messages.create(model=MODEL, max_tokens=max_tokens, messages=msgs,
                                         thinking={"type": "disabled"}, output_config={"effort": "low"})
        await budget.charge(r.usage)
        return "".join(b.text for b in r.content if b.type == "text"), r.stop_reason


async def item(client, sem, budget, text, q, n):
    y = no = other = 0
    for _ in range(n):
        out, _stop = await ask(client, sem, budget, fork(text, q), max_tokens=3)
        lab = classify(out)
        y += lab == "yes"
        no += lab == "no"
        other += lab is None
    return y, no, other


async def run(cfg, contexts, items):
    budget = Budget()
    client = anthropic.AsyncAnthropic(api_key=KEY.read_text().strip(), max_retries=5)
    sem = asyncio.Semaphore(cfg.concurrency)
    t0 = time.time()
    res = []
    try:
        for i, row in enumerate(contexts.itertuples()):
            if not budget.ok():
                print("STOP: budget cap reached", flush=True)
                break
            counts = await asyncio.gather(*[item(client, sem, budget, row.text, q, cfg.n) for _, q in items])
            reply, stop = await ask(client, sem, budget, [{"role": "user", "content": row.text}], max_tokens=400)
            res.append({"id": row.id, "source": row.source, "counts": counts, "reply": reply, "stop_reason": stop})
            if i % 16 == 0:
                print(f"{i + 1}/{len(contexts)} contexts, spend ${budget.d['usd']:.3f}, {time.time() - t0:.0f}s", flush=True)
    finally:
        budget.save()
    return res, budget.d


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    R = np.load("/data/models/graybox_ws/models/live/readouts.npz", allow_pickle=True)
    keep = list(R["items"])
    items = [(k, q) for k, q in BATTERY_V2 if k in keep]
    C = pd.read_parquet(L2 / "contexts.parquet")
    conf = C[C.split == "confirm"]
    rng = np.random.default_rng(cfg.seed)
    if cfg.stage == "pilot":
        ctx = conf[conf.source.isin(["harmful", "benign", "sst2"])].groupby("source").head(1)
        cfg.n = 2
    else:
        parts = [g.iloc[rng.choice(len(g), min(cfg.n_ctx_per_source, len(g)), replace=False)]
                 for s, g in conf.groupby("source") if s in ("harmful", "benign", "sst2")]
        ctx = pd.concat(parts).reset_index(drop=True)
    print(f"{cfg.stage}: {len(ctx)} contexts x {len(items)} items x {cfg.n} samples = {len(ctx) * len(items) * cfg.n} calls"
          f" + {len(ctx)} replies; cap ${CAP_USD}", flush=True)
    res, spend = asyncio.run(run(cfg, ctx, items))
    out = OUT / f"claude_haiku55_{cfg.stage}.json"
    out.write_text(json.dumps({"model": MODEL, "items": [k for k, _ in items], "n": cfg.n, "results": res, "spend": spend}))
    parsed = sum(sum(y + no for y, no, _ in r["counts"]) for r in res)
    total = sum(sum(y + no + o for y, no, o in r["counts"]) for r in res)
    print(f"saved {out}; parsed {parsed}/{total}; spend so far ${spend['usd']:.4f} "
          f"({spend['calls']} calls, {spend['input_tokens']} in / {spend['output_tokens']} out tokens)", flush=True)


if __name__ == "__main__":
    main()
