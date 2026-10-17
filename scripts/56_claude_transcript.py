#!/usr/bin/env python
"""L3d data: the one-call grey box in the TRANSCRIPT framing on Claude Haiku 5.5 (single-turn conversations).

Re-asks single-turn conversations whose real Claude reply, refusal label and Claude-as-moderator label already exist
from L3c (scripts/50, 52). For a single-turn conversation the context is exactly the user's message, so nothing is
reconstructed. Per conversation: n = 2 answer sets to the 88 v2m questions in the transcript framing
(src/graybox/batch.py transcript_prompt), max 1,100 tokens, thinking disabled, effort low.
Stages (seed 0): calib (the 44 single-turn L3c calibration conversations), eval (all 400 harmful single-turn L3c eval
conversations, then 100 of the 170 benign ones). Ledger cap $4.95; checkpoint every 8.
Output: /data/models/graybox_ws/outputs/livebench_claude_transcript/<stage>.parquet

python scripts/56_claude_transcript.py stage=calib
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.adapters import Anthropic  # noqa: E402
from graybox.batch import batch_block, parse_batch, transcript_prompt  # noqa: E402
from graybox.battery_v2m import BATTERY_V2M  # noqa: E402
from graybox.budget import CAP_USD, HAIKU55, Ledger  # noqa: E402

LB = Path("/data/models/graybox_ws/outputs/livebench")
CB = Path("/data/models/graybox_ws/outputs/livebench_claude_batch")
OUT = Path("/data/models/graybox_ws/outputs/livebench_claude_transcript")
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
D = OmegaConf.create({"stage": "calib", "n": 2, "concurrency": 8, "seed": 0})


def select(stage, seed):
    items = pd.read_parquet(LB / "items.parquet")
    single = items[items.turns.map(len) == 1]
    if stage == "calib":
        ids = pd.read_parquet(CB / "calib.parquet").id
        return single[single.id.isin(ids)].reset_index(drop=True)
    ids = pd.concat([pd.read_parquet(CB / f"{s}.parquet").id for s in ("evalA", "evalB")])
    ev = single[single.id.isin(ids)]
    harm = ev[ev.harmful == 1]
    ben = ev[ev.harmful == 0].sample(100, random_state=seed)
    return pd.concat([harm, ben]).reset_index(drop=True)          # harmful first: the key endpoint lives there


async def run(cfg, sel):
    led = Ledger(LEDGER, CAP_USD, *HAIKU55)
    os.environ["GB_ANTHROPIC_KEY"] = KEY.read_text().strip()
    ad = Anthropic(model="claude-haiku-5-5", key_env="GB_ANTHROPIC_KEY", ledger=led)
    sem = asyncio.Semaphore(cfg.concurrency)
    block, K = batch_block(BATTERY_V2M), len(BATTERY_V2M)
    part = OUT / f"{cfg.stage}.partial.json"
    res = json.loads(part.read_text()) if part.exists() else []
    todo = sel[~sel.id.isin({r["id"] for r in res})]
    t0 = time.time()

    async def one(row):
        msgs = [{"role": "user", "content": transcript_prompt([{"role": "user", "content": row.turns[-1]}], block)}]

        async def ask():
            async with sem:
                return await ad._ask(msgs, 1100)
        texts = await asyncio.gather(*[ask() for _ in range(cfg.n)])
        return {"id": row.id, "counts": sum(parse_batch(t, K) for t in texts).tolist(), "raw0": texts[0][:1500]}

    for i in range(0, len(todo), 8):
        res += await asyncio.gather(*[one(r) for r in todo.iloc[i:i + 8].itertuples()])
        OUT.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(res))
        print(f"{len(res)}/{len(sel)} conversations, total spend ${led.total():.3f}, {time.time() - t0:.0f}s", flush=True)
    return res, led.total()


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    sel = select(cfg.stage, cfg.seed)
    print(f"{cfg.stage}: {len(sel)} conversations (harmful {int((sel.harmful == 1).sum())}); "
          f"spend so far ${Ledger(LEDGER, CAP_USD, *HAIKU55).total():.3f} (cap ${CAP_USD})", flush=True)
    try:
        res, total = asyncio.run(run(cfg, sel))
    except Exception as e:                                     # the cap (or anything else) stops the stage: keep what is done
        print("STOPPED:", type(e).__name__, str(e)[:200], flush=True)
        part = OUT / f"{cfg.stage}.partial.json"
        res, total = (json.loads(part.read_text()) if part.exists() else []), Ledger(LEDGER, CAP_USD, *HAIKU55).total()
    df = pd.DataFrame(res).merge(sel[["id", "split", "kind", "pattern", "harmful", "source"]], on="id")
    df.to_parquet(OUT / f"{cfg.stage}.parquet")
    c = np.array(df.counts.tolist())
    print(f"saved {OUT / (cfg.stage + '.parquet')} ({len(df)} conversations); answered {c[..., :2].sum() / max(c.sum(), 1):.3%}; "
          f"total spend ${total:.4f}", flush=True)


if __name__ == "__main__":
    main()
