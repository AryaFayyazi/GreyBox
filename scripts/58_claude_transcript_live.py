#!/usr/bin/env python
"""Recorded fallback for the in-browser Live page: the scripted 9-turn conversation with Claude Haiku 5.5, read in the
transcript framing (the format the page uses). Claude's replies and its moderator labels are reused from the L3c
live stage (scripts/50, one real conversation); this adds the 2 transcript-framing answer sets per turn.
Output: /data/models/graybox_ws/outputs/livebench_claude_transcript/live.parquet
python scripts/58_claude_transcript_live.py
"""
import asyncio
import importlib
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.adapters import Anthropic  # noqa: E402
from graybox.batch import batch_block, parse_batch, transcript_prompt  # noqa: E402
from graybox.battery_v2m import BATTERY_V2M  # noqa: E402
from graybox.budget import CAP_USD, HAIKU55, Ledger  # noqa: E402

rec = importlib.import_module("27_live_record")
CB = Path("/data/models/graybox_ws/outputs/livebench_claude_batch")
OUT = Path("/data/models/graybox_ws/outputs/livebench_claude_transcript")
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")


async def main():
    L = pd.read_parquet(CB / "live.parquet")
    L["i"] = L.id.str.split("_").str[1].astype(int)
    L = L.sort_values("i").reset_index(drop=True)
    led = Ledger(LEDGER, CAP_USD, *HAIKU55)
    os.environ["GB_ANTHROPIC_KEY"] = KEY.read_text().strip()
    ad = Anthropic(model="claude-haiku-5-5", key_env="GB_ANTHROPIC_KEY", ledger=led)
    block, K = batch_block(BATTERY_V2M), len(BATTERY_V2M)
    convo, rows = [], []
    for r in L.itertuples():
        msgs = [{"role": "user", "content": transcript_prompt(convo + [{"role": "user", "content": rec.SCRIPT[r.i]}], block)}]
        texts = await asyncio.gather(*[ad._ask(msgs, 1100) for _ in range(2)])
        rows.append({"id": r.id, "counts": sum(parse_batch(t, K) for t in texts).tolist(), "reply": r.reply,
                     "claude_mod": r.claude_mod, "user": rec.SCRIPT[r.i]})
        convo += [{"role": "user", "content": rec.SCRIPT[r.i]}, {"role": "assistant", "content": r.reply}]
        print(r.i, f"answered {sum(parse_batch(t, K)[:, :2].sum() for t in texts) / (2 * K):.2f}", f"spend ${led.total():.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "live.parquet")
    print("saved", OUT / "live.parquet", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
