#!/usr/bin/env python
"""L3e data: the Live tool exactly as deployed, on fresh conversations, with Claude Haiku 5.5 (prospective test).

Per conversation (outputs/l3e/items.parquet, frozen):
  1. multi-turn: Claude's own replies to the earlier user turns (max 600 tokens, as the page does), turn by turn
  2. Ψ: 2 answer sets in the transcript framing (src/graybox/batch.py transcript_prompt), max 1,100 tokens each
  3. Claude as a text moderator of the latest message alone (MOD_PROMPT of scripts/50)
  4. Claude's real reply to the full conversation (max 400 tokens), judged for refusal later (scripts/65)
Thinking disabled, effort low (the page's settings). One process; checkpoint every 8; shared ledger, cap $9.90 (the
user added $5 on 2026-10-16: total key $10).
Output: outputs/l3e/claude.parquet
python scripts/64_l3e_claude.py
"""
import asyncio
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.adapters import Anthropic  # noqa: E402
from graybox.batch import batch_block, parse_batch, transcript_prompt  # noqa: E402
from graybox.battery_v2m import BATTERY_V2M  # noqa: E402
from graybox.budget import HAIKU55, Ledger  # noqa: E402

MOD_PROMPT = importlib.import_module("50_claude_batch").MOD_PROMPT
OUT = Path("/data/models/graybox_ws/outputs/l3e")
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
CAP = 9.90


async def run(items):
    led = Ledger(LEDGER, CAP, *HAIKU55)
    os.environ["GB_ANTHROPIC_KEY"] = KEY.read_text().strip()
    ad = Anthropic(model="claude-haiku-5-5", key_env="GB_ANTHROPIC_KEY", ledger=led)
    sem = asyncio.Semaphore(8)
    block, K = batch_block(BATTERY_V2M), len(BATTERY_V2M)
    part = OUT / "claude.partial.json"
    res = json.loads(part.read_text()) if part.exists() else []
    todo = items[~items.id.isin({r["id"] for r in res})]
    t0 = time.time()

    async def ask(msgs, n):
        async with sem:
            return await ad._ask(msgs, n)

    async def one(row):
        convo = []
        for u in row.turns[:-1]:
            convo.append({"role": "user", "content": u})
            convo.append({"role": "assistant", "content": (await ask(convo, 600)).strip() or "(no reply)"})
        last = row.turns[-1]
        tx = [{"role": "user", "content": transcript_prompt(convo + [{"role": "user", "content": last}], block)}]
        texts = await asyncio.gather(ask(tx, 1100), ask(tx, 1100))
        mtxt, reply = await asyncio.gather(ask([{"role": "user", "content": MOD_PROMPT.format(msg=last[:6000])}], 5),
                                           ask(convo + [{"role": "user", "content": last}], 400))
        m = re.search(r"(safe|controversial|unsafe)", mtxt or "", re.I)
        return {"id": row.id, "counts": sum(parse_batch(t, K) for t in texts).tolist(), "claude_mod": m.group(1).capitalize() if m else None,
                "reply": reply, "history": [x["content"][:400] for x in convo]}

    for i in range(0, len(todo), 8):
        res += await asyncio.gather(*[one(r) for r in todo.iloc[i:i + 8].itertuples()])
        part.write_text(json.dumps(res))
        print(f"{len(res)}/{len(items)} conversations, total spend ${led.total():.3f}, {time.time() - t0:.0f}s", flush=True)
    return res, led.total()


def main():
    items = pd.read_parquet(OUT / "items.parquet")
    print(f"L3e: {len(items)} conversations; spend so far ${Ledger(LEDGER, CAP, *HAIKU55).total():.3f} (cap ${CAP})", flush=True)
    try:
        res, total = asyncio.run(run(items))
    except Exception as e:                                  # the cap or anything else: keep what is done
        print("STOPPED:", type(e).__name__, str(e)[:200], flush=True)
        part = OUT / "claude.partial.json"
        res, total = (json.loads(part.read_text()) if part.exists() else []), Ledger(LEDGER, CAP, *HAIKU55).total()
    df = pd.DataFrame(res).merge(items[["id", "kind", "pattern", "harmful", "source", "jailbreak"]], on="id")
    df.to_parquet(OUT / "claude.parquet")
    c = np.array(df.counts.tolist())
    print(f"saved {len(df)} conversations; answered {c[..., :2].sum() / max(c.sum(), 1):.3%}; claude_mod "
          f"{df.claude_mod.value_counts(dropna=False).to_dict()}; total spend ${total:.4f}", flush=True)


if __name__ == "__main__":
    main()
