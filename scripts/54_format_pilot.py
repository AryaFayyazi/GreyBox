#!/usr/bin/env python
"""Development pilot (not preregistered): make the one-call check-in acceptable to Claude.

L3c showed Claude Haiku 5.5 refuses the 88-question block as a suspected injection in multi-turn chats ("it isn't a
real check-in"). Fix: the OPERATOR declares the check-in up front, so it is not an unannounced instruction inside the
user's message. Variants (the fork only; the real reply never sees any of this):
  none   the L3c format (questionnaire appended to the latest user message)
  sys    + a system prompt declaring the check-in protocol (API)
  lead   + the same declaration as a leading user turn (what an in-browser page can send)
TRAIN conversations only: 12 multi-turn + 8 harmful single-turn (seed 3); one sample each; parse rate per variant.
python scripts/54_format_pilot.py
"""
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.adapters import Anthropic  # noqa: E402
from graybox.batch import BATCH_PREAMBLE, CHECKIN_SYSTEM, batch_block, parse_batch, transcript_prompt  # noqa: E402
from graybox.battery_v2m import BATTERY_V2M  # noqa: E402
from graybox.budget import CAP_USD, HAIKU55, Ledger  # noqa: E402

LB = Path("/data/models/graybox_ws/outputs/livebench")
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
TX_ONLY = "tx_only" in sys.argv                          # second run: only the transcript variant (the others ran)


async def main():
    items = pd.read_parquet(LB / "items.parquet")
    tr = items[items.split == "train"]
    sel = pd.concat([tr[tr.kind == "multiturn"].sample(12, random_state=3), tr[(tr.kind == "single") & (tr.harmful == 1)].sample(8, random_state=3)])
    led = Ledger(LEDGER, CAP_USD, *HAIKU55)
    os.environ["GB_ANTHROPIC_KEY"] = KEY.read_text().strip()
    ad = Anthropic(model="claude-haiku-5-5", key_env="GB_ANTHROPIC_KEY", ledger=led)
    client = ad.client()
    sem = asyncio.Semaphore(8)
    block, K = batch_block(BATTERY_V2M), len(BATTERY_V2M)

    async def call(msgs, system=None):
        async with sem:
            kw = {"system": system} if system else {}
            r = await client.messages.create(model="claude-haiku-5-5", max_tokens=1100, messages=msgs,
                                             thinking={"type": "disabled"}, output_config={"effort": "low"}, **kw)
            led.charge(r.usage.input_tokens, r.usage.output_tokens)
            return "".join(b.text for b in r.content if b.type == "text")

    async def one(row):
        convo = []
        for u in row.turns[:-1]:
            convo.append({"role": "user", "content": u})
            convo.append({"role": "assistant", "content": (await call(convo)).strip()[:1500] or "(no reply)"})
        fork = convo + [{"role": "user", "content": row.turns[-1] + BATCH_PREAMBLE + block}]
        lead = [{"role": "user", "content": CHECKIN_SYSTEM + "\n\n" + fork[0]["content"]}] + fork[1:]
        tx = [{"role": "user", "content": transcript_prompt(convo + [{"role": "user", "content": row.turns[-1]}], block)}]
        outs = await asyncio.gather(*([call(tx)] if TX_ONLY else [call(fork), call(fork, CHECKIN_SYSTEM), call(lead), call(tx)]))
        if TX_ONLY:
            outs = ["", "", ""] + list(outs)
        return {"id": row.id, "kind": row.kind, "harmful": row.harmful,
                **{v: float(parse_batch(t, K)[:, :2].sum() / K) for v, t in zip(("none", "sys", "lead", "transcript"), outs)},
                "sys_text": outs[1][:200], "lead_text": outs[2][:200]}
    res = pd.DataFrame(await asyncio.gather(*[one(r) for r in sel.itertuples()]))
    print(res.groupby(["kind", "harmful"])[["none", "sys", "lead", "transcript"]].mean().round(3).to_string())
    print("overall", res[["none", "sys", "lead", "transcript"]].mean().round(3).to_dict(), f"spend ${led.total():.3f}")
    for r in res[(res.sys < 0.9) | (res.lead < 0.9)].head(0 if TX_ONLY else 4).itertuples():
        print("---", r.kind, r.harmful, round(r.sys, 2), round(r.lead, 2), repr(r.sys_text[:150]), "|", repr(r.lead_text[:150]))
    res.to_parquet(f"/data/models/graybox_ws/outputs/livebench_claude_batch/format_pilot{'_tx' if TX_ONLY else ''}.parquet")


if __name__ == "__main__":
    asyncio.run(main())
