#!/usr/bin/env python
"""Live v2 benchmark on a real black box: Claude Haiku 5.5 (text-only API; no logprobs).

Uses the FROZEN bundle (models/live/readouts_v2.npz: all 88 battery-v2m items; readouts fit only on open models).
88 items x 4 samples was chosen over 24 or 48 designed items x 8 samples on the open models before any Claude data
(results/livebench: at equal cost, breadth of questions beats repeated samples). For each selected conversation:
  1. multi-turn: Claude's own replies to the earlier user turns (max 200 tokens), turn by turn
  2. Ψ: each of the 88 items, n=4 sampled one-word answers (thinking disabled, effort low, max 3 tokens)
  3. Claude's real reply to the full conversation (max 200 tokens), judged for refusal later (Qwen3Guard, scripts/37)
Selection (seed 0):
  calib  48 benign TRAIN conversations (uniform over benign train): only to set Claude's thresholds (95th percentile)
  eval   200 EVAL conversations, weighted toward harmful ones (the key endpoint is which harmful requests Claude
         complies with): ToxicChat 60 harmful + 30 benign; XSTest 40 unsafe + 40 safe; multi-turn 20 B>H + 10 H>B
Every call is charged to the shared ledger (graybox.budget; cap $4.00 over all runs). Key: read from the owner-only
file and passed to the SDK through the environment of this process only.
Output: /data/models/graybox_ws/outputs/livebench_claude/{calib,eval}.parquet (counts per item, replies, history)

python scripts/41_claude_livebench.py stage=calib      # then stage=eval
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
from graybox.adapters import Anthropic, _classify, fork  # noqa: E402
from graybox.battery_v2m import BATTERY_V2M, PREAMBLE  # noqa: E402
from graybox.budget import Ledger  # noqa: E402

MODEL = "claude-haiku-5-5"
LB = Path("/data/models/graybox_ws/outputs/livebench")
OUT = Path("/data/models/graybox_ws/outputs/livebench_claude")
BUNDLE = Path("/data/models/graybox_ws/models/live/readouts_v2.npz")
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
D = OmegaConf.create({"stage": "calib", "n": 4, "concurrency": 12, "seed": 0, "dry": False})


def select(items, stage, seed):
    rng = np.random.default_rng(seed)

    def pick(mask, k):
        idx = np.flatnonzero(mask)
        return list(rng.choice(idx, min(k, len(idx)), replace=False))
    tr, ev = (items.split == "train").values, (items.split == "eval").values
    h, kind, pat = items.harmful.values, items.kind.values, items.pattern.values
    if stage == "calib":
        sel = pick(tr & (h == 0), 48)
    else:
        sel = (pick(ev & (kind == "toxicchat") & (h == 1), 60) + pick(ev & (kind == "toxicchat") & (h == 0), 30)
               + pick(ev & (kind == "xstest") & (h == 1), 40) + pick(ev & (kind == "xstest") & (h == 0), 40)
               + pick(ev & (pat == "B>H"), 20) + pick(ev & (pat == "H>B"), 10))
    return items.iloc[sorted(sel)].reset_index(drop=True)


async def run(cfg, sel, battery):
    led = Ledger(LEDGER, 4.00, 0.10 / 1e6, 0.50 / 1e6)
    os.environ["GB_ANTHROPIC_KEY"] = KEY.read_text().strip()
    ad = Anthropic(model=MODEL, key_env="GB_ANTHROPIC_KEY", n_samples=cfg.n, concurrency=cfg.concurrency,
                   preamble=PREAMBLE, ledger=led)
    sem = asyncio.Semaphore(cfg.concurrency)
    t0 = time.time()

    async def one(row):
        convo = []
        for u in row.turns[:-1]:                                  # Claude's own replies to earlier turns
            convo.append({"role": "user", "content": u})
            async with sem:
                convo.append({"role": "assistant", "content": (await ad._ask(convo, 200)).strip() or "(no reply)"})
        convo.append({"role": "user", "content": row.turns[-1]})

        async def item(q):
            y = no = other = 0
            for _ in range(cfg.n):
                async with sem:
                    lab = _classify(await ad._ask(fork(convo, q, PREAMBLE), 3))
                y += lab == "yes"
                no += lab == "no"
                other += lab is None
            return [y, no, other]
        counts = await asyncio.gather(*[item(q) for _, q in battery])
        async with sem:
            reply = await ad._ask(convo, 200)
        return {"id": row.id, "counts": counts, "reply": reply, "history": [m["content"][:300] for m in convo[:-1]]}

    # deviation 1 (2026-10-16 02:25): checkpoint after every wave and resume from it, so a crash loses at most one wave
    part = OUT / f"{cfg.stage}.partial.json"
    res = json.loads(part.read_text()) if part.exists() else []
    done = {r["id"] for r in res}
    todo = sel[~sel.id.isin(done)]
    for i in range(0, len(todo), 8):                              # small waves: the ledger cap is checked per call
        res += await asyncio.gather(*[one(r) for r in todo.iloc[i:i + 8].itertuples()])
        OUT.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(res))
        print(f"{len(res)}/{len(sel)} conversations, total spend ${led.total():.3f}, {time.time() - t0:.0f}s", flush=True)
    return res, led.total()


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    items = pd.read_parquet(LB / "items.parquet")
    R = np.load(BUNDLE, allow_pickle=True)
    keep = list(R["items"])
    battery = [(k, q) for k, q in BATTERY_V2M if k in keep]
    assert [k for k, _ in battery] == keep, "bundle item order must follow the battery order"
    sel = select(items, cfg.stage, cfg.seed)
    nturn = int(sum(len(t) - 1 for t in sel.turns))
    est_calls = len(sel) * (len(battery) * cfg.n + 1) + nturn
    print(f"{cfg.stage}: {len(sel)} conversations x {len(battery)} items x {cfg.n} samples; about {est_calls} calls; "
          f"spend so far ${Ledger(LEDGER, 4.0, 1e-7, 5e-7).total():.3f}", flush=True)
    if cfg.dry:
        return
    res, total = asyncio.run(run(cfg, sel, battery))
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res).merge(sel[["id", "split", "kind", "pattern", "harmful", "source"]], on="id")
    df.to_parquet(OUT / f"{cfg.stage}.parquet")
    (OUT / f"{cfg.stage}_meta.json").write_text(json.dumps({"model": MODEL, "items": keep, "n": cfg.n,
                                                             "bundle": str(BUNDLE), "spend_total_after": total}))
    c = np.array(df.counts.map(lambda x: np.array(list(x))).tolist())
    print(f"saved {OUT / (cfg.stage + '.parquet')}; parsed {int(c[..., :2].sum())}/{int(c.sum())}; "
          f"total spend ${total:.4f}", flush=True)


if __name__ == "__main__":
    main()
