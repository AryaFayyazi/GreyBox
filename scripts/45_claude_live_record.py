#!/usr/bin/env python
"""Record a Grey Box Live v2 session with Claude Haiku 5.5 (text-only API): the same turns as the open-model session.

Readouts: the frozen v2 bundle (88 items, fit on open models only). Thresholds: Claude's own black-box calibration,
the 95th percentile of its scores on the 48 benign calibration conversations of scripts/41 (computed exactly as in
scripts/42). Each turn: 88 questions x 4 sampled answers, then Claude's real reply. Every call is charged to the shared
ledger (cap $4.00). Output: outputs/live/session_v2_claude_haiku55.json (then scripts/43 adds moderation and refusal).

python scripts/45_claude_live_record.py
"""
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.adapters import Anthropic  # noqa: E402
from graybox.budget import Ledger  # noqa: E402
from graybox.live import BUNDLE_V2, LiveSession, Readouts  # noqa: E402

rec = importlib.import_module("27_live_record")
a42 = importlib.import_module("42_claude_livebench_analysis")
OUT = Path("/data/models/graybox_ws/outputs/live")
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
NAME = "claude_haiku55"


def main():
    R = Readouts.load(BUNDLE_V2)
    cal = pd.read_parquet("/data/models/graybox_ws/outputs/livebench_claude/calib.parquet")
    sc = a42.scores(cal, np.load(BUNDLE_V2, allow_pickle=True))
    for c in R.concepts:                                    # in memory only; the frozen bundle file is not changed
        c["threshold"] = dict(c["threshold"], **{NAME: round(float(np.quantile(sc[c["name"]], 0.95)), 4)})
    os.environ["GB_ANTHROPIC_KEY"] = KEY.read_text().strip()
    led = Ledger(LEDGER, 4.00, 0.10 / 1e6, 0.50 / 1e6)
    ad = Anthropic(model="claude-haiku-5-5", key_env="GB_ANTHROPIC_KEY", n_samples=4, concurrency=12, ledger=led)
    sess = LiveSession(adapter=ad, readouts=R, model_key=NAME)
    events = []
    for msg in rec.SCRIPT:
        ev = asyncio.run(sess.turn(msg))
        events.append(ev)
        print(json.dumps({k: ev.get(k) for k in ("turn", "seconds", "point")}),
              [(c["name"], c["score"], c["flag"]) for c in ev["concepts"]], f"spend ${led.total():.3f}", flush=True)
    out = {"model": NAME, "hf": "claude-haiku-5-5 (API)", "bundle": "v2", "battery": R.battery, "events": events,
           "cloud": R.cloud.round(4).tolist(), "cloud_label": R.cloud_label, "questions": dict(sess.battery()),
           "concept_meta": [{"name": c["name"], "kind": c.get("kind"), "validation": c["validation"],
                             "threshold": R.threshold(c, NAME)} for c in R.concepts],
           "spend_total_after": led.total()}
    (OUT / f"session_v2_{NAME}.json").write_text(json.dumps(out, default=float))
    print("saved", OUT / f"session_v2_{NAME}.json", flush=True)


if __name__ == "__main__":
    main()
