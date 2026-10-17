#!/usr/bin/env python
"""L3b sensitivity (deviation 2): re-run the unchanged analysis (scripts/42) without the eval conversations in which
the provider's content filter blocked Claude's reply (or a reply to an earlier turn).

python scripts/46_claude_livebench_sensitivity.py
"""
import importlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.adapters import BLOCKED  # noqa: E402

a42 = importlib.import_module("42_claude_livebench_analysis")
CL = a42.CL
TMP = Path("/data/models/graybox_ws/outputs/livebench_claude/sensitivity_no_blocked")


def main():
    ev = pd.read_parquet(CL / "eval.parquet")
    blocked = ev.reply.eq(BLOCKED) | ev.history.map(lambda h: any(BLOCKED in x for x in h))
    print(f"blocked conversations: {int(blocked.sum())} / {len(ev)}", flush=True)
    TMP.mkdir(parents=True, exist_ok=True)
    ev[~blocked].to_parquet(TMP / "eval.parquet")
    shutil.copy(CL / "calib.parquet", TMP / "calib.parquet")
    rf = pd.read_parquet(CL / "eval_refusals.parquet")
    rf[rf.id.isin(ev.id[~blocked])].to_parquet(TMP / "eval_refusals.parquet")
    a42.CL = TMP
    a42.RES = a42.RES / "sensitivity_no_blocked"
    a42.main()
    (a42.RES / "blocked.json").write_text(json.dumps({"blocked": int(blocked.sum()), "n_eval": len(ev),
                                                      "ids": ev.id[blocked].tolist()}))


if __name__ == "__main__":
    main()
