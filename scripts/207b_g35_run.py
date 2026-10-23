#!/usr/bin/env python
"""Run G35's locked analysis (scripts/207) with the recorded deviation (docs/deviation_G35_2026-10-22.md): calls whose
stored state is not ok are dropped before loading, as G25 does (scripts/192). The locked file is not changed."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("g35", ROOT / "scripts/207_g35_readout_reads_state.py")
G35 = importlib.util.module_from_spec(s)
s.loader.exec_module(G35)
_read = pd.read_parquet
_m171 = G35._mod("m171_dev", "scripts/171_decision_greybox_LOCKED.py")
_bad = {}


def read_filtered(path, *a, **k):
    B = _read(path, *a, **k)
    name = Path(str(path)).name
    for agent in ("gemma4", "gptoss20b", "qwen38", "museglimmer"):
        if name in (f"{agent}_fresh.parquet", f"{agent}_heldout.parquet") and {"episode_id", "event_id"} <= set(B.columns):
            if agent not in _bad:
                U, _ = _m171.states(agent)
                _bad[agent] = set(zip(U.episode_id[~U.ok.values], U.event_id[~U.ok.values].astype(int)))
            keep = [(e, int(v)) not in _bad[agent] for e, v in zip(B.episode_id, B.event_id)]
            print(f"deviation: {name}: dropped {len(B) - sum(keep)} call(s) without a valid state", flush=True)
            return B[keep].reset_index(drop=True)
    return B


pd.read_parquet = read_filtered
G35.pd.read_parquet = read_filtered
if __name__ == "__main__":
    G35.main()
