#!/usr/bin/env python
"""Run a LOCKED analysis script unchanged, with one documented deviation (G14 deviation 2, G15 deviation 1): calls
whose decision state was not stored (ok = False) are dropped from the battery files as they are read. Such calls cannot
be scored by the white-box comparison; the locked scripts assert that every call has a state. The locked files are not
edited (their SHA-256 still match the locks); this wrapper patches pandas.read_parquet for the battery files only.
  python scripts/175_run_locked_excluding_missing_states.py scripts/171_decision_greybox_LOCKED.py stage=g14
  python scripts/175_run_locked_excluding_missing_states.py scripts/174_g15_training_free.py mode=heldout
"""
import importlib.util
import re
import runpy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
spec = importlib.util.spec_from_file_location("m171locked", ROOT / "scripts/171_decision_greybox_LOCKED.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
_read, _bad, DROPPED = pd.read_parquet, {}, {}
PAT = re.compile(r"^(gemma4|gptoss20b|qwen38|museglimmer)_(dev|fresh|heldout)\.parquet$")


def bad_keys(actor):
    if actor not in _bad:
        U, _ = M.states(actor)
        _bad[actor] = set(zip(U.episode_id[~U.ok.values], U.event_id[~U.ok.values].astype(int)))
    return _bad[actor]


def read_parquet(path, *a, **k):
    df = _read(path, *a, **k)
    p = Path(path)
    m = PAT.match(p.name)
    if p.parent == M.OUT and m:
        keep = [(e, int(v)) not in bad_keys(m.group(1)) for e, v in zip(df.episode_id, df.event_id)]
        n = len(df) - sum(keep)
        if n:
            DROPPED[p.name] = n
            df = df[keep].reset_index(drop=True)
    return df


pd.read_parquet = read_parquet
target, args = sys.argv[1], sys.argv[2:]
sys.argv = [target] + args
try:
    runpy.run_path(str(ROOT / target), run_name="__main__")
finally:
    print("calls dropped for a missing decision state:", DROPPED, file=sys.stderr)
