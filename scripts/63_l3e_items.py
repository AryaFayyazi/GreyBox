#!/usr/bin/env python
"""L3e items: fresh conversations for a prospective test of the Live tool on Claude (docs/prereg_L3e_claude_tool.md).

Sources never shown to Claude and never used to fit any readout. Exact-text de-duplication against every turn of the
Live benchmark (outputs/livebench/items.parquet):
  ToxicChat (lmsys/toxic-chat 0124, train+test CSVs): unused toxic prompts (all, incl. jailbreak attempts), unused benign
  HarmfulQA (declare-lab): harmful questions, never used anywhere in this project
Conversations (seed 2026):
  single  H: all unused ToxicChat toxic + 150 HarmfulQA;  B: 200 unused ToxicChat benign
  multi   B>H 120 (ToxicChat benign, then HarmfulQA);  H>B 80 (HarmfulQA, then ToxicChat benign);  B>B 80
The label `harmful` is the LATEST message's label. Output: outputs/l3e/items.parquet (frozen; its sha256 is locked).
python scripts/63_l3e_items.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/data/models/graybox_ws/outputs/l3e")


def main():
    rng = np.random.default_rng(2026)
    lb = pd.read_parquet("/data/models/graybox_ws/outputs/livebench/items.parquet")
    used = set(t.strip() for ts in lb.turns for t in ts)
    tc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("/data/models/datasets--lmsys--toxic-chat/snapshots/*/data/0124/*.csv"))])
    tc = tc[~tc.user_input.str.strip().isin(used)].drop_duplicates("user_input")
    tc = tc[tc.user_input.str.len().between(5, 4000)]
    tox, ben = tc[tc.toxicity == 1], tc[tc.toxicity == 0]
    hq = pd.DataFrame(json.load(open(glob.glob("/data/models/datasets--declare-lab--HarmfulQA/blobs/8d5a*")[0])))
    hq = hq[~hq.question.str.strip().isin(used)].drop_duplicates("question")
    hq_i = rng.permutation(len(hq))
    ben_i = rng.permutation(len(ben))
    q = lambda k: hq.question.iloc[hq_i[k]].strip()                     # noqa: E731
    b = lambda k: ben.user_input.iloc[ben_i[k]].strip()                 # noqa: E731
    rows, qh, qb = [], 0, 0
    for _, r in tox.iterrows():
        rows.append({"kind": "single", "pattern": "H", "turns": [r.user_input.strip()], "harmful": 1, "source": "toxicchat", "jailbreak": int(r.jailbreaking)})
    for _ in range(150):
        rows.append({"kind": "single", "pattern": "H", "turns": [q(qh)], "harmful": 1, "source": "harmfulqa", "jailbreak": 0}); qh += 1
    for _ in range(200):
        rows.append({"kind": "single", "pattern": "B", "turns": [b(qb)], "harmful": 0, "source": "toxicchat", "jailbreak": 0}); qb += 1
    for _ in range(120):
        rows.append({"kind": "multi", "pattern": "B>H", "turns": [b(qb), q(qh)], "harmful": 1, "source": "mixed", "jailbreak": 0}); qb += 1; qh += 1
    for _ in range(80):
        rows.append({"kind": "multi", "pattern": "H>B", "turns": [q(qh), b(qb)], "harmful": 0, "source": "mixed", "jailbreak": 0}); qb += 1; qh += 1
    for _ in range(80):
        rows.append({"kind": "multi", "pattern": "B>B", "turns": [b(qb), b(qb + 1)], "harmful": 0, "source": "toxicchat", "jailbreak": 0}); qb += 2
    df = pd.DataFrame(rows)
    df["id"] = [f"e{i}" for i in range(len(df))]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "items.parquet")
    print(df.groupby(["kind", "pattern", "source"]).size().to_string(), "\njailbreaks", int(df.jailbreak.sum()), "total", len(df))


if __name__ == "__main__":
    main()
