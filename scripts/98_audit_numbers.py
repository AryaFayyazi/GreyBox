#!/usr/bin/env python
"""Every decimal in the given documents must match (to its printed precision) a number in some saved results/**/*.json.
Prints the unmatched ones with their lines. Signs are ignored; integers and percentages are not checked.
python scripts/98_audit_numbers.py docs/paper_draft.md README.md docs/TOOL.md
"""
import glob
import json
import re
import sys

import numpy as np

vals = []


def walk(x):
    if isinstance(x, dict):
        [walk(v) for v in x.values()]
    elif isinstance(x, list):
        [walk(v) for v in x]
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        vals.append(float(x))


for f in glob.glob("results/**/*.json", recursive=True):
    try:
        walk(json.load(open(f)))
    except Exception:
        pass
V = np.abs(np.array(vals))
total = 0
for doc in sys.argv[1:]:
    txt = open(doc).read()
    lines = txt.splitlines()
    miss = {}
    for m in re.finditer(r"(?<![\w.])[+−-]?(0\.\d{2,4})(?!\d)", txt):
        x, nd = float(m.group(1)), len(m.group(1)) - 2
        if not np.any(np.abs(V - x) <= 0.5 * 10 ** -nd + 1e-9):
            miss.setdefault(txt.count("\n", 0, m.start()) + 1, []).append(m.group(0))
    total += sum(len(v) for v in miss.values())
    print(f"== {doc}: {sum(len(v) for v in miss.values())} unmatched decimals")
    for ln, xs in miss.items():
        print(f"  {ln}: {xs} | {lines[ln - 1].strip()[:110]}")
print("total unmatched", total)
