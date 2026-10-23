#!/usr/bin/env python
"""Run a figure script with every matplotlib text naming the closed model as "Claude Haiku 5.5" (the user's decision,
2026-10-22; replaces scripts/196, which wrote "Closed-A").
python scripts/196b_run_named.py scripts/<fig>.py [function ...]   (no function: run the script as __main__)"""
import re
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.text as mtext  # noqa: E402

PAIRS = [("Closed-A's", "Claude Haiku 5.5's"), ("Closed-A", "Claude Haiku 5.5"), ("closed-A", "Claude Haiku 5.5")]


WRAP = {"Claude (plain), CAPE": "Claude Haiku 5.5\n(plain), CAPE", "Claude (rule by rule), CAPE": "Claude Haiku 5.5\n(rule by rule), CAPE",
        "Claude": "Claude\nHaiku 5.5", "Claude, told it\nis monitored": "Claude Haiku 5.5,\ntold it is\nmonitored"}   # short tick labels


def rename(s):
    if isinstance(s, str) and s in WRAP:
        return WRAP[s]
    if isinstance(s, str):
        for a, b in PAIRS:
            s = s.replace(a, b)
        s = re.sub(r"Claude(?! Haiku)(?!~Haiku)", "Claude Haiku 5.5", s)
    return s


_set = mtext.Text.set_text
mtext.Text.set_text = lambda self, s: _set(self, rename(s if isinstance(s, str) else ("" if s is None else str(s))))
script, funcs = sys.argv[1], sys.argv[2:]
sys.argv = [script]
if funcs:
    sys.path.insert(0, "scripts")
    g = runpy.run_path(script, run_name="figmod")
    for f in funcs:
        g[f]()
        print("ran", f)
else:
    runpy.run_path(script, run_name="__main__")
