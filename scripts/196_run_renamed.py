#!/usr/bin/env python
"""Run a figure script unchanged, with every matplotlib text renamed: the closed model is "Closed-A" in all publications.
python scripts/196_run_renamed.py scripts/160_fig_law.py [args...]"""
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.text as mtext  # noqa: E402

PAIRS = [("Claude Haiku 5.5", "Closed-A"), ("Claude Haiku~5.5", "Closed-A"), ("Claude's", "Closed-A's"), ("Claude", "Closed-A")]


def rename(s):
    if isinstance(s, str):
        for a, b in PAIRS:
            s = s.replace(a, b)
    return s


_set = mtext.Text.set_text
mtext.Text.set_text = lambda self, s: _set(self, rename(s if isinstance(s, str) else ("" if s is None else str(s))))
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
