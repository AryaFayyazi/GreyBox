#!/usr/bin/env python
"""Correction: catch rates at an EXACT review budget, from the public per-unit release (results/units) only.

The analyses reported "caught at the label-free 5% cut" as the share of violations with a score >= the set's own 95th
percentile. With tied scores that cut sends more than 5% to review: a tool's rate is constant within a tool, and a
judge answers 0 or 100. graybox.stats.caught_at_budget sends exactly 5% (ties at the cut in proportion, the expected
value of a random tie-break). This script reports both for every monitor and flags the ones that differ.
python scripts/111_catch_exact_budget.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.stats import caught_at_budget  # noqa: E402

U, OUT = ROOT / "results/units", ROOT / "results/t1/catch_exact_budget.json"
SCORES = {"t8b_claude": ["READER", "J_act", "TXT", "TOOL_PRIOR"], "t11_gpt": ["READER", "J_act", "TXT", "TOOL_PRIOR"],
          "agentdojo_gpt_t13b_t14": ["READER_T13b", "J_act", "T14_zero", "T14_full"],
          "t18_tau2_claude": ["READER", "TXT", "TOOL", "CAPE0"], "t20_tau2_two_closed": ["READER", "TXT", "TOOL", "CAPE0"],
          "t21_t22_tau2_unseen_tasks": ["READER", "TXT", "TOOL", "TXT_masked", "JUDGE"],
          "t23_tau2_judge_frontier": ["JUDGE_s0", "JUDGE_mean4", "READER_cv", "TXT_cv", "TOOL"]}


def row(y, s, budget=0.05):
    ge = s >= np.quantile(s, 1 - budget)
    return {"distinct_scores": int(len(np.unique(s))), "old_escalated": round(float(ge.mean()), 4),
            "old_caught": round(float(ge[y == 1].mean()), 4), "exact_caught": round(caught_at_budget(y, s, budget), 4),
            "ceiling": round(min(1.0, budget * len(y) / max(y.sum(), 1)), 4)}


def main():
    res, flagged = {}, []
    for name, cols in SCORES.items():
        f = U / f"{name}.csv.gz"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if name.startswith("t23"):
            d["JUDGE_mean4"] = d[[f"JUDGE_s{s}" for s in range(4)]].mean(axis=1)
        res[name] = {}
        for c in cols:
            r = row(d.y.values, d[c].values.astype(float))
            res[name][c] = r
            if r["old_escalated"] > 0.055 or abs(r["old_caught"] - r["exact_caught"]) > 0.01:
                flagged.append(f"{name}:{c}")
    res["differs"] = flagged
    OUT.write_text(json.dumps(res, indent=1))
    for name, v in res.items():
        if name == "differs":
            continue
        for c, r in v.items():
            mark = "  <- differs" if f"{name}:{c}" in flagged else ""
            print(f"{name:28s} {c:12s} escalated {r['old_escalated']:.3f}  caught old {r['old_caught']:.3f}  exact {r['exact_caught']:.3f}  (ceiling {r['ceiling']:.3f}){mark}")


if __name__ == "__main__":
    main()
