#!/usr/bin/env python
"""Exploratory (not preregistered): where does T16c's goal difference come from? Each review-arm episode is grouped by
what the reviewer did, and compared with T16b's ungated episode of the same instance (the preregistered control).
Output: results/t1/t16c_goal_decomposition_exploratory.json
python scripts/97b_t16c_goal_decomposition.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    R = pd.DataFrame([json.loads(l) for l in open("/data/models/graybox_ws/outputs/t16c/run.partial.jsonl")]).set_index("instance_id")
    B = pd.DataFrame([json.loads(l) for l in open("/data/models/graybox_ws/outputs/t16b/run.partial.jsonl")])
    U = B[B.arm == "ungated"].set_index("instance_id")
    R["goal_ungated"], R["violation_ungated"] = U.loc[R.index, "goal"].values, U.loc[R.index, "unsafe_episode"].values
    groups = {"a_violation_held_and_blocked": R.blocked > 0, "held_and_approved_no_block": ((R.held - R.blocked) > 0) & (R.blocked == 0),
              "never_held": R.held == 0}
    out = {k: {"episodes": int(m.sum()), "goal_review": round(float(R[m].goal.mean()), 4), "goal_same_instances_ungated": round(float(R[m].goal_ungated.mean()), 4),
               "ungated_had_a_violation": round(float(R[m].violation_ungated.mean()), 4)} for k, m in groups.items()}
    out["note"] = ("In never-held episodes the two arms ran the same way (the monitor did nothing), so their goal difference is "
                   "run-to-run variation: the control is T16b's earlier run, not a concurrent arm.")
    (ROOT / "results/t1/t16c_goal_decomposition_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
