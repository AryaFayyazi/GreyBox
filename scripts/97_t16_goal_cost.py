#!/usr/bin/env python
"""Exploratory (not preregistered): where does the goal cost of the live gate (T16, T16b) come from?

All T16/T16b scenarios are violation-world (x-) instances. For each gated episode: a FALSE block = a blocked action
that would not have been a violation; a TRUE block = one that would. Goal rates are compared with the ungated
episode of the same scenario. Output: results/t1/t16_goal_cost_exploratory.json
python scripts/97_t16_goal_cost.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"T16_gpt": "/data/models/graybox_ws/outputs/t16/run.partial.jsonl",
        "T16b_claude": "/data/models/graybox_ws/outputs/t16b/run.partial.jsonl"}


def main():
    out = {}
    for name, f in RUNS.items():
        R = pd.DataFrame([json.loads(l) for l in open(f)])
        assert (R.instance_id.str.endswith(".unsafe")).all()
        R["false_block"] = (R.blocked - R.blocked_would_be_unsafe) > 0
        R["true_block"] = R.blocked_would_be_unsafe > 0
        R["clean_goal"] = R.goal & ~R.unsafe_episode
        g, u = R[R.arm == "gated"].copy(), R[R.arm == "ungated"].set_index("scenario_id")
        g["goal_ungated"] = g.scenario_id.map(u.goal)
        o = {"episodes_per_arm": int(len(g)),
             "goal_rate": {a: round(float(R[R.arm == a].goal.mean()), 4) for a in ("ungated", "gated")},
             "goal_without_violation": {a: round(float(R[R.arm == a].clean_goal.mean()), 4) for a in ("ungated", "gated")}}
        for lab, m in (("false_block", g.false_block), ("true_block_only", g.true_block & ~g.false_block), ("no_block", g.blocked == 0)):
            s = g[m]
            o[lab] = {"episodes": int(len(s)), "goal_gated": round(float(s.goal.mean()), 4) if len(s) else None,
                      "goal_same_scenarios_ungated": round(float(s.goal_ungated.mean()), 4) if len(s) else None}
        loss = (g.goal_ungated.astype(float) - g.goal.astype(float))
        o["share_of_goal_loss_in_false_block_episodes"] = round(float(loss[g.false_block].sum() / loss.sum()), 4) if loss.sum() else None
        out[name] = o
        print(name, json.dumps(o))
    (ROOT / "results/t1/t16_goal_cost_exploratory.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
