#!/usr/bin/env python
"""Summaries of private per-action files, so that every number the paper and site quote has a public source
(reviewer finding: "numbers with no public source"). Output: results/t1/public_summaries.json
  reader_latency_T6   the reader's forward pass per action (Qwen3-8B, one H100, unbatched), T6's 5,229 actions
  t17_fidelity        teacher-forced accuracy / probability of the next 16 tokens of each agent's actual tool call
  t18_episodes        all T18 episodes vs episodes with at least one labelled (ALLOWED / NOT_ALLOWED) write
python scripts/107_public_summaries.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
W = Path("/data/models/graybox_ws/outputs")


def main():
    out = {}
    lat = np.load(W / "t6/gptoss_act.npz")["latency_s"] * 1000
    out["reader_latency_T6"] = {"n": int(len(lat)), "median_ms": round(float(np.median(lat)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1),
                                "hardware": "1 NVIDIA H100 NVL, bf16, unbatched"}
    fid = {}
    for a in ("gptoss20b", "qwen38", "museglimmer"):
        d = np.load(W / f"t17/{a}_decision_hidden.npz")
        acc, p = d["acc_next"][d["ok"]], d["p_next"][d["ok"]]
        fid[a] = {"steps": int(d["ok"].sum()), "mean_accuracy": round(float(np.nanmean(acc)), 4), "mean_probability": round(float(np.nanmean(p)), 4)}
    out["t17_fidelity_next16_tokens"] = fid
    sims = json.load(open("/data/models/cape_ws/datasets/tau2-bench/data/simulations/gb_tau2_claude_conf/results.json"))["simulations"]
    te = pd.read_parquet(W / "tau2_claude/t18_test_gb_tau2_claude_conf.parquet")
    out["t18_episodes"] = {"all_episodes": len(sims), "episodes_with_a_labelled_write": int(te.sim_id.nunique()),
                           "labelled_writes": int(len(te)), "violations": int(te.y.sum())}
    (ROOT / "results/t1/public_summaries.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
