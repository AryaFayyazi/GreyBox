#!/usr/bin/env python
"""G25: G23's test on the METHOD's own data: each agent's OWN answers (its API, its tokens). (docs/prereg_G25_fisher_snr_own.md)
Same label-free signal-to-noise ratio, same endpoints and code as G23 (scripts/190 stats(), snr(), raw_var()), on the
8 questions each agent answered about its own calls (4 answers each):
  CAPE-Bench (G15): Gemma-4 (fresh), gpt-oss-20b, Qwen3.8, Muse-Glimmer (held-out; strata template, clusters scenario),
    and Claude Haiku 5.5's 495 actions; AgentDojo (G17): Gemma-4 and gpt-oss-20b (strata suite, clusters episode);
  tau2-bench (G18): Claude Haiku 5.5's 360 writes (strata tool, clusters task).  CPU only.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G23 = _mod("g23", "scripts/190_g23_fisher_snr.py")
DG = Path("/data/models/graybox_ws/outputs/decision_greybox")
AD = Path("/data/models/graybox_ws/outputs/agentdojo_battery")
TAU = Path("/data/models/graybox_ws/outputs/tau2_claude")
B_N, SEED = 2000, 2500


def targets():
    M = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    A17 = _mod("a17", "scripts/181_agentdojo_battery.py")
    g18 = _mod("g18", "scripts/184_tau2_claude_readout.py")
    k = ["episode_id", "event_id"]
    cape_sign = np.array([-1] + [1] * 7)
    out = []
    for t in ["gemma4", "gptoss20b", "qwen38", "museglimmer"]:
        B = pd.read_parquet(DG / f"{t}_{'fresh' if t == 'gemma4' else 'heldout'}.parquet")
        U, _ = M.states(t)
        bad = set(zip(U.episode_id[~U.ok.values], U.event_id[~U.ok.values].astype(int)))
        B = B[[(e, int(v)) not in bad for e, v in zip(B.episode_id, B.event_id)]].reset_index(drop=True)
        out.append((f"cape_{t}", G23.counts(B.counts), cape_sign, B.y.values.astype(int), B.template.values, B.scenario_id.values))
    C = pd.DataFrame([json.loads(x) for x in open(DG / "claude_t8b.jsonl")])
    C = C[~C.set_index(k).index.isin(M.PILOT)].reset_index(drop=True)
    out.append(("cape_claude", G23.counts(C.counts), cape_sign, C.y.values.astype(int), C.template.values, C.scenario_id.values))
    s17 = np.array([s for _, s in A17.QUESTIONS])
    for a in ["gemma4", "gptoss20b"]:
        B = pd.read_parquet(AD / f"{a}_conf.parquet")
        out.append((f"agentdojo_{a}", G23.counts(B.counts), s17, B.y.values.astype(int), B.suite.values, B.episode.values))
    B = pd.read_json(TAU / "g18_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    out.append(("tau2_claude", G23.counts(B.counts), np.array([s for _, s in g18.QUESTIONS]), B.y.values.astype(int), B.tool.values, B.task_id.values))
    return out


def main():
    """G23.main() on these targets, its output written to a private directory (never to G23's own result file)."""
    import tempfile
    G23.targets = targets
    G23.B_N, G23.SEED = B_N, SEED
    with tempfile.TemporaryDirectory(dir="/data/models/graybox_ws/tmp") as d:
        G23.ROOT = Path(d)
        (G23.ROOT / "results/t1").mkdir(parents=True)
        G23.main()
        res = json.loads((G23.ROOT / "results/t1/g23_fisher_snr.json").read_text())
    res["note"] = "G25: G23's analysis on each agent's own answers (scripts/192)"
    (ROOT / "results/t1/g25_fisher_snr_own.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
