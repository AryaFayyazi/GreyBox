#!/usr/bin/env python
"""EXPLORATORY (after G18, G33, G37, G41): where does Claude Haiku 5.5's tau2-bench read-out fail, and why? Per violating
task: the G18 read-out's AUROC of its violations against the allowed writes of the same tool, and the share of its
'complies?' answers that say yes. Per violation type: CAPE's oracle decision text. The agent's reply before each failing
write is printed for inspection (knowing violation vs honest mistake). CPU only.
python scripts/217_tau2_knowing_vs_mistake_exploratory.py  -> results/t1/tau2_knowing_vs_mistake_exploratory.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = Path("/data/models/graybox_ws/outputs/tau2_claude")


def main():
    s = importlib.util.spec_from_file_location("g18", ROOT / "scripts/184_tau2_claude_readout.py")
    G18 = importlib.util.module_from_spec(s)
    s.loader.exec_module(G18)
    U = pd.read_json(O / "g18_units_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    U = G18.sample(U, G18.CFG["n_units"], G18.CFG["seed"]).reset_index(drop=True)
    U["k"] = U.groupby(["sim_id", "msg_idx"]).cumcount()
    G = pd.read_json(O / "g18_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    G["k"] = G.groupby(["sim_id", "msg_idx"]).cumcount()
    J = U[["sim_id", "msg_idx", "k", "task_id", "tool", "y", "messages"]].merge(G[["sim_id", "msg_idx", "k", "counts"]], on=["sim_id", "msg_idx", "k"])
    sg = np.array([s_ for _, s_ in G18.QUESTIONS])
    C = np.array([np.array(c, float) for c in J.counts])
    T = 2 * np.arcsin(np.sqrt((C[..., 0] + 3 / 8) / (C[..., 1] + 3 / 4)))
    J["s"] = (T * sg).mean(1)
    J["says_complies"] = C[:, 0, 0] / np.maximum(C[:, 0, 1], 1)
    out = {"per_task": {}}
    for task, g in J[J.y == 1].groupby("task_id"):
        a = []
        for x, tool in zip(g.s, g.tool):
            neg = J[(J.y == 0) & (J.tool == tool)].s.values
            a.append(float((x > neg).mean() + 0.5 * (x == neg).mean()))
        last = [m for m in json.loads(g.messages.iloc[0]) if m["role"] == "assistant" and m.get("content")]
        out["per_task"][task] = {"violations": int(len(g)), "tools": sorted(set(g.tool)), "readout_auroc": round(float(np.mean(a)), 4),
                                 "says_complies": round(float(g.says_complies.mean()), 4),
                                 "agent_reply_before": [str(m["content"])[:600] for m in last[-2:]]}
    ok = [v for v in out["per_task"].values() if v["readout_auroc"] >= 0.75]
    out["tasks_read_well"] = f"{len(ok)} of {len(out['per_task'])} tasks, {sum(v['violations'] for v in ok)} of {int(J.y.sum())} violations"
    (ROOT / "results/t1/tau2_knowing_vs_mistake_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: ({t: {kk: vv for kk, vv in v.items() if kk != "agent_reply_before"} for t, v in out["per_task"].items()} if k == "per_task" else v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
