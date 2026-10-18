#!/usr/bin/env python
"""Tool check: how much does a monitor lose when it sees only what an MCP proxy sees? (graybox.mcp_proxy)
GPT example (site/dist/data/example_gpt_openai.jsonl). The text monitor (current defaults) is trained on half of the
tasks (hash split) and scored on the other half with: the full context (as Guard in the agent loop); the instructions
and earlier calls but no user task (the proxy with --instructions); only the earlier calls (the proxy with nothing).
Output: results/tool/proxy_context_cost.json
python scripts/117_proxy_context_cost.py
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.monitor import Step, TextMonitor, act_text, ctx_text  # noqa: E402
from graybox.stats import caught_at_budget  # noqa: E402
from graybox.train_reader import load_steps  # noqa: E402


def main():
    st = load_steps(str(ROOT / "site/dist/data/example_gpt_openai.jsonl"))
    y = np.array([int(s["label"]) for s in st])
    test = np.array([zlib.crc32(s["group"].encode()) % 2 == 1 for s in st])
    tm = TextMonitor().fit([st[i] for i in np.where(~test)[0]], y[~test], [st[i]["group"] for i in np.where(~test)[0]])
    res = {"train_actions": int((~test).sum()), "test_actions": int(test.sum()), "test_violations": int(y[test].sum())}
    for name, keep_instr, keep_task in (("full_context", True, True), ("no_task", True, False), ("calls_only", False, False)):
        z = []
        for i in np.where(test)[0]:
            s = st[i]
            hist = [Step(h["tool"], h.get("args") or {}, h.get("result", "")) for h in s.get("history") or []]
            ctx = ctx_text(s.get("instructions", "") if keep_instr else "", s.get("task", "") if keep_task else "", hist)
            z.append(tm.decision(None, None, None, None, rendered=(ctx, act_text(s["action"]["tool"], s["action"].get("args")))))
        z, yt = np.array(z), y[test]
        res[name] = {"auroc": round(float(roc_auc_score(yt, z)), 4), "caught_at_5pct": round(caught_at_budget(yt, z, 0.05), 4)}
    (ROOT / "results/tool/proxy_context_cost.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
