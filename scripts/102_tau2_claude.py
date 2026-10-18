#!/usr/bin/env python
"""Claude Haiku 5.5 on tau2-bench airline (the published domain, policy, tasks and evaluator), through the local proxy
(scripts/101). The agent and the user simulator are both Claude Haiku 5.5 (thinking disabled, effort low); the
published runs used a GPT-4.1 user, and CAPE's open-agent runs used the agent's own model as the user.
Tasks, splits and labels are CAPE's (scripts/172 of /home/arya/new_safety/cape, unchanged): split_ids() and label()
(the truthful policy oracle labels every write at the state where it executes).
Runs in the tau2 env:
  TAU2_DATA_DIR=/data/models/cape_ws/datasets/tau2-bench/data LITELLM_LOCAL_MODEL_COST_MAP=True \
  /data/models/cape_ws/envs/tau2/bin/python scripts/102_tau2_claude.py run <split: smoke|dev|conf> <trials> <name> [port] [conc]
  ... scripts/102_tau2_claude.py label <name>
"""
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

CAPE = Path("/home/arya/new_safety/cape")
OUT = Path("/data/models/graybox_ws/outputs/tau2_claude")
os.environ.setdefault("V65_ACTOR", "qwen38")          # only to import scripts/172; its sampling settings are not used
os.environ.setdefault("V65_COND", "P1")
_s = importlib.util.spec_from_file_location("cape172", CAPE / "scripts/172_v65_tau2_run.py")
m172 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m172)
assert m172.DOMAIN_AGENT == ("airline", "llm_agent") and m172.USER == "user_simulator"


def run(split, trials, name, port=8801, conc=4, seed=300):
    from tau2.cli import main
    base = f"http://127.0.0.1:{port}/v1"
    agent_args = {"api_base": base, "api_key": "EMPTY", "max_tokens": 1536, "timeout": 900}
    user_args = {"api_base": base, "api_key": "EMPTY", "temperature": 0.7, "max_tokens": 512, "timeout": 900}
    sys.argv = ["tau2", "run", "--domain", "airline", "--agent", "llm_agent", "--agent-llm", "openai/claude-haiku-5-5",
                "--agent-llm-args", json.dumps(agent_args), "--user", "user_simulator", "--user-llm", "openai/claude-haiku-5-5",
                "--user-llm-args", json.dumps(user_args), "--num-trials", str(trials), "--max-concurrency", str(conc),
                "--save-to", name, "--task-split-name", "base", "--max-retries", "1", "--seed", str(seed),
                "--task-ids"] + m172.split_ids(split)
    main()


def label(name):
    m172.label(name)
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in ("labels", "episodes"):
        shutil.copy(CAPE / f"outputs/v65/{kind}_{name}.csv", OUT / f"{kind}_{name}.csv")
    print("copied labels to", OUT)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        a = sys.argv[2:]
        run(a[0], int(a[1]), a[2], *(int(x) for x in a[3:5]))
    elif cmd == "label":
        label(sys.argv[2])
