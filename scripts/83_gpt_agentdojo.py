#!/usr/bin/env python
"""GPT-6-luna on AgentDojo with CAPE's exact harness (cape/scripts/191_v74_agentdojo.py, v1.2.2, arm 'spot', dev split):
same suites, user and injection tasks, important_instructions attack, spotlighting prompt, per-call 'injected' labels
(attacker-only argument strings). Only the model call changes: OpenAI Chat Completions with gpt-6-luna,
reasoning_effort "none" and parallel_tool_calls false (as T11). Every call goes to the OpenAI ledger (cap $5.00).
Runs in the AgentDojo env:
  /data/models/cape_ws/envs/agentdojo/bin/python -I scripts/83_gpt_agentdojo.py pilot      # 6 episodes, never analysed
  /data/models/cape_ws/envs/agentdojo/bin/python -I scripts/83_gpt_agentdojo.py run         # dev attack + benign
-> outputs/agentdojo_gpt/{pilot,dev_gpt6luna_spot_attack,dev_gpt6luna_spot_benign}.jsonl (CAPE's row format)
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

GB = Path("/home/arya/graybox")
CAPE = Path("/home/arya/new_safety/cape")
_b = importlib.util.spec_from_file_location("gb_budget", GB / "src/graybox/budget.py")   # the ledger only (this env has no numpy)
_budget = importlib.util.module_from_spec(_b)
_b.loader.exec_module(_budget)
Ledger = _budget.Ledger

MODEL = "gpt-6-luna"
PRICE = (0.10 / 1e6, 0.50 / 1e6)
LEDGER, CAP = Path("/data/models/graybox_ws/outputs/openai/spend.json"), 5.00
OUT = Path("/data/models/graybox_ws/outputs/agentdojo_gpt")

spec = importlib.util.spec_from_file_location("cape191", CAPE / "scripts/191_v74_agentdojo.py")
m = importlib.util.module_from_spec(spec)
sys.argv_saved = sys.argv
spec.loader.exec_module(m)
m.ACTOR, m.OUT = "gpt6luna", OUT
m.SAMP["gpt6luna"], m.PROSE["gpt6luna"] = ({}, {}), "GPT"

import openai  # noqa: E402
from openai.types.chat import ChatCompletionMessage  # noqa: E402

CLIENT = openai.OpenAI(api_key=Path("/data/models/graybox_ws/.secrets/openai_key").read_text().strip(), max_retries=8)
LED = Ledger(LEDGER, CAP, *PRICE)


def query(self, query, runtime, env=None, messages=(), extra_args={}):
    rid = f"{self.arm}|{self.key}|{self.turn}"
    msgs = [m.to_plain(m._message_to_openai(x, MODEL)) for x in messages]
    tools = [m._function_to_openai(t) for t in runtime.functions.values()]
    LED.check()
    r = CLIENT.chat.completions.create(model=MODEL, messages=msgs, tools=tools, parallel_tool_calls=False, max_completion_tokens=2048,
                                       reasoning_effort="none", seed=int(hashlib.sha256(f"v74|{self.key}|{self.turn}".encode()).hexdigest(), 16) % 2**31)
    LED.charge(r.usage.prompt_tokens, r.usage.completion_tokens)
    msg = {k: v for k, v in r.choices[0].message.model_dump(exclude_none=True).items() if k in ("role", "content", "tool_calls")}
    msg.setdefault("content", None)
    out = m._openai_to_assistant_message(ChatCompletionMessage.model_validate(msg))
    self.transcript = msgs + [msg]
    tc = (msg.get("tool_calls") or [None])[0]
    self.calls.append({"turn": self.turn, "rid": rid, "tool": tc["function"]["name"] if tc else None,
                       "args": json.loads(tc["function"]["arguments"]) if tc else None, "call_id": tc["id"] if tc else None})
    self.turn += 1
    return query, runtime, env, [*messages, out], extra_args


m.CapeLLM.query = query

if __name__ == "__main__":
    import os
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "pilot":
        jobs = []
        for sn in m.SUITES:
            S = m.get_suites(m.VERSION)[sn]
            ut = sorted(u for u in S.user_tasks if m.split_of(sn, u) == "dev")[0]
            jobs += [(sn, ut, sorted(S.injection_tasks)[0], 0)]
        jobs += [(jobs[0][0], jobs[0][1], None, 0), (jobs[1][0], jobs[1][1], None, 0)]
        with open(OUT / "pilot.jsonl", "w") as f:
            for j in jobs:
                row = m.episode("spot", *j)
                f.write(json.dumps(row, default=str) + "\n")
                print(row["key"], "err", row["error"], "calls", len(row["calls"]), "security", row["security"], "spend", round(LED.total(), 4), flush=True)
    else:
        os.environ.setdefault("V74_CONC", "8")
        m.run("spot", "dev", "attack", "dev_gpt6luna_spot_attack", 1)
        m.run("spot", "dev", "benign", "dev_gpt6luna_spot_benign", 1)
