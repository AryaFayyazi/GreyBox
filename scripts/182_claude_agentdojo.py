#!/usr/bin/env python
"""Claude Haiku 5.5 on AgentDojo with CAPE's exact harness (cape/scripts/191_v74_agentdojo.py, v1.2.2): same suites, user
and injection tasks, important_instructions attack, per-call 'injected' labels. Only the model call changes (as
scripts/83 for GPT): Anthropic Messages API, thinking disabled, effort low, no parallel tool use. Every call is charged to
the shared Anthropic ledger and stops at the cap given on the command line. Runs in the AgentDojo env:
  /data/models/cape_ws/envs/agentdojo/bin/python -I scripts/182_claude_agentdojo.py <arm> <split> <kind> <cap>
  e.g. none dev attack 38.90   -> outputs/agentdojo_claude/<split>_claude_<arm>_<kind>.jsonl (CAPE's row format)
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

GB = Path("/home/arya/graybox")
CAPE = Path("/home/arya/new_safety/cape")
_b = importlib.util.spec_from_file_location("gb_budget", GB / "src/graybox/budget.py")
_budget = importlib.util.module_from_spec(_b)
_b.loader.exec_module(_budget)
Ledger, HAIKU55 = _budget.Ledger, _budget.HAIKU55

MODEL = os.environ.get("G17_MODEL", "claude-haiku-5-5")
PRICES = {"claude-haiku-5-5": None, "claude-haiku-4-5": (1.0 / 1e6, 5.0 / 1e6)}   # Haiku 4.5: list price, charged conservatively
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
OUT = Path("/data/models/graybox_ws/outputs/agentdojo_claude")
spec = importlib.util.spec_from_file_location("cape191", CAPE / "scripts/191_v74_agentdojo.py")
m = importlib.util.module_from_spec(spec)
sys.argv_saved = sys.argv
spec.loader.exec_module(m)
m.ACTOR, m.OUT = ("claude" if MODEL == "claude-haiku-5-5" else MODEL.replace("-", "")), OUT
m.SAMP[m.ACTOR], m.PROSE[m.ACTOR] = ({}, {}), "Claude"

import anthropic  # noqa: E402
from openai.types.chat import ChatCompletionMessage  # noqa: E402

CLIENT = anthropic.Anthropic(api_key=Path("/data/models/graybox_ws/.secrets/anthropic_key").read_text().strip(), max_retries=8)
LED = Ledger(LEDGER, float(sys.argv[4]) if len(sys.argv) > 4 else 0.0, *(PRICES.get(MODEL) or HAIKU55))


def to_anthropic(msgs):
    system, out = [], []

    def add(role, blocks):
        if out and out[-1]["role"] == role:
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": list(blocks)})
    for x in msgs:
        if x["role"] == "system":
            system.append(x.get("content") or "")
        elif x["role"] == "user":
            add("user", [{"type": "text", "text": x.get("content") or "(empty)"}])
        elif x["role"] == "assistant":
            b = [{"type": "text", "text": x["content"]}] if x.get("content") else []
            for tc in x.get("tool_calls") or []:
                b.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": json.loads(tc["function"].get("arguments") or "{}")})
            add("assistant", b or [{"type": "text", "text": "(no content)"}])
        elif x["role"] == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": x["tool_call_id"], "content": str(x.get("content") or "")}])
    return "\n\n".join(system), out


def query(self, query, runtime, env=None, messages=(), extra_args={}):
    rid = f"{self.arm}|{self.key}|{self.turn}"
    msgs = [m.to_plain(m._message_to_openai(x, MODEL)) for x in messages]
    tools = [m._function_to_openai(t)["function"] for t in runtime.functions.values()]
    atools = [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]} for t in tools]
    system, amsgs = to_anthropic(msgs)
    LED.check()
    extra = {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}} if MODEL == "claude-haiku-5-5" else {}
    r = CLIENT.messages.create(model=MODEL, system=system, messages=amsgs, tools=atools, max_tokens=2048,
                               tool_choice={"type": "auto", "disable_parallel_tool_use": True}, **extra)
    LED.charge(r.usage.input_tokens, r.usage.output_tokens)
    text = "".join(b.text for b in r.content if b.type == "text")
    uses = [b for b in r.content if b.type == "tool_use"]
    msg = {"role": "assistant", "content": text or None}
    if uses:
        msg["tool_calls"] = [{"id": u.id, "type": "function", "function": {"name": u.name, "arguments": json.dumps(u.input)}} for u in uses[:1]]
    out = m._openai_to_assistant_message(ChatCompletionMessage.model_validate(msg))
    self.transcript = msgs + [msg]
    tc = (msg.get("tool_calls") or [None])[0]
    self.calls.append({"turn": self.turn, "rid": rid, "tool": tc["function"]["name"] if tc else None,
                       "args": json.loads(tc["function"]["arguments"]) if tc else None, "call_id": tc["id"] if tc else None})
    self.turn += 1
    return query, runtime, env, [*messages, out], extra_args


m.CapeLLM.query = query

ATTACK = os.environ.get("G17_ATTACK")                  # AgentDojo's registered attack (default: CAPE's important_instructions)
if ATTACK:
    import agentdojo.attacks.base_attacks as _ba
    from agentdojo.attacks.attack_registry import load_attack
    _ba.get_model_name_from_pipeline = lambda pipeline: "Claude"          # the model's name inside the injection text
    for _mod_name in ("agentdojo.attacks.important_instructions_attacks", "agentdojo.attacks.baseline_attacks"):
        try:
            __import__(_mod_name, fromlist=["x"]).get_model_name_from_pipeline = _ba.get_model_name_from_pipeline
        except Exception:  # noqa: BLE001
            pass

    class _Registered:
        def __init__(self, suite, p):
            self.a = load_attack(ATTACK, suite, p)

        def attack(self, user_task, injection_task):
            return self.a.attack(user_task, injection_task)
    m.Attack = _Registered

if __name__ == "__main__":
    arm, split, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("V74_CONC", "6")
    print("ledger", round(LED.total(), 3), "cap", sys.argv[4], flush=True)
    m.run(arm, split, kind, f"{split}_claude_{arm}_{kind}" + (f"_{ATTACK}" if ATTACK else ""), 1)
    print("ledger", round(LED.total(), 3), flush=True)
