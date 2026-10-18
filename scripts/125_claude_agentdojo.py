#!/usr/bin/env python
"""Claude Haiku 5.5 on AgentDojo WITHOUT a defense (arm 'none', AgentDojo's undefended pipeline), with CAPE's exact
harness (cape/scripts/191_v74_agentdojo.py: suites, user and injection tasks, important_instructions attack, per-call
'injected' labels = attacker-only argument strings), as scripts/83/85 ran GPT-6-luna. Only the model call changes:
Anthropic tool use, claude-haiku-5-5, thinking disabled, effort low, one tool call per turn (as T8b/T16b/T24).
The transcript is recorded in the OpenAI-plain format of scripts/83, so the T13b/T14 unit code applies unchanged.
Every call goes to the shared Anthropic ledger (outputs/l3/spend.json) with a hard cap.

  pilot  40 dev attack episodes, a fixed random sample (rng 125): cost and injected-call rate; never analysed
  run    every dev + conf attack episode and every benign episode, in ONE fixed random order (rng 1250), until the
         cap: a run cut short by the budget is a random subset, not the first suites
Runs in the AgentDojo env: /data/models/cape_ws/envs/agentdojo/bin/python -I scripts/125_claude_agentdojo.py pilot|run
"""
import hashlib
import importlib.util
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GB = Path("/home/arya/graybox")
CAPE = Path("/home/arya/new_safety/cape")
_b = importlib.util.spec_from_file_location("gb_budget", GB / "src/graybox/budget.py")   # the ledger only (no numpy here)
_budget = importlib.util.module_from_spec(_b)
_b.loader.exec_module(_budget)
Ledger = _budget.Ledger

MODEL = "claude-haiku-5-5"
PRICE = (0.10 / 1e6, 0.50 / 1e6)
LEDGER, CAP = Path("/data/models/graybox_ws/outputs/l3/spend.json"), 19.90       # the user's remaining balance (2026-10-18)
OUT = Path("/data/models/graybox_ws/outputs/agentdojo_claude")
STOP_MARGIN = 0.03                                                                 # no new episode within $0.03 of the cap

spec = importlib.util.spec_from_file_location("cape191", CAPE / "scripts/191_v74_agentdojo.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.ACTOR, m.OUT = "claudehaiku55", OUT
m.SAMP["claudehaiku55"], m.PROSE["claudehaiku55"] = ({}, {}), "Claude"

import anthropic  # noqa: E402
from agentdojo.agent_pipeline.llms.anthropic_llm import (_anthropic_to_assistant_message, _conversation_to_anthropic,  # noqa: E402
                                                          _function_to_anthropic)

CLIENT = anthropic.Anthropic(api_key=Path("/data/models/graybox_ws/.secrets/anthropic_key").read_text().strip(), max_retries=8)
LED = Ledger(LEDGER, CAP, *PRICE)


def query(self, query, runtime, env=None, messages=(), extra_args={}):
    rid = f"{self.arm}|{self.key}|{self.turn}"
    plain = [m.to_plain(m._message_to_openai(x, MODEL)) for x in messages]           # the transcript format of scripts/83
    system, amsgs = _conversation_to_anthropic(messages)
    tools = [_function_to_anthropic(t) for t in runtime.functions.values()]
    LED.check()
    r = CLIENT.messages.create(model=MODEL, max_tokens=2048, system=system or anthropic.NOT_GIVEN, messages=amsgs,
                               tools=tools or anthropic.NOT_GIVEN, tool_choice={"type": "auto", "disable_parallel_tool_use": True} if tools else anthropic.NOT_GIVEN,
                               thinking={"type": "disabled"}, output_config={"effort": "low"})
    LED.charge(r.usage.input_tokens, r.usage.output_tokens)
    out = _anthropic_to_assistant_message(r)
    msg = m.to_plain(m._message_to_openai(out, MODEL))
    self.transcript = plain + [msg]
    tc = (msg.get("tool_calls") or [None])[0]
    args = None
    if tc:
        a = tc["function"]["arguments"]
        args = json.loads(a) if isinstance(a, str) else a
    self.calls.append({"turn": self.turn, "rid": rid, "tool": tc["function"]["name"] if tc else None, "args": args,
                       "call_id": tc["id"] if tc else None, "stop_reason": r.stop_reason})
    self.turn += 1
    return query, runtime, env, [*messages, out], extra_args


m.CapeLLM.query = query


def all_jobs():
    jobs = []
    for split in ("dev", "conf"):
        for sn in m.SUITES:
            S = m.get_suites(m.VERSION)[sn]
            for ut in S.user_tasks:
                if m.split_of(sn, ut) != split:
                    continue
                jobs += [(split, "attack", sn, ut, it) for it in S.injection_tasks] + [(split, "benign", sn, ut, None)]
    return sorted(jobs, key=lambda j: (j[0], j[1], j[2], j[3], j[4] or ""))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "pilot":
        dev = [j for j in all_jobs() if j[0] == "dev" and j[1] == "attack"]
        jobs = random.Random(125).sample(dev, 40)
        fn, conc = OUT / "pilot.jsonl", 4
    else:
        jobs = all_jobs()
        random.Random(1250).shuffle(jobs)
        fn, conc = OUT / "run.jsonl", 6
    done = {json.loads(l)["key"] for l in open(fn)} if fn.exists() else set()
    todo = [j for j in jobs if f"{j[2]}|{j[3]}|{j[4] or 'benign'}|0" not in done]
    print(f"{fn.name}: {len(todo)} to run ({len(done)} done); spend ${LED.total():.3f} of cap ${CAP}", flush=True)
    lock, f, n = threading.Lock(), open(fn, "a"), [0]
    stop = threading.Event()

    def one(j):
        if stop.is_set() or LED.total() > CAP - STOP_MARGIN:
            stop.set()
            return
        split, kind, sn, ut, it = j
        row = m.episode("none", sn, ut, it, 0)
        row["split"], row["kind"] = split, kind
        if row["error"] and ("credit balance" in row["error"].lower() or "BudgetExceeded" in row["error"] or "cap" in row["error"].lower()):
            stop.set()                                                         # budget or account exhausted: do not record
            print("stopping:", row["error"][:160], flush=True)
            return
        with lock:
            f.write(json.dumps(row, default=str) + "\n"); f.flush(); n[0] += 1
            if n[0] % 20 == 0:
                print(f"  {n[0]}/{len(todo)} episodes; spend ${LED.total():.3f}", flush=True)

    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, todo))
    f.close()
    rows = [json.loads(l) for l in open(fn)]
    ok = [r for r in rows if r["error"] is None]
    inj = sum(bool(c.get("injected")) for r in ok for c in r["calls"])
    att = [r for r in ok if r.get("injection_task")]
    print(f"{fn.name}: {len(ok)}/{len(rows)} ok; attack success {sum(bool(r['security']) for r in att)}/{len(att)}; "
          f"injected calls {inj}; executed calls {sum(len(r['calls']) for r in ok)}; spend ${LED.total():.3f}", flush=True)


if __name__ == "__main__":
    main()
