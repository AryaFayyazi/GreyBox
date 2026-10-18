"""Conversation-aware monitor context, for agents that talk with a user during the task (customer service, tau2-bench).

graybox.monitor.ctx_text keeps the instructions, the first user message and the tool calls. In a conversation the
facts that decide an action can come from later user turns (a cancellation reason, a confirmation). chat_ctx keeps
them, in order, in the same text format:
  [Operating instructions] / [User task] (the first user message) / [User] / [Agent said] / [Agent called] + [Tool result]
Tool results are cut to 2,000 characters; a context of 30,000 characters or more keeps its first 6,000 and last
24,000 characters (as ctx_text).
"""
from __future__ import annotations

import json


def chat_ctx(instructions: str, turns: list[dict]) -> str:
    """turns: dicts in order, each {"user": text} | {"agent": text} | {"call": (tool, args), "result": text}."""
    lines = [f"[Operating instructions]\n{instructions}"]
    first = True
    for t in turns:
        if "user" in t:
            lines.append(f"[User task]\n{t['user']}" if first else f"[User] {t['user']}")
            first = False
        elif "agent" in t:
            if t["agent"] and t["agent"].strip():
                lines.append(f"[Agent said] {t['agent']}")
        else:
            tool, args = t["call"]
            r = t.get("result") or ""
            lines.append(f"[Agent called] {tool} {json.dumps(args or {}, ensure_ascii=False)}")
            lines.append(f"[Tool result] {r[:2000]}{' ...' if len(r) > 2000 else ''}")
    s = "\n\n".join(lines)
    return s if len(s) < 30000 else s[:6000] + "\n...\n" + s[-24000:]


def openai_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    """(system prompt, turns) from OpenAI Chat Completions messages (tool_calls with JSON-string arguments)."""
    system = "\n\n".join(m.get("content") or "" for m in messages if m.get("role") in ("system", "developer"))
    results = {m.get("tool_call_id"): str(m.get("content") or "") for m in messages if m.get("role") == "tool"}
    turns = []
    for m in messages:
        r = m.get("role")
        if r == "user":
            turns.append({"user": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))})
        elif r == "assistant":
            turns.append({"agent": m.get("content") if isinstance(m.get("content"), str) else ""})
            for tc in m.get("tool_calls") or []:
                f = tc["function"]
                try:
                    args = json.loads(f.get("arguments") or "{}") if isinstance(f.get("arguments"), str) else f.get("arguments")
                except json.JSONDecodeError:
                    args = {"_raw": f.get("arguments")}
                turns.append({"call": (f["name"], args), "result": results.get(tc.get("id"), ""), "call_id": tc.get("id")})
    return system, turns
