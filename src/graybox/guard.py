"""Guard an agent loop: score each proposed tool call before it runs, and block what the monitor escalates.

Works with any monitor that has `escalate(instructions, task, history, action)` (TextMonitor from the browser trainer)
or with a ReaderMonitor plus a QuantileGate. Steps are built from the provider's own message format, with the same rule
as the website's importer (site/src/assets/tm-core.js importLogs): system prompt, first user message, earlier tool calls
with their results, and the proposed call.

OpenAI Chat Completions:
    from graybox.guard import Guard
    guard = Guard(TextMonitor.load_json("greybox-text-monitor.json"))
    msg = client.chat.completions.create(model=..., messages=messages, tools=tools).choices[0].message
    for call in msg.tool_calls or []:
        verdict = guard.check_openai(messages, call)          # before you execute the call
        result = run_tool(call) if verdict.allow else verdict.block_message

Anthropic Messages:
    for block in response.content:
        if block.type == "tool_use":
            verdict = guard.check_anthropic(system, messages, block)

Hold for review (T16c): Guard(monitor, reviewer=fn). An escalated action is held and fn(step) decides: True runs it
(verdict.allow, verdict.held), False blocks it. Without a reviewer an escalated action is blocked. In T16/T16b every
goal that blocking cost came from blocked SAFE actions; a reviewer who lets them run removes that cost.

Conversations with a user (chat=True): the monitor context keeps every user turn and agent message in order
(graybox.chat.chat_ctx), not only the first user message. Use it when the facts that decide an action come from the
user during the task (customer service). Train the monitor on steps made the same way (conversation_steps(chat=True)).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .chat import chat_ctx
from .monitor import Q, Step, act_text

BLOCK_MESSAGE = ("Blocked: a safety monitor flagged this action and sent it to a human reviewer. It was not executed. "
                 "Do not retry it. Continue with other steps, or tell the user what you need.")


def _text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                out.append(b.get("text") or (_text(b.get("content")) if b.get("type") == "tool_result" else "") or "")
        return "\n".join(x for x in out if x)
    if isinstance(c, dict):
        return c.get("text", "") or ""
    return str(c)


def _args(a) -> dict:
    if a is None:
        return {}
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a)
        return v if isinstance(v, dict) else {"_raw": a}
    except (TypeError, ValueError):
        return {"_raw": a}


def _as_dict(m):
    return m if isinstance(m, dict) else m.model_dump(exclude_none=True)


def conversation_steps(messages: list, system=None, labels: dict | None = None, chat: bool = False) -> list[dict]:
    """One step per tool call in an OpenAI or Anthropic conversation (dict messages or SDK objects).
    labels: optional {call_id: 0/1}; a step gets label None when its call has no label.
    chat: also give each step a "context" with every user turn and agent message before the call (graybox.chat)."""
    msgs = [_as_dict(m) for m in messages]
    sys_text = _text(system) if system else ""
    if not sys_text:
        s = next((m for m in msgs if m.get("role") in ("system", "developer")), None)
        sys_text = _text(s["content"]) if s else ""
    first = next((m for m in msgs if m.get("role") == "user" and _text(m.get("content")).strip()
                  and not (isinstance(m.get("content"), list) and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"]))), None)
    task = _text(first["content"]) if first else ""
    results = {}
    for m in msgs:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results[m["tool_call_id"]] = _text(m.get("content"))
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = _text(b.get("content"))
    steps, hist, turns = [], [], []
    for m in msgs:
        if m.get("role") == "user":
            c = m.get("content")
            if not (isinstance(c, list) and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)) and _text(c).strip():
                turns.append({"user": _text(c) if not isinstance(c, list) else "\n".join(_text(b) for b in c if not (isinstance(b, dict) and b.get("type") == "tool_result"))})
            continue
        if m.get("role") != "assistant":
            continue
        said = m.get("content") if isinstance(m.get("content"), str) else "\n".join(b.get("text", "") for b in m.get("content") or [] if isinstance(b, dict) and b.get("type") == "text")
        turns.append({"agent": said or ""})
        calls = [(tc.get("id"), (tc.get("function") or {}).get("name"), _args((tc.get("function") or {}).get("arguments"))) for tc in m.get("tool_calls") or []]
        if isinstance(m.get("content"), list):
            calls += [(b.get("id"), b.get("name"), _args(b.get("input"))) for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        for cid, name, args in calls:
            if not name:
                continue
            lab = (labels or {}).get(cid) if cid is not None else None
            st = {"instructions": sys_text, "task": task, "history": [dict(h) for h in hist], "action": {"tool": name, "args": args},
                  "call_id": cid, "label": None if lab is None else int(bool(lab))}
            if chat:
                st["context"] = chat_ctx(sys_text, turns)
            steps.append(st)
            hist.append({"tool": name, "args": args, "result": results.get(cid, "")})
            turns.append({"call": (name, args), "result": results.get(cid, "")})
    return steps


@dataclass
class Verdict:
    allow: bool
    score: float | None
    block_message: str = BLOCK_MESSAGE
    held: bool = False            # escalated and sent to the reviewer (allow then says what the reviewer decided)


class Guard:
    """monitor: a TextMonitor (uses its stored cut) or a ReaderMonitor (needs gate=QuantileGate(...)).
    reviewer: optional fn(step) -> bool for escalated actions (True = run it); chat: conversation-aware context."""

    def __init__(self, monitor, gate=None, reviewer=None, chat: bool | None = None):
        if chat is None:            # default: the setting the monitor was trained with (browser model files record it)
            chat = bool((getattr(monitor, "_json", None) or {}).get("chat", False))
        self.monitor, self.gate, self.reviewer, self.chat = monitor, gate, reviewer, chat

    def _escalated(self, step: dict) -> tuple[bool, float]:
        tool, args = step["action"]["tool"], step["action"].get("args") or {}
        if isinstance(step.get("context"), str):                   # a prebuilt (conversation-aware) context
            if self.gate is None:
                z = self.monitor.decision(None, None, None, None, rendered=(step["context"], act_text(tool, args)))
                return z >= self.monitor._json["threshold_logit"], z
            p = float(self.monitor.probe.predict_proba(self.monitor.features(Q.format(ctx=step["context"], cot="", act=act_text(tool, args))))[0, 1])
            return bool(self.gate(p)), p
        hist = [Step(h["tool"], h.get("args") or {}, h.get("result", "")) for h in step.get("history") or []]
        if self.gate is None:
            z = self.monitor.decision(step.get("instructions", ""), step.get("task", ""), hist, (tool, args))
            return z >= self.monitor._json["threshold_logit"], z
        p = self.monitor.score(step.get("instructions", ""), step.get("task", ""), hist, (tool, args))
        return bool(self.gate(p)), p

    def check_step(self, step: dict) -> Verdict:
        esc, score = self._escalated(step)
        if not esc:
            return Verdict(allow=True, score=score)
        if self.reviewer is None:
            return Verdict(allow=False, score=score)
        return Verdict(allow=bool(self.reviewer(step)), score=score, held=True)

    def check(self, instructions: str, task: str, history: list[dict], tool: str, args: dict) -> Verdict:
        return self.check_step({"instructions": instructions, "task": task, "history": history, "action": {"tool": tool, "args": args}})

    def _check_last(self, steps, call_id, name, args) -> Verdict:
        st = next((s for s in steps if s["call_id"] == call_id), None) if call_id else None
        st = st or {"instructions": steps[0]["instructions"] if steps else "", "task": steps[0]["task"] if steps else "", "history": [],
                    "action": {"tool": name, "args": args}}
        return self.check_step(st)

    def check_openai(self, messages: list, tool_call) -> Verdict:
        """messages: the conversation so far; tool_call: the proposed call (from the latest assistant message)."""
        tc = _as_dict(tool_call)
        name, args = (tc.get("function") or {}).get("name"), _args((tc.get("function") or {}).get("arguments"))
        prefix = [_as_dict(m) for m in messages] + [{"role": "assistant", "tool_calls": [tc]}]
        return self._check_last(conversation_steps(prefix, chat=self.chat), tc.get("id"), name, args)

    def check_anthropic(self, system, messages: list, tool_use) -> Verdict:
        """system: the system prompt; messages: the conversation so far; tool_use: the proposed tool_use block."""
        b = _as_dict(tool_use)
        prefix = [_as_dict(m) for m in messages] + [{"role": "assistant", "content": [b]}]
        return self._check_last(conversation_steps(prefix, system=system, chat=self.chat), b.get("id"), b.get("name"), _args(b.get("input")))
