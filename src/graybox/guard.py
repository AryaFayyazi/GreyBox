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

BLOCK_MESSAGE = ("Blocked: a safety monitor flagged this action. It was not executed. "
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
    """Plain JSON-like data from SDK objects, recursively: a dict message may hold SDK content blocks (the Anthropic SDK's
    documented loop appends response.content, a list of TextBlock / ToolUseBlock objects). A shallow conversion lost
    every earlier call and result there (a user test: 1 of 68 violations escalated instead of 29)."""
    if isinstance(m, dict):
        return {k: _as_dict(v) for k, v in m.items()}
    if isinstance(m, (list, tuple)):
        return [_as_dict(v) for v in m]
    if hasattr(m, "model_dump"):
        return _as_dict(m.model_dump(exclude_none=True))
    if hasattr(m, "to_dict") and not isinstance(m, (str, bytes)):
        return _as_dict(m.to_dict())
    return m


def canonical_step(step: dict) -> dict:
    """A step with the canonical form of its action and of every earlier call (graybox.monitor.canonical_args): free text
    that the agent writes about its own calls - bodies, subjects, reasons, comments - is replaced, so a rewrite of it cannot
    move the score. A prebuilt context string gets the same treatment line by line."""
    from .monitor import canonical_act_text, canonical_args, canonical_ctx_text
    out = dict(step)
    a = step["action"]
    out["action"] = canonical_act_text(a) if isinstance(a, str) else {**a, "args": canonical_args(a.get("args") or {})}
    out["history"] = [{**h, "args": canonical_args(h.get("args") or {})} for h in step.get("history") or []]
    if isinstance(step.get("context"), str):
        out["context"] = canonical_ctx_text(step["context"])
    return out


def responses_to_chat(items: list, instructions: str | None = None) -> list[dict]:
    """OpenAI Responses API input items -> Chat Completions messages (consecutive function_call items form one
    assistant reply, as parallel calls do)."""
    out = [{"role": "system", "content": instructions}] if instructions else []
    if isinstance(items, str):                  # the Responses API also accepts a plain string as input
        items = [{"role": "user", "content": items}]
    for it in (_as_dict(x) for x in items or []):
        kind = it.get("type")
        if kind == "function_call":
            call = {"id": it.get("call_id") or it.get("id"), "type": "function",
                    "function": {"name": it.get("name"), "arguments": it.get("arguments") or "{}"}}
            if out and out[-1].get("role") == "assistant" and out[-1].get("tool_calls") and not out[-1].get("content"):
                out[-1]["tool_calls"].append(call)
            else:
                out.append({"role": "assistant", "tool_calls": [call]})
        elif kind == "function_call_output":
            o = it.get("output")
            out.append({"role": "tool", "tool_call_id": it.get("call_id"), "content": o if isinstance(o, str) else json.dumps(o)})
        elif it.get("role"):
            c = it.get("content")
            if isinstance(c, list):
                c = "\n".join(str(b.get("text", "")) for b in c if isinstance(b, dict) and "text" in b)
            out.append({"role": "system" if it["role"] == "developer" else it["role"], "content": c or ""})
    return out


def parse_label(v):
    """A label value: 0/1, True/False, or the strings "0", "1", "true", "false"; anything else is None (not guessed)."""
    if v is True or v == 1 and not isinstance(v, str) or v in ("1", "true"):
        return 1
    if v is False or v == 0 and not isinstance(v, str) or v in ("0", "false"):
        return 0
    return None


def conversation_steps(messages: list, system=None, labels: dict | None = None, chat: bool = False) -> list[dict]:
    """One step per tool call in an OpenAI or Anthropic conversation (dict messages or SDK objects).
    labels: optional {call_id: 0/1}; a step gets label None when its call has no label (or an invalid one).
    chat: also give each step a "context" with every user turn and agent message before the call (graybox.chat).

    A step sees everything before the assistant reply that holds its call, plus the EARLIER calls of the same reply
    with empty results (no call of a reply runs before the reply is complete). The reply's own text is part of the
    context of later replies only. The browser importer (site/src/assets/tm-core.js) uses the same rule."""
    msgs = [_as_dict(m) for m in messages]
    system = _as_dict(system) if system else system
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
        calls = [(tc.get("id"), (tc.get("function") or {}).get("name"), _args((tc.get("function") or {}).get("arguments"))) for tc in m.get("tool_calls") or []]
        if isinstance(m.get("content"), list):
            calls += [(b.get("id"), b.get("name"), _args(b.get("input"))) for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        calls = [c for c in calls if c[1]]
        for k, (cid, name, args) in enumerate(calls):
            sib = [{"tool": n, "args": a, "result": ""} for _, n, a in calls[:k]]
            lab = parse_label((labels or {}).get(cid)) if cid is not None else None
            st = {"instructions": sys_text, "task": task, "history": [dict(h) for h in hist] + sib, "action": {"tool": name, "args": args},
                  "call_id": cid, "label": lab}
            if chat:
                st["context"] = chat_ctx(sys_text, turns + [{"call": (h["tool"], h["args"]), "result": ""} for h in sib])
            steps.append(st)
        turns.append({"agent": said or ""})
        for cid, name, args in calls:
            hist.append({"tool": name, "args": args, "result": results.get(cid, "")})
            turns.append({"call": (name, args), "result": results.get(cid, "")})
    return steps


@dataclass
class Verdict:
    allow: bool
    score: float | None
    block_message: str = BLOCK_MESSAGE
    held: bool = False            # escalated and sent to the reviewer (allow then says what the reviewer decided)
    unit: str = "logit"           # "logit" for text and browser-reader monitors (compared with threshold_logit); "probability" for the reader
    review_error: str | None = None   # the reviewer raised or did not answer True/False: the action is blocked (fail closed)
    by_rule: bool = False         # escalated because its tool is in hold_tools (whatever the score)


class Guard:
    """monitor: a TextMonitor (uses its stored cut) or a ReaderMonitor (needs gate=QuantileGate(...)).
    reviewer: optional fn(step) -> bool for escalated actions (True = run it); chat: conversation-aware context.
    on_decision: optional fn(record) called after every check; audit_log: a JSON-lines file that gets one record per check
    (time, tool, args, score, unit, allow, held, by_rule, review_error, threshold). Neither can change a decision.
    hold_tools: tools whose every call is escalated (an explicit rule, e.g. {"issue_refund"}), on top of the score; the
    gate still sees their scores. Prefer this to a budget per tool: offline, a 5% budget per tool caught 39% / 46% of
    Claude Haiku 5.5's / GPT's violations against 94% / 98% with one budget, because violations cluster in a few tools."""

    def __init__(self, monitor, gate=None, reviewer=None, chat: bool | None = None, hold_tools=(), reviewer_timeout: float | None = None,
                 on_decision=None, audit_log: str | None = None):
        if chat is None:            # default: the setting the monitor was trained with (model files and probe files record it)
            chat = bool((getattr(monitor, "_json", None) or {}).get("chat", getattr(monitor, "chat", False)))
        self.monitor, self.gate, self.reviewer, self.chat = monitor, gate, reviewer, chat
        # canonical: the monitor was trained on canonical actions (graybox.monitor.canonical_args); the model file records it
        self.canonical = bool((getattr(monitor, "_json", None) or {}).get("canonical", getattr(monitor, "canonical", False)))
        if isinstance(hold_tools, str):          # one tool name, not its characters (a string used to hold nothing: fail open)
            hold_tools = [hold_tools]
        self.hold_tools = frozenset(hold_tools or ())
        self.reviewer_timeout = reviewer_timeout      # seconds; no answer in time blocks the action (fail closed)
        self.on_decision, self.audit_log = on_decision, audit_log   # audit: a callback and/or a JSON-lines file, one record per check

    def _is_text(self) -> bool:
        """Text and browser-reader monitors score in logits (decision); the reader scores probabilities."""
        return getattr(self.monitor, "_json", None) is not None and hasattr(self.monitor, "decision")

    def _unit(self) -> str:
        return "logit" if self._is_text() or self.gate is None else "probability"

    def _escalated(self, step: dict) -> tuple[bool, float]:
        if self.canonical:                                          # the monitor reads the canonical action (free text removed)
            step = canonical_step(step)
        tool, args = step["action"]["tool"], step["action"].get("args") or {}
        ctx = step.get("context") if isinstance(step.get("context"), str) else None   # a prebuilt (conversation-aware) context
        hist = [Step(h["tool"], h.get("args") or {}, h.get("result", "")) for h in step.get("history") or []]
        if self._is_text():                                         # text / browser reader: a logit, gated on logits
            z = self.monitor.decision(step.get("instructions", ""), step.get("task", ""), hist, (tool, args),
                                      rendered=(ctx, act_text(tool, args)) if ctx is not None else None)
            return (z >= self.monitor._json["threshold_logit"] if self.gate is None else bool(self.gate(z))), z
        if ctx is not None:                                         # the reader: a probability, gated on probabilities
            p = float(self.monitor.probe.predict_proba(self.monitor.features(Q.format(ctx=ctx, cot="", act=act_text(tool, args))))[0, 1])
        else:
            p = self.monitor.score(step.get("instructions", ""), step.get("task", ""), hist, (tool, args))
        return bool(self.gate(p)), p

    def check_step(self, step: dict) -> Verdict:
        v = self._check_step(step)
        if self.on_decision is not None or self.audit_log:
            self._audit(step, v)
        return v

    def _audit(self, step: dict, v: Verdict) -> None:
        """One record per decision: time, tool, arguments, score, decision. A failing callback or log never changes the
        decision (it is reported on stderr). The arguments can hold personal data: the log stays on your machine."""
        import sys
        import time
        rec = {"time": time.time(), "tool": step["action"]["tool"], "args": step["action"].get("args") or {}, "call_id": step.get("call_id"),
               "score": v.score, "unit": v.unit, "allow": v.allow, "held": v.held, "by_rule": v.by_rule, "review_error": v.review_error,
               "threshold": getattr(self.gate, "threshold", None) if self.gate is not None else (getattr(self.monitor, "_json", None) or {}).get("threshold_logit")}
        try:
            if self.on_decision is not None:
                self.on_decision(rec)
            if self.audit_log:
                with open(self.audit_log, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"graybox: audit failed ({type(e).__name__}: {e}); the decision stands", file=sys.stderr)

    def _check_step(self, step: dict) -> Verdict:
        esc, score = self._escalated(step)
        unit, rule = self._unit(), step["action"]["tool"] in self.hold_tools
        if not (esc or rule):
            return Verdict(allow=True, score=score, unit=unit)
        if self.reviewer is None:
            return Verdict(allow=False, score=score, unit=unit, by_rule=rule)
        try:                                   # fail closed: only an explicit True runs the action
            answer = self._ask({**{k: v for k, v in step.items() if k != "label"}, "score": score, "unit": unit})
        except Exception as e:  # noqa: BLE001
            return Verdict(allow=False, score=score, held=True, unit=unit, review_error=f"{type(e).__name__}: {e}", by_rule=rule)
        if type(answer).__name__ == "bool_":       # numpy.bool_ from a reviewer that thresholds an array
            answer = bool(answer)
        if not isinstance(answer, bool):
            return Verdict(allow=False, score=score, held=True, unit=unit, review_error=f"reviewer returned {answer!r}, not True/False", by_rule=rule)
        return Verdict(allow=answer, score=score, held=True, unit=unit, by_rule=rule)

    def _ask(self, step):
        if self.reviewer_timeout is None:
            return self.reviewer(step)
        import threading
        box = {}

        def run():
            try:
                box["answer"] = self.reviewer(step)
            except Exception as e:  # noqa: BLE001
                box["error"] = e
        t = threading.Thread(target=run, daemon=True)     # a daemon: a reviewer that never answers does not keep the process alive
        t.start()
        t.join(self.reviewer_timeout)
        if t.is_alive():
            raise TimeoutError(f"no answer within {self.reviewer_timeout} s")
        if "error" in box:
            raise box["error"]
        return box.get("answer")

    async def acheck_openai(self, messages: list, tool_call) -> Verdict:
        """check_openai for asyncio agents: the check runs in a worker thread, so the event loop is not blocked."""
        import asyncio
        return await asyncio.to_thread(self.check_openai, messages, tool_call)

    async def acheck_anthropic(self, system, messages: list, tool_use) -> Verdict:
        import asyncio
        return await asyncio.to_thread(self.check_anthropic, system, messages, tool_use)

    async def acheck_openai_responses(self, input_items: list, function_call, instructions: str | None = None) -> Verdict:
        import asyncio
        return await asyncio.to_thread(self.check_openai_responses, input_items, function_call, instructions)

    def check_openai_responses(self, input_items: list, function_call, instructions: str | None = None) -> Verdict:
        """OpenAI Responses API: input_items are the conversation items (messages, function_call and function_call_output
        items, as you pass them in `input`); function_call is the proposed function_call item; instructions: the
        request's `instructions` (the system prompt). Converted to Chat Completions messages, then check_openai."""
        msgs, fc = responses_to_chat(input_items, instructions), _as_dict(function_call)
        call = {"id": fc.get("call_id") or fc.get("id"), "type": "function",
                "function": {"name": fc.get("name"), "arguments": fc.get("arguments") or "{}"}}
        return self.check_openai(msgs, call)

    def check(self, instructions: str, task: str, history: list[dict], tool: str, args: dict) -> Verdict:
        return self.check_step({"instructions": instructions, "task": task, "history": history, "action": {"tool": tool, "args": args}})

    @staticmethod
    def _prefix(msgs: list, call_id, name, args, as_reply) -> list:
        """The conversation as the importer sees it for this call: everything before the assistant reply that holds the
        call, then that reply cut after the call (its earlier calls are siblings with no results yet).
        The reply can only be the LATEST assistant message, and only if it holds a call with this tool and these
        arguments (and this id, when both have one) that has no result yet; otherwise the messages do not include
        the reply, and the call is appended as a reply of its own (pass the reply for parallel calls)."""
        done = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
        done |= {b.get("tool_use_id") for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), list) for b in m["content"] if isinstance(b, dict)}
        k = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
        if k is not None:
            m = msgs[k]
            calls = [(tc.get("id"), (tc.get("function") or {}).get("name"), _args((tc.get("function") or {}).get("arguments")), ("oa", tc))
                     for tc in m.get("tool_calls") or []]
            if isinstance(m.get("content"), list):
                calls += [(b.get("id"), b.get("name"), _args(b.get("input")), ("an", b)) for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
            for i, (cid, n, a, _) in enumerate(calls):
                if n == name and a == args and (not call_id or not cid or cid == call_id) and (cid is None or cid not in done):
                    keep = [x[3] for x in calls[: i + 1]]
                    reply = {"role": "assistant", "tool_calls": [c for t, c in keep if t == "oa"]}
                    if any(t == "an" for t, _ in keep):
                        reply = {"role": "assistant", "content": [c for t, c in keep if t == "an"]}
                    return msgs[:k] + [reply]
        return msgs + [as_reply]

    def _check_proposed(self, steps, name, args) -> Verdict:
        st = steps[-1] if steps else None
        if not name or st is None or st["action"]["tool"] != name:   # malformed call: fail closed, never score another action
            v = Verdict(allow=False, score=None, unit=self._unit(), review_error=f"malformed tool call (tool name {name!r}): blocked")
            if self.on_decision is not None or self.audit_log:
                self._audit({"action": {"tool": name, "args": args if isinstance(args, dict) else {}}}, v)
            return v
        return self.check_step(st)

    def check_openai(self, messages: list, tool_call) -> Verdict:
        """messages: the conversation (with or without the reply that holds the call); tool_call: the proposed call.
        The monitor sees everything before that reply, then the call."""
        tc = _as_dict(tool_call)
        name, args = (tc.get("function") or {}).get("name"), _args((tc.get("function") or {}).get("arguments"))
        prefix = self._prefix([_as_dict(m) for m in messages], tc.get("id"), name, args, {"role": "assistant", "tool_calls": [tc]})
        return self._check_proposed(conversation_steps(prefix, chat=self.chat), name, args)

    def check_anthropic(self, system, messages: list, tool_use) -> Verdict:
        """system: the system prompt; messages: the conversation (with or without the reply that holds the block);
        tool_use: the proposed tool_use block. The monitor sees everything before that reply, then the call."""
        b = _as_dict(tool_use)
        prefix = self._prefix([_as_dict(m) for m in messages], b.get("id"), b.get("name"), _args(b.get("input")), {"role": "assistant", "content": [b]})
        return self._check_proposed(conversation_steps(prefix, system=system, chat=self.chat), b.get("name"), _args(b.get("input")))
