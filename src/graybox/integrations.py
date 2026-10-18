"""Grey Box in three agent frameworks: a Guard decides on each tool call before it runs.

Claude Agent SDK (a PreToolUse hook; the session transcript gives the history):

    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from graybox.integrations import claude_agent_hook
    hook = claude_agent_hook(guard, instructions=MY_SYSTEM_PROMPT)          # on_escalate="deny" (default) or "ask"
    options = ClaudeAgentOptions(hooks={"PreToolUse": [HookMatcher(hooks=[hook])]})

OpenAI Agents SDK (tool guardrails; the output guardrail records each call's result for later checks in the run):

    from graybox.integrations import openai_agents_guardrails
    check, record = openai_agents_guardrails(guard)
    tool = function_tool(my_fn, tool_input_guardrails=[check], tool_output_guardrails=[record])

LangGraph (a ToolNode wrapper; the graph state's messages give the history):

    from graybox.integrations import langgraph_tool_wrapper
    node = ToolNode(tools, wrap_tool_call=langgraph_tool_wrapper(guard))

A blocked call does not run: the agent gets Guard's neutral BLOCK_MESSAGE instead of the tool's result (the score and
the reason are never shown to the agent; see SECURITY.md). With on_escalate="ask" (Claude Agent SDK) an escalated call
goes to the user's permission prompt instead. Each adapter imports its framework lazily, so graybox never depends on it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .guard import BLOCK_MESSAGE, Guard, Verdict


# ---------------------------------------------------------------- Claude Agent SDK
def claude_transcript_messages(path) -> list[dict]:
    """Anthropic messages from a Claude Code / Claude Agent SDK session transcript (JSON lines). Entries of one assistant
    reply (one line per content block, same message id) are merged; sub-agent (sidechain) entries and non-message
    entries are skipped; unreadable lines are ignored."""
    msgs: list[dict] = []
    last_id = None
    p = Path(path)
    if not p.exists():
        return msgs
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or d.get("type") not in ("user", "assistant") or d.get("isSidechain"):
            continue
        m = d.get("message")
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [b for b in content or [] if isinstance(b, dict)]
        if m["role"] == "assistant" and msgs and msgs[-1]["role"] == "assistant" and m.get("id") and m.get("id") == last_id:
            msgs[-1]["content"].extend(blocks)
        else:
            msgs.append({"role": m["role"], "content": blocks})
        last_id = m.get("id") if m["role"] == "assistant" else None
    return msgs


def claude_agent_hook(guard: Guard, instructions: str = "", on_escalate: str = "deny"):
    """A PreToolUse hook callback for the Claude Agent SDK. instructions: the agent's system prompt (the transcript does
    not hold it). on_escalate: "deny" blocks an escalated call; "ask" sends it to the user's permission prompt.
    An allowed call returns no decision, so the SDK's own permission rules still apply."""
    if on_escalate not in ("deny", "ask"):
        raise ValueError('on_escalate must be "deny" or "ask"')

    async def hook(input_data, tool_use_id, context):
        if not isinstance(input_data, dict) or input_data.get("hook_event_name") != "PreToolUse":
            return {}
        msgs = claude_transcript_messages(input_data.get("transcript_path") or "")
        block = {"type": "tool_use", "id": input_data.get("tool_use_id") or tool_use_id,
                 "name": input_data.get("tool_name"), "input": input_data.get("tool_input") or {}}
        v = await guard.acheck_anthropic(instructions, msgs, block)
        if v.allow:
            return {}
        decision = "ask" if on_escalate == "ask" and v.review_error is None else "deny"   # a malformed call is always denied
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                       "permissionDecisionReason": v.block_message}}
    return hook


# ---------------------------------------------------------------- OpenAI Agents SDK
def openai_agents_guardrails(guard: Guard):
    """(input_guardrail, output_guardrail) for function tools of the OpenAI Agents SDK.

    The SDK gives a tool guardrail only the run's original input, not the calls made earlier in the run, so the output
    guardrail records every call and its result, and the input guardrail adds them to the history it checks. Records
    are kept per run (keyed by the run's usage object, which every tool call of a run shares) and per agent
    instructions. Attach both guardrails to every tool of the agent; a tool without the output guardrail leaves its
    calls out of the history."""
    from agents import ToolGuardrailFunctionOutput, tool_input_guardrail, tool_output_guardrail

    runs: dict[int, list] = {}

    def key(ctx):
        return id(getattr(ctx, "usage", ctx))

    def instructions_of(agent):
        ins = getattr(agent, "instructions", None)
        return ins if isinstance(ins, str) else None              # dynamic instructions are not resolved here

    @tool_input_guardrail(name="greybox_check")
    async def check(data):
        ctx = data.context
        call = ctx.tool_call or {"type": "function_call", "call_id": ctx.tool_call_id, "name": ctx.tool_name,
                                 "arguments": ctx.tool_arguments}
        items = list(ctx.turn_input or []) + runs.get(key(ctx), [])
        v = await guard.acheck_openai_responses(items, call, instructions=instructions_of(data.agent))
        if v.allow:
            return ToolGuardrailFunctionOutput.allow(output_info={"score": v.score})
        runs.setdefault(key(ctx), []).extend(_oa_items(ctx, BLOCK_MESSAGE))   # the agent sees the block in its history too
        return ToolGuardrailFunctionOutput.reject_content(v.block_message, output_info={"score": v.score, "held": v.held})

    @tool_output_guardrail(name="greybox_record")
    async def record(data):
        runs.setdefault(key(data.context), []).extend(_oa_items(data.context, data.output))
        return ToolGuardrailFunctionOutput.allow()

    return check, record


def _oa_items(ctx, output) -> list[dict]:
    return [{"type": "function_call", "call_id": ctx.tool_call_id, "name": ctx.tool_name, "arguments": ctx.tool_arguments},
            {"type": "function_call_output", "call_id": ctx.tool_call_id, "output": output if isinstance(output, str) else json.dumps(output, default=str)}]


# ---------------------------------------------------------------- LangGraph
def _lc_messages(state) -> list[dict]:
    """OpenAI chat messages from a LangGraph state (a dict with "messages", a list, or an object with .messages)."""
    from langchain_core.messages import convert_to_openai_messages
    msgs = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", state)
    return convert_to_openai_messages(list(msgs or []))


def langgraph_tool_wrapper(guard: Guard, async_: bool = False):
    """A wrap_tool_call (or, with async_=True, awrap_tool_call) handler for langgraph.prebuilt.ToolNode. A blocked call
    does not run; the agent gets a ToolMessage with BLOCK_MESSAGE and status "error"."""
    from langchain_core.messages import ToolMessage

    def to_openai(tc) -> dict:
        return {"id": tc.get("id"), "type": "function", "function": {"name": tc.get("name"), "arguments": json.dumps(tc.get("args") or {})}}

    def blocked(tc, v: Verdict):
        return ToolMessage(content=v.block_message, tool_call_id=tc.get("id"), name=tc.get("name"), status="error")

    def handler(request, execute):
        tc = request.tool_call
        v = guard.check_openai(_lc_messages(request.state), to_openai(tc))
        return execute(request) if v.allow else blocked(tc, v)

    async def ahandler(request, execute):
        tc = request.tool_call
        v = await guard.acheck_openai(_lc_messages(request.state), to_openai(tc))
        if not v.allow:
            return blocked(tc, v)
        out = execute(request)
        return await out if asyncio.iscoroutine(out) else out

    return ahandler if async_ else handler
