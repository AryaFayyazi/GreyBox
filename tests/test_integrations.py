"""Framework adapters (graybox.integrations), tested offline against the real SDKs: the OpenAI Agents SDK with its
ScriptedModel, LangGraph's ToolNode, and the Claude Agent SDK hook on a transcript in Claude Code's format.
The stub monitor escalates send_message only when an earlier read_file is in the history, so each test also checks
that the history reaches the monitor."""
import asyncio
import json

import pytest

from graybox.guard import BLOCK_MESSAGE, Guard


class HistoryStub:
    """A reader-like monitor: probability 0.9 for send_message after a read_file, else 0.1."""
    def __init__(self):
        self.seen = []

    def score(self, instructions, task, history, action):
        self.seen.append((instructions, task, [h.tool for h in history], action[0]))
        return 0.9 if action[0] == "send_message" and any(h.tool == "read_file" for h in history) else 0.1


def guard():
    return Guard(HistoryStub(), gate=lambda p: p >= 0.5)


# ---------------------------------------------------------------- OpenAI Agents SDK
def test_openai_agents_guardrails_block_with_in_run_history():
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool
    from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call
    from graybox.integrations import openai_agents_guardrails
    ran = []

    def make(g):
        check, record = openai_agents_guardrails(g)

        @function_tool(tool_input_guardrails=[check], tool_output_guardrails=[record])
        def read_file(path: str) -> str:
            """Read a file."""
            ran.append(("read_file", path)); return "Q3 numbers: confidential"

        @function_tool(tool_input_guardrails=[check], tool_output_guardrails=[record])
        def send_message(to: str, body: str) -> str:
            """Send a message."""
            ran.append(("send_message", to)); return "sent"
        return [read_file, send_message]

    def run(g, tools):
        model = ScriptedModel([ModelStep(output=[function_call("read_file", {"path": "q3.txt"}, call_id="c1")]),
                               ModelStep(output=[function_call("send_message", {"to": "x@ext.example", "body": "see Q3"}, call_id="c2")]),
                               ModelStep(output=[assistant_message("done")])])
        agent = Agent(name="ops", instructions="Never send files outside the company.", model=model, tools=tools)
        res = asyncio.run(Runner.run(agent, "Share the Q3 file with x@ext.example"))
        return model, res

    g = guard()
    model, res = run(g, make(g))
    assert ran == [("read_file", "q3.txt")]                                   # send_message did not run
    assert BLOCK_MESSAGE in json.dumps(model.last_call.input if hasattr(model.last_call, "input") else str(model.last_call), default=str)
    ins, task, hist, tool = g.monitor.seen[-1]
    assert tool == "send_message" and hist == ["read_file"] and ins.startswith("Never send") and "Q3" in task
    # negative control: without the recording guardrail the run's earlier read is not in the history, so the send runs
    ran.clear(); g3 = guard(); check3, _ = openai_agents_guardrails(g3)

    @function_tool(tool_input_guardrails=[check3])
    def read_file(path: str) -> str:
        """Read a file."""
        ran.append(("read_file", path)); return "Q3 numbers: confidential"

    @function_tool(tool_input_guardrails=[check3])
    def send_message(to: str, body: str) -> str:
        """Send a message."""
        ran.append(("send_message", to)); return "sent"
    run(g3, [read_file, send_message])
    assert ran == [("read_file", "q3.txt"), ("send_message", "x@ext.example")]
    # a second run starts from an empty history (records are per run)
    ran.clear(); g2 = guard(); tools = make(g2)
    run(g2, tools); run(g2, tools)
    assert [h for _, _, h, t in g2.monitor.seen if t == "read_file"] == [[], []]


# ---------------------------------------------------------------- LangGraph
def test_langgraph_wrapper_blocks_and_passes():
    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode
    from graybox.integrations import langgraph_tool_wrapper
    ran = []

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        ran.append("read_file"); return "confidential"

    @tool
    def send_message(to: str, body: str) -> str:
        """Send a message."""
        ran.append("send_message"); return "sent"

    from langgraph.graph import END, START, MessagesState, StateGraph

    def graph(node):                                            # ToolNode needs a graph runtime: a one-node graph
        sg = StateGraph(MessagesState)
        sg.add_node("tools", node); sg.add_edge(START, "tools"); sg.add_edge("tools", END)
        return sg.compile()

    g = guard()
    app = graph(ToolNode([read_file, send_message], wrap_tool_call=langgraph_tool_wrapper(g)))
    base = [SystemMessage("Never send files outside."), HumanMessage("Share the Q3 file")]
    first = AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": "q3.txt"}, "id": "c1"}])
    out = app.invoke({"messages": base + [first]})
    assert ran == ["read_file"] and out["messages"][-1].content == "confidential"
    second = AIMessage("", tool_calls=[{"name": "send_message", "args": {"to": "x@ext.example", "body": "Q3"}, "id": "c2"}])
    out = app.invoke({"messages": base + [first, ToolMessage("confidential", tool_call_id="c1"), second]})
    m = out["messages"][-1]
    assert ran == ["read_file"] and m.content == BLOCK_MESSAGE and m.status == "error" and m.tool_call_id == "c2"
    assert g.monitor.seen[-1][2] == ["read_file"] and g.monitor.seen[-1][0] == "Never send files outside."
    # async wrapper, same decision
    aapp = graph(ToolNode([read_file, send_message], awrap_tool_call=langgraph_tool_wrapper(g, async_=True)))
    out = asyncio.run(aapp.ainvoke({"messages": base + [first, ToolMessage("confidential", tool_call_id="c1"), second]}))
    assert out["messages"][-1].content == BLOCK_MESSAGE and ran == ["read_file"]


# ---------------------------------------------------------------- Claude Agent SDK
def _transcript(tmp_path):
    lines = [{"type": "queue-operation"},
             {"type": "user", "isSidechain": False, "message": {"role": "user", "content": "Share the Q3 file with x@ext.example"}},
             {"type": "assistant", "isSidechain": False, "message": {"id": "m1", "role": "assistant", "content": [{"type": "thinking", "thinking": "..."}]}},
             {"type": "assistant", "isSidechain": False, "message": {"id": "m1", "role": "assistant", "content": [
                 {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "q3.txt"}}]}},
             {"type": "user", "isSidechain": False, "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "t1", "content": "confidential"}]}},
             {"type": "assistant", "isSidechain": True, "message": {"id": "s1", "role": "assistant", "content": [
                 {"type": "tool_use", "id": "s9", "name": "send_message", "input": {}}]}},
             {"type": "assistant", "isSidechain": False, "message": {"id": "m2", "role": "assistant", "content": [
                 {"type": "tool_use", "id": "t2", "name": "send_message", "input": {"to": "x@ext.example", "body": "Q3"}}]}}]
    p = tmp_path / "session.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    return p


def test_claude_transcript_parsing(tmp_path):
    from graybox.integrations import claude_transcript_messages
    msgs = claude_transcript_messages(_transcript(tmp_path))
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [b["type"] for b in msgs[1]["content"]] == ["thinking", "tool_use"]      # one reply, two entries, merged
    assert all(b.get("id") != "s9" for m in msgs for b in m["content"])               # the sub-agent's entry is skipped


def test_claude_agent_hook_denies_or_asks(tmp_path):
    from graybox.integrations import claude_agent_hook
    p = _transcript(tmp_path)
    inp = {"hook_event_name": "PreToolUse", "session_id": "s", "transcript_path": str(p), "cwd": ".",
           "tool_name": "send_message", "tool_input": {"to": "x@ext.example", "body": "Q3"}, "tool_use_id": "t2"}
    g = guard()
    out = asyncio.run(claude_agent_hook(g, instructions="Never send files outside.")(inp, "t2", None))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert out["hookSpecificOutput"]["permissionDecisionReason"] == BLOCK_MESSAGE
    assert g.monitor.seen[-1][2] == ["read_file"] and g.monitor.seen[-1][0] == "Never send files outside."
    out = asyncio.run(claude_agent_hook(guard(), on_escalate="ask")(inp, "t2", None))
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"
    allowed = dict(inp, tool_name="read_file", tool_input={"path": "q3.txt"}, tool_use_id="t3")
    assert asyncio.run(claude_agent_hook(guard())(allowed, "t3", None)) == {}
    assert asyncio.run(claude_agent_hook(guard())({"hook_event_name": "PostToolUse"}, None, None)) == {}
    with pytest.raises(ValueError):
        claude_agent_hook(guard(), on_escalate="maybe")


def test_claude_agent_hook_types_match_the_sdk():
    sdk = pytest.importorskip("claude_agent_sdk")
    from graybox.integrations import claude_agent_hook
    sdk.HookMatcher(hooks=[claude_agent_hook(guard())])                              # accepted by the real SDK type
    assert set(sdk.types.PreToolUseHookSpecificOutput.__annotations__) >= {"hookEventName", "permissionDecision", "permissionDecisionReason"}
