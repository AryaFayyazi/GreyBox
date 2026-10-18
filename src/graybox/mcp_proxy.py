"""Put a Grey Box monitor in front of any MCP server (stdio): every tool call is checked before it reaches the server.

    python -m graybox.mcp_proxy --monitor greybox-text-monitor.json --instructions policy.txt \\
        [--task "what the agent is asked to do"] [--hold-tools issue_refund,delete_db] [--skip-tools read_file,list_files] \\
        [--audit-log decisions.jsonl] -- <the upstream MCP server command ...>

Point your MCP client at this command instead of the upstream server. The proxy lists the upstream tools unchanged.
For each call it builds the step the monitor reads - the instructions, the task, the earlier calls of this session with
their results, and the proposed call - and asks graybox.guard.Guard. An escalated call is not forwarded: the client
gets an error result with the block message. Requires the [mcp] extra (mcp >= 2).

Limit: an MCP server never sees the agent's conversation. The proxy gives the monitor only what you pass with
--instructions and --task, and the calls it has seen. Trained monitors read the system prompt and the user's task too;
without them they are weaker (the docs give the measured cost). Use Guard in the agent loop when you can.
"""
from __future__ import annotations

import argparse
import sys


def _text(result) -> str:
    """The text of an MCP tool result (content blocks), as the monitor reads tool results."""
    parts = []
    for b in getattr(result, "content", None) or []:
        t = getattr(b, "text", None)
        parts.append(t if isinstance(t, str) else f"[{getattr(b, 'type', 'content')}]")
    return "\n".join(parts)


async def serve(a, upstream: list[str]) -> None:
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    from .guard import Guard
    from .monitor import TextMonitor

    instructions = open(a.instructions, encoding="utf-8").read() if a.instructions else ""
    guard = Guard(TextMonitor.load_json(a.monitor), hold_tools=[t for t in a.hold_tools.split(",") if t], audit_log=a.audit_log)
    skip = {t for t in a.skip_tools.split(",") if t}
    history: list[dict] = []
    params = StdioServerParameters(command=upstream[0], args=upstream[1:])
    async with stdio_client(params) as (r, w), ClientSession(r, w) as up:
        await up.initialize()
        tools = (await up.list_tools()).tools

        async def on_list_tools(ctx, params):
            return types.ListToolsResult(tools=tools)

        async def on_call_tool(ctx, params):
            name, args = params.name, dict(params.arguments or {})
            if name not in skip:
                v = guard.check(instructions, a.task, history, name, args)
                if not v.allow:
                    return types.CallToolResult(content=[types.TextContent(type="text", text=v.block_message)], is_error=True)
            res = await up.call_tool(name, args)
            history.append({"tool": name, "args": args, "result": _text(res)})
            return res

        server = Server("greybox-mcp-proxy", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
        async with stdio_server() as (sr, sw):
            await server.run(sr, sw, server.create_initialization_options())


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m graybox.mcp_proxy --monitor FILE [options] -- <upstream MCP server command>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--monitor", required=True, help="a text monitor file (browser trainer or graybox.train_text)")
    ap.add_argument("--instructions", default=None, help="a file with the agent's operating instructions (its system prompt)")
    ap.add_argument("--task", default="", help="the user's task, if fixed for this session")
    ap.add_argument("--hold-tools", default="", help="comma-separated tools whose every call is blocked for review")
    ap.add_argument("--skip-tools", default="", help="comma-separated read-only tools that are forwarded without a check")
    ap.add_argument("--audit-log", default=None, help="a JSON-lines file with one record per checked call")
    a = ap.parse_args(argv[:i])
    if not argv[i + 1:]:
        raise SystemExit("give the upstream MCP server command after --")
    try:
        import anyio
        import mcp  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit('the MCP proxy needs the [mcp] extra: pip install "greybox-monitor[mcp]"') from None
    anyio.run(serve, a, argv[i + 1:])


if __name__ == "__main__":
    main()
