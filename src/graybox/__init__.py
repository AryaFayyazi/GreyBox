"""Grey Box: monitor an AI agent's proposed tool calls from its transcript, before they run.

Main entry points (numpy only to score; training needs the [text] extra):
    from graybox import Guard, TextMonitor, QuantileGate, conversation_steps
    guard = Guard(TextMonitor.load_json("greybox-text-monitor.json"))
    v = guard.check_openai(messages, tool_call)          # or check_anthropic(system, messages, tool_use)
Train a monitor: python -m graybox.train_text labelled.jsonl (CPU) or in the browser trainer; the GPU reader:
graybox.train_reader / graybox.reader_load; the HTTP server: graybox.monitor_server. Docs: docs.html on the website.

graybox.sdk (Reader, AgentGuard) is the earlier research interface (black-box state from answers); graybox.server is
the research viewer for that interface. Neither is needed for monitoring.
"""
from .guard import Guard, Verdict, conversation_steps  # noqa: F401
from .monitor import QuantileGate, TextMonitor  # noqa: F401
from .sdk import AgentGuard, Reader, Reading  # noqa: F401  (research interface, kept for compatibility)

__version__ = "0.7.0"
