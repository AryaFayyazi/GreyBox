"""Grey Box: monitor a closed AI agent through its own API, with no training.

The method (training-free, API only):
    from graybox.selfaudit import observable_state, openai_asker, anthropic_asker
    st = observable_state(ask, messages, "send_email", {"to": "x@partner.com"}, n=4)
    st.score     # fixed read-out of the agent's answers to 8 yes/no questions; hold above your traffic's 95th percentile
    from graybox.selfaudit import SelfAudit       # one-question gate (self-policing)
The observable geometry (numpy only):
    from graybox.geometry import interrogate, crossfit_distances, mds     # or the command `graybox-geometry`
Trained monitors, kept as comparisons and options (they need labels):
    from graybox import Guard, TextMonitor, QuantileGate, conversation_steps
Train one: python -m graybox.train_text labelled.jsonl (CPU), the browser trainer, or graybox.train_reader (GPU);
the HTTP server: graybox.monitor_server. graybox.sdk (Reader, AgentGuard) is the earlier research interface.
"""
from .guard import Guard, Verdict, conversation_steps  # noqa: F401
from .monitor import QuantileGate, TextMonitor  # noqa: F401
from .sdk import AgentGuard, Reader, Reading  # noqa: F401  (research interface, kept for compatibility)

__version__ = "0.12.0"
