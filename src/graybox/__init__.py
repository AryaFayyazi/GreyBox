"""Grey Box: read a black-box model's state from its answers, and guard agents through the API.

Light entry points (numpy only): graybox.sdk.Reader, graybox.sdk.AgentGuard. Research code (server, Othello, CAPE
bridge) needs the optional extras and is imported explicitly.
"""
from .sdk import AgentGuard, Reader, Reading  # noqa: F401

__version__ = "0.3.0"
