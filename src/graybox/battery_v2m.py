"""Battery v2m (2026-10-16, before any v2m data): the 88 items of battery v2 with a preamble that pins every question
to the user's LATEST message. Fixes the multi-turn carry-over seen in the first live session (questions about "the
message above" were answered about the whole conversation)."""
import hashlib
import json

from .battery_v2 import BATTERY_V2

PREAMBLE = ("\n\n---\n[Before you respond, answer one question about the user's LATEST message only (the one just "
            "above this line), ignoring earlier turns of the conversation. Answer with a single word, Yes or No.]\n")
BATTERY_V2M = BATTERY_V2


def battery_sha(items=BATTERY_V2M, preamble=PREAMBLE) -> str:
    return hashlib.sha256(json.dumps({"preamble": preamble, "items": items}, sort_keys=True).encode()).hexdigest()
