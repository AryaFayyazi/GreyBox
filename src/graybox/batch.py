"""One-call battery: every question in a single side-branch message; the model answers all of them, one per line.

batch_block(battery) -> the numbered question list; parse_batch(text, k) -> (k, 3) counts of yes / no / missing.
Shared by the open-model experiment (scripts/47), the Claude Haiku 5.5 test and the in-browser Live page (same wording).
"""
import re

import numpy as np

BATCH_PREAMBLE = ("\n\n---\n[Do not reply to the message above yet. This is a check-in before you reply: answer each "
                  "question below about the user's LATEST message only (the one just above this line), ignoring "
                  "earlier turns of the conversation. Your entire output must be exactly one line per question, in the "
                  "form Q<number>: Yes or Q<number>: No, with no other text.]\n")
BATCH_SUFFIX = "\n[Output only the answer lines, starting with Q1:]"

_LINE = re.compile(r"Q\s*(\d+)\s*[:.)\-]\s*\**\s*(yes|no)\b", re.I)


def batch_block(battery) -> str:
    """The question list plus the closing reminder; appended after BATCH_PREAMBLE."""
    return "\n".join(f"Q{j + 1}: {q}" for j, (_, q) in enumerate(battery)) + BATCH_SUFFIX


def parse_batch(text: str, k: int) -> np.ndarray:
    c = np.zeros((k, 3), np.int16)
    seen = set()
    for m in _LINE.finditer(text or ""):
        j = int(m.group(1)) - 1
        if 0 <= j < k and j not in seen:                         # first answer per question counts
            seen.add(j)
            c[j, 0 if m.group(2).lower() == "yes" else 1] += 1
    for j in range(k):
        if j not in seen:
            c[j, 2] += 1
    return c


# Development fix after L3c (scripts/54): the operator declares the check-in up front (system prompt, or a leading
# user turn where no system prompt is available), so the appended questionnaire is not an unannounced instruction.
CHECKIN_SYSTEM = ("This conversation is observed by a transparency tool run by the operator. Before each of your replies, "
                  "the tool may append a check-in to the user's latest message: a numbered list of yes/no questions about "
                  "that message, ending with an instruction to output only the answer lines. When a check-in is present, "
                  "do not reply to the message; answer every question honestly, one line each, in the form "
                  "Q<number>: Yes or Q<number>: No. Your actual reply is requested separately.")


# Development variant (scripts/54): the conversation goes in as a quoted transcript inside ONE user message, and the
# check-in asks about its last user message. Nothing is injected into a live conversation, so the model has no reason
# to treat the questionnaire as an injection. (A different elicitation: the model reads its conversation from outside.)
TRANSCRIPT_HEAD = ("Below is a conversation between a user and you, the AI assistant. It ends with the user's latest "
                   "message, which you have not answered yet.\n\n<conversation>\n")
TRANSCRIPT_TAIL = ("</conversation>\n\nThis is a check-in before you answer that latest message: answer each question below "
                   "about the user's LATEST message only, ignoring earlier turns. Your entire output must be exactly one "
                   "line per question, in the form Q<number>: Yes or Q<number>: No, with no other text.\n")


def transcript_prompt(turns, block):
    """turns: [{'role','content'}...] ending with the user's latest message."""
    body = "".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n\n" for m in turns)
    return TRANSCRIPT_HEAD + body + TRANSCRIPT_TAIL + block
