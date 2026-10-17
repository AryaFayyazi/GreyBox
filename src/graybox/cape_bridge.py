"""Bridge to CAPE (/home/arya/new_safety/cape): episodes, scenario specs, chat rendering.

The rendering follows CAPE's teacher-forced replay (scripts/56_xm_replay_extract.py) so that the context a grey-box
query sees is exactly the context the agent saw. Differences: the turn-start unit includes the tool result of every
earlier action, and question prompts append one user turn plus a per-model answer prefill (no thinking phase).
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

CAPE = Path("/home/arya/new_safety/cape")
CAPE_OUT = Path("/data/models/cape_ws/outputs")
if str(CAPE / "src") not in sys.path:
    sys.path.insert(0, str(CAPE / "src"))

from cape.envs.base import TOOLS  # noqa: E402

ACT = {"gemma4": {"model": "google/gemma-4-26B-A4B-it", "kwargs": {"enable_thinking": True}, "suffix": "",
                  "prefill": "<|channel>thought\n<channel|>"},
       "gptoss20b": {"model": "openai/gpt-oss-20b", "kwargs": {"reasoning_effort": "medium"}, "suffix": "",
                     "prefill": "<|channel|>final<|message|>"},
       "museglimmer": {"model": "meta-models/Muse-Glimmer-30B", "kwargs": {},
                       "suffix": "\n\nReasoning strength: medium", "prefill": "<|channel|>final<|message|>"},
       "qwen38": {"model": "Qwen/Qwen3.8-27B", "kwargs": {"enable_thinking": True}, "suffix": "",
                  "prefill": "\n</think>\n\n"},
       # reader-only models (Tier-3 control: an open model placed in the agent's shoes on the same transcript)
       "qwen3_8b": {"model": "Qwen/Qwen3-8B", "kwargs": {"enable_thinking": True}, "suffix": "",
                    "prefill": "<think>\n\n</think>\n\n"}}
READS = {"list_files", "read_file", "open_mock_url"}
TAG = re.compile(r"<\|[^|<>]{0,40}\|>")


@lru_cache(maxsize=None)
def specs(bench: str) -> dict:
    out = {}
    for line in open(CAPE / f"outputs/capebench/{bench}/scenarios.jsonl"):
        s = json.loads(line)
        out[s["instance_id"]] = s
    return out


def assistant_msg(e: dict) -> dict:
    try:
        a = json.loads(e["args_json"]) if e["args_json"] else {}
    except Exception:
        a = {}
    r = TAG.sub("", e["reasoning_text"] or "")
    return {"role": "assistant", "content": TAG.sub("", e["content_text"] or ""),
            "tool_calls": [{"id": f"call_{e['event_id']}", "type": "function",
                            "function": {"name": e["tool"], "arguments": a}}],
            "thinking": r, "reasoning": r, "reasoning_content": r}


def turn_start_messages(spec: dict, events: list, t: int, suffix: str) -> list:
    """Context at the start of the agent turn that produced event t: all earlier tool calls WITH their results."""
    msgs = [{"role": "system", "content": spec["system_prompt"] + suffix}, {"role": "user", "content": spec["task_prompt"]}]
    for e in events:
        if int(e["event_id"]) >= t:
            break
        if not e["is_tool_call"] or e["tool"] is None:
            continue
        msgs.append(assistant_msg(e))
        msgs.append({"role": "tool", "tool_call_id": f"call_{e['event_id']}", "content": e["tool_result"] or ""})
    return msgs


def _merge_content_into_thinking(msgs):
    out = []
    for x in msgs:
        if x.get("role") == "assistant" and x.get("tool_calls") and x.get("content") and x.get("thinking"):
            t = x["thinking"] + "\n" + x["content"]
            x = dict(x, content="", thinking=t, reasoning=t, reasoning_content=t)
        out.append(x)
    return out


def _render(tok, msgs, kw, gen):
    err = None
    for as_dict in (True, False):
        try:
            m = msgs if as_dict else [dict(x, tool_calls=[dict(c, function=dict(c["function"], arguments=json.dumps(
                c["function"]["arguments"]) if not isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]))
                for c in x["tool_calls"]]) if "tool_calls" in x else x for x in msgs]
            return tok.apply_chat_template(m, tools=TOOLS, tokenize=False, add_generation_prompt=gen, **kw)
        except Exception as ex:  # noqa: PERF203
            err = ex
    raise err


def render(tok, msgs, kw, gen=True):
    try:
        return _render(tok, msgs, kw, gen)
    except Exception:  # gpt-oss: "cannot pass both content and thinking with tool calls"
        return _render(tok, _merge_content_into_thinking(msgs), kw, gen)


def question_prompt(tok, msgs, actor: str, question: str, preamble: str) -> str:
    a = ACT[actor]
    q = msgs + [{"role": "user", "content": preamble + question}]
    return render(tok, q, a["kwargs"], gen=True) + a["prefill"]


def load_units(actor: str, sets: list[str] | None = None) -> pd.DataFrame:
    """CAPE v57 decision units (VIOL / XOTHER / XPLUS) for one actor; one row per proposed non-read action."""
    d = pd.read_parquet(CAPE_OUT / "v57/scores.parquet")
    d = d[d.model == actor]
    if sets:
        d = d[d.set.isin(sets)]
    return d.reset_index(drop=True)


GB_ROLLOUTS = Path("/data/models/graybox_ws/cape_root/outputs/rollouts")   # fresh episodes generated for this project


@lru_cache(maxsize=64)
def rollouts(set_name: str) -> pd.DataFrame:
    p = CAPE_OUT / f"rollouts/{set_name}.parquet"
    if not p.exists():
        p = GB_ROLLOUTS / f"{set_name}.parquet"
    return pd.read_parquet(p).set_index("episode_id", drop=False)
