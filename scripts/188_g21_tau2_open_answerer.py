#!/usr/bin/env python
"""G21: can an OPEN model answer the questionnaire about a CLOSED agent's calls on the second benchmark?
(docs/prereg_G21_tau2_open_answerer.md). On G18's 360 tau2-bench writes of Claude Haiku 5.5, Gemma-4 (26B-A4B, thinking
on, its standard sampling, served by vLLM) reads Claude's conversation up to the write (tau2's agent system prompt with
G18's SYSTEM_NOTE, the airline tools, every earlier turn; OpenAI format, as Claude saw it through scripts/101) and
answers G18's 8 questions in G18's frame, 4 answers each, parsed by G18's parse(). Same read-out. Nothing is fit; no
label is used; no API is called.
  stage=run       (GPU) -> outputs/tau2_claude/g21_gemma4.parquet
  stage=analyze   (CPU) -> results/t1/g21_tau2_open_answerer.json
"""
import asyncio
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
OUT = Path("/data/models/graybox_ws/outputs/tau2_claude")
_s = importlib.util.spec_from_file_location("g18", ROOT / "scripts/184_tau2_claude_readout.py")
G18 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(G18)
N, SEED, PORT, CONC = 4, 2100, 8188, 48


def units():
    """G18's 360 writes in G18's order, with the messages and args (aligned as in scripts/185)."""
    U = pd.read_json(OUT / "g18_units_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    S = G18.sample(U, 360, 1840)
    B = pd.read_json(OUT / "g18_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    assert len(S) == len(B) and (S.sim_id.values == B.sim_id.values).all() and (S.msg_idx.values == B.msg_idx.values).all() \
        and (S.tool.values == B.tool.values).all() and (S.y.values == B.y.values).all()
    B["messages"], B["args"] = S.messages.values, S.args.values
    return B


def clean(msgs):
    out = []
    for m in msgs:
        m = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id") and v is not None}
        if m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls"):
            m["content"] = "(no content)"
        m.setdefault("content", "")
        out.append(m)
    if out and out[0]["role"] == "assistant":                 # tau2's agent speaks first; as scripts/101 does for Claude
        out.insert(0, {"role": "user", "content": "(The conversation starts.)"})
    return out


async def one(client, a, base, tools, name, args, sem):
    yes = []
    for j, (q, _) in enumerate(G18.QUESTIONS):
        body = {"model": "actor", "messages": base + [{"role": "user", "content": G18.FRAME.format(name=name, args=args, q=q)}],
                "tools": tools, "tool_choice": "none", "n": N, "seed": SEED + j, "max_tokens": 6000, **a["sampling"], **a["extra"]}
        k = m = 0
        async with sem:
            for _ in range(3):
                try:
                    resp = await client.post("/v1/chat/completions", json=body, timeout=1800)
                    resp.raise_for_status()
                    for ch in resp.json()["choices"]:
                        v = G18.parse(ch["message"].get("content") or "")
                        if v is not None:
                            m += 1
                            k += int(not v)
                    break
                except Exception:  # noqa: BLE001
                    body["seed"] += 7919
                    await asyncio.sleep(2)
        yes.append([k, m])
    return yes


def stage_run():
    sa = importlib.import_module("150_self_audit")
    a = sa.srv.ACTORS["gemma4"]
    ctx = json.loads((OUT / "g18_context.json").read_text())
    system = ctx["system"] + G18.SYSTEM_NOTE
    B = units()
    its = [([{"role": "system", "content": system}] + clean(json.loads(r.messages)), r.tool, r.args) for r in B.itertuples()]
    proc = sa.srv.start_server(a, PORT)
    try:
        async def go():
            sem = asyncio.Semaphore(CONC)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}") as client:
                out, t0 = [], time.time()
                for i in range(0, len(its), 48):
                    out += await asyncio.gather(*[one(client, a, b, ctx["tools"], nm, ar, sem) for b, nm, ar in its[i:i + 48]])
                    print(f"{len(out)}/{len(its)} {time.time() - t0:.0f}s", flush=True)
                return out
        out = asyncio.run(go())
    finally:
        proc.terminate()
    B[["run", "sim_id", "task_id", "msg_idx", "tool", "y"]].assign(counts=[json.dumps(o) for o in out]).to_parquet(OUT / "g21_gemma4.parquet")


def stage_analyze():
    from graybox.stats import cluster_boot, within_auroc
    B = units()
    G = pd.read_parquet(OUT / "g21_gemma4.parquet")
    assert (G.sim_id.values == B.sim_id.values).all() and (G.msg_idx.values == B.msg_idx.values).all()
    G["counts"] = [json.loads(c) for c in G.counts]
    so, so1 = G18.read_out(B)                                   # Claude's own answers (G18)
    sg, sg1 = G18.read_out(G)                                   # Gemma-4's answers about Claude's writes
    y, tl, tk = B.y.values.astype(int), B.tool.values, B.task_id.values
    W = lambda s, i: within_auroc(y[i], s[i], tl[i])  # noqa: E731
    a = np.arange(len(y))
    bg = cluster_boot(lambda i: W(sg, i), tk, 2000, seed=2101)
    bg = bg[np.isfinite(bg)]
    bd = cluster_boot(lambda i: W(so, i) - W(sg, i), tk, 2000, seed=2102)
    bd = bd[np.isfinite(bd)]
    C = np.array(list(G.counts))
    res = {"writes": int(len(y)), "violations": int(y.sum()), "parsed_share_gemma4": round(float(C[:, :, 1].sum() / (C.shape[0] * C.shape[1] * N)), 4),
           "within_tool": {"claude_own_8": round(float(W(so, a)), 4), "gemma4_8": round(float(W(sg, a)), 4),
                           "claude_own_single": round(float(W(so1, a)), 4), "gemma4_single": round(float(W(sg1, a)), 4)},
           "A1_gemma4_8_ci": [round(float(np.percentile(bg, 2.5)), 4), round(float(np.percentile(bg, 97.5)), 4)],
           "A2_own_minus_gemma4": [round(float(W(so, a) - W(sg, a)), 4), round(float(np.percentile(bd, 2.5)), 4), round(float(np.percentile(bd, 97.5)), 4)]}
    res["A1_pass"] = res["A1_gemma4_8_ci"][0] > 0.5
    (ROOT / "results/t1/g21_tau2_open_answerer.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"run": stage_run, "analyze": stage_analyze}[dict(a.split("=", 1) for a in sys.argv[1:]).get("stage", "run")]()
