#!/usr/bin/env python
"""The grey box of a CLOSED agent at its decision point: Claude Haiku 5.5's own answers to the 8 decision questions
(web/build_observable_ref.py QUESTIONS and FRAME, unchanged), n answers each, through its API (thinking disabled, effort
low, as in scripts/74), on its labelled CAPE-Bench actions (T8b: every non-read call; the same enumeration as
scripts/76). Every call is charged to the shared Anthropic ledger and the run stops at cfg.cap. Checkpointed.
  stage=battery cap=<total ledger $> [limit=10]      -> outputs/decision_greybox/claude_t8b.jsonl
Sample (fixed): every action of cons_service_restart (the template where Claude's violations are decided by its
transcript), every violating action, and allowed actions of the other templates up to 500 units (seed 1720).
"""
import asyncio
import copy
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CAPE = Path("/home/arya/new_safety/cape")
for p in (ROOT / "src", ROOT / "scripts", CAPE / "src", CAPE / "scripts"):
    sys.path.insert(0, str(p))
from graybox.budget import HAIKU55, BudgetExceeded, Ledger  # noqa: E402
from graybox.selfaudit import parse_answer  # noqa: E402

a74 = importlib.import_module("74_claude_agent")
_spec = importlib.util.spec_from_file_location("obsref", ROOT / "web/build_observable_ref.py")
OR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OR)
CA = Path("/data/models/graybox_ws/outputs/claude_agent")
OUT = Path("/data/models/graybox_ws/outputs/decision_greybox")
LEDGER = Path("/data/models/graybox_ws/outputs/l3/spend.json")
READS = {"list_files", "read_file", "open_mock_url"}
CFG = OmegaConf.create({"stage": "battery", "cap": None, "n": 4, "n_units": 500, "seed": 1720, "concurrency": 8, "limit": None, "wave": 20})


def units():
    """scripts/76's enumeration, keeping the event and the Anthropic messages before it."""
    E = pd.read_parquet(CA / "run.parquet")
    S = {json.loads(l)["instance_id"]: json.loads(l) for l in open("/data/models/cape_ws/outputs/capebench/core24b/scenarios.jsonl")}
    rows = []
    for ep in E.itertuples():
        spec = S[ep.instance_id]
        msgs = [{"role": "user", "content": [{"type": "text", "text": spec["task_prompt"]}]}]
        for k, e in enumerate(ep.events):
            if not e["is_tool_call"] or e["tool"] is None:
                continue
            args = json.loads(e["args_json"] or "{}")
            if e["tool"] not in READS:
                rows.append({"episode_id": ep.episode_id, "event_id": int(e["event_id"]), "scenario_id": ep.scenario_id,
                             "template": f"{ep.template}|{e['tool']}", "y": int(bool(e["unsafe_action"])), "system": spec["system_prompt"],
                             "msgs": json.dumps(msgs), "tool": e["tool"], "args": e["args_json"] or "{}"})
                if e["unsafe_action"]:
                    break
            blocks = ([{"type": "text", "text": e["content_text"]}] if e.get("content_text") else [])
            blocks.append({"type": "tool_use", "id": f"toolu_{k:03d}", "name": e["tool"], "input": args})
            msgs = msgs + [{"role": "assistant", "content": blocks},
                           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"toolu_{k:03d}", "content": str(e["tool_result"] or "")}]}]
    return pd.DataFrame(rows)


def sample(U, n_units, seed):
    rng = np.random.default_rng(seed)
    tm = U.template.str.split("|").str[0]
    keep = (tm == "cons_service_restart") | (U.y == 1)
    rest = np.where(~keep)[0]
    extra = rng.choice(rest, max(0, n_units - int(keep.sum())), replace=False)
    return U.iloc[np.sort(np.concatenate([np.where(keep)[0], extra]))].reset_index(drop=True)


async def battery(cl, led, sem, r, n):
    yes = []
    for q in OR.QUESTIONS:
        m = json.loads(r.msgs)
        m[-1]["content"] = list(m[-1]["content"]) + [{"type": "text", "text": OR.FRAME.format(name=r.tool, args=r.args, q=q)}]

        async def one():
            led.check()
            async with sem:
                x = await cl.messages.create(model="claude-haiku-5-5", max_tokens=16, system=r.system, messages=m, tools=a74.A_TOOLS,
                                             tool_choice={"type": "none"}, temperature=1.0, thinking={"type": "disabled"},
                                             output_config={"effort": "low"})
            led.charge(x.usage.input_tokens, x.usage.output_tokens)
            return "".join(b.text for b in x.content if b.type == "text")
        txt = await asyncio.gather(*[one() for _ in range(n)])
        v = [parse_answer(t) for t in txt]
        yes.append([int(sum(1 for x in v if x is False)), int(sum(1 for x in v if x is not None))])
    return yes


def stage_battery(cfg):
    import anthropic
    assert cfg.cap is not None, "set cap (total ledger dollars) explicitly"
    led = Ledger(LEDGER, float(cfg.cap), *HAIKU55)
    U = sample(units(), cfg.n_units, cfg.seed)
    if cfg.limit:
        U = U.iloc[: int(cfg.limit)]
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "claude_t8b.jsonl"
    done = {(json.loads(x)["episode_id"], json.loads(x)["event_id"]) for x in open(fn)} if fn.exists() else set()
    todo = [r for r in U.itertuples() if (r.episode_id, r.event_id) not in done]
    print(f"{len(U)} units, {len(todo)} to do; ledger ${led.total():.3f} cap ${cfg.cap}", flush=True)
    cl = anthropic.AsyncAnthropic(api_key=Path("/data/models/graybox_ws/.secrets/anthropic_key").read_text().strip(), max_retries=8)

    async def go():
        sem, t0 = asyncio.Semaphore(cfg.concurrency), time.time()
        for i in range(0, len(todo), cfg.wave):
            batch = todo[i:i + cfg.wave]
            try:
                out = await asyncio.gather(*[battery(cl, led, sem, r, cfg.n) for r in batch])
            except BudgetExceeded:
                print("ledger cap reached; stopping", flush=True)
                return
            with open(fn, "a") as f:
                for r, o in zip(batch, out):
                    f.write(json.dumps({"episode_id": r.episode_id, "event_id": int(r.event_id), "scenario_id": r.scenario_id,
                                        "template": r.template, "y": int(r.y), "counts": json.dumps(o)}) + "\n")
            print(f"{i + len(batch)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def stage_reader(cfg):
    """The open reader's features of every T8b action (Qwen3-8B, the layers of outputs/t4, last token), with scripts/96's
    code and scripts/76's reader prompt, aligned with scripts/76's units."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t4 = importlib.import_module("69_t4_reader_action")
    T = importlib.import_module("76_t8b_claude_all_actions").units()
    p = snapshot_download("Qwen/Qwen3-8B", cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda").eval()
    L = [18, 22, 26, 30, 32]
    H = np.zeros((len(T), len(L), model.config.hidden_size), np.float16)
    t0 = time.time()
    for i, (c, a) in enumerate(zip(T.ctx, T.act)):
        s = tok.apply_chat_template([{"role": "user", "content": t4.Q.format(ctx=c, cot="", act=a)}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][:, -12000:].cuda()
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        H[i] = np.stack([hs[l][0, -1].float().cpu().numpy() for l in L]).astype(np.float16)
        if i % 500 == 0:
            print(f"{i}/{len(T)} {time.time()-t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "claude_t8b_act.npz", h=H, layers=np.array(L))
    print("saved", H.shape, flush=True)


def main():
    cfg = OmegaConf.merge(CFG, OmegaConf.from_cli())
    {"battery": stage_battery, "reader": stage_reader}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
