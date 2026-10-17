#!/usr/bin/env python
"""Live v2 benchmark, one open model (vLLM; the model is used only through prompts and next-token logprobs).

For every benchmark item (scripts/35):
  1. multi-turn: the model's own greedy replies (max 200 tokens) to each earlier user turn, generated turn by turn,
     so the context is a realistic conversation with this model
  2. Ψ: the 88 battery-v2m questions about the LATEST user message (question appended to it, answer prefill),
     top-20 logprobs -> P(yes), P(no)
  3. the model's real greedy reply to the full conversation (max 200 tokens), for refusal labels (judged later)
Output: /data/models/graybox_ws/outputs/livebench/<model>/{psi.npz, replies.parquet}

python scripts/36_live_bench_run.py model=qwen3_8b
"""
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.battery_v2m import BATTERY_V2M, PREAMBLE, battery_sha  # noqa: E402

run12 = importlib.import_module("12_l2_run")
LB = Path("/data/models/graybox_ws/outputs/livebench")
D = OmegaConf.create({"model": "qwen3_8b", "max_len": 8192})


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    m = run12.MODELS[cfg.model]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    path = run12.path_of(m["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    items = pd.read_parquet(LB / "items.parquet")
    llm = LLM(model=path, enable_prefix_caching=True, max_model_len=cfg.max_len, gpu_memory_utilization=0.85,
              max_logprobs=20, trust_remote_code=True, limit_mm_per_prompt={"image": 0} if "gemma3" in cfg.model else None)
    gen = SamplingParams(max_tokens=200, temperature=0.0)
    one = SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, seed=0)

    def render(msgs, prefill=""):
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **m["kw"])
        return s + prefill

    def enc(ps):
        out = []
        for i in range(0, len(ps), 4096):
            out += [{"prompt_token_ids": ids} for ids in tok(ps[i:i + 4096], add_special_tokens=False)["input_ids"]]
        return out

    t0 = time.time()
    convs = [[] for _ in range(len(items))]                      # message lists, filled turn by turn
    max_turns = int(items.turns.map(len).max())
    for k in range(max_turns):                                   # generate replies to earlier turns, round by round
        idx = [i for i, t in enumerate(items.turns) if len(t) > k + 1]
        for i in idx:
            convs[i].append({"role": "user", "content": items.turns.iloc[i][k]})
        if idx:
            outs = llm.generate(enc([render(convs[i], m["prefill"] or "") for i in idx]), gen)
            for i, o in zip(idx, outs):
                convs[i].append({"role": "assistant", "content": o.outputs[0].text.strip()})
    for i, t in enumerate(items.turns):
        convs[i].append({"role": "user", "content": t[-1]})
    print(f"{cfg.model}: conversations built {time.time()-t0:.0f}s; battery v2m {battery_sha()[:12]}", flush=True)
    finals = llm.generate(enc([render(c, m["prefill"] or "") for c in convs]), gen)
    replies = [o.outputs[0].text for o in finals]
    qp = []
    for c in convs:
        for _, q in BATTERY_V2M:
            cc = c[:-1] + [{"role": "user", "content": c[-1]["content"] + PREAMBLE + q}]
            qp.append(render(cc, m["prefill"]))
    outs = llm.generate(enc(qp), one)
    n, k = len(items), len(BATTERY_V2M)
    PY, PN = np.zeros((n, k), np.float32), np.zeros((n, k), np.float32)
    for j, o in enumerate(outs):
        PY[j // k, j % k], PN[j // k, j % k] = run12.yes_no(o.outputs[0].logprobs[0])
    out = LB / cfg.model
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "psi.npz", p_yes=PY, p_no=PN, items=np.array([a for a, _ in BATTERY_V2M]), battery_sha=battery_sha())
    pd.DataFrame({"id": items.id, "reply": replies,
                  "history": [[x["content"][:300] for x in c[:-1]] for c in convs]}).to_parquet(out / "replies.parquet")
    mass = PY + PN
    print(f"done {time.time()-t0:.0f}s; yes+no mass median {np.median(mass):.3f}; <0.5 in {(mass < 0.5).mean():.2%}", flush=True)


if __name__ == "__main__":
    main()
