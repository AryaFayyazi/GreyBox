#!/usr/bin/env python
"""One-call grey box, TRANSCRIPT framing (scripts/54 showed Claude refuses an in-conversation questionnaire in multi-turn
chats but answers 100% when the conversation is quoted as a transcript). Same as scripts/47 otherwise.

Original description (scripts/47): one-call grey box: all battery questions in ONE side-branch call, sampled n times.

The per-question fork (scripts/36) costs one call per question (88 logprob calls, or 88 x n sampled calls on a
text-only API). Here the 88 v2m questions go to the model at once, appended to the user's latest message (a side
branch: the real reply never sees it), and the model answers every one on its own line. n = 4 samples at temperature
1 give q_j = (yes_j + 0.5) / (yes_j + no_j + 1) per item (the text-tier estimator of scripts/42).
Conversations and replies are rebuilt exactly as in scripts/36 (greedy, max 200 tokens, turn by turn).
Output: /data/models/graybox_ws/outputs/livebench_batch/<model>/psi.npz (p_yes = q, p_no = 1 - q, counts) plus
symlinks to the benchmark's items, moderation and this model's refusal labels, so scripts/38 runs unchanged on it.

python scripts/55_transcript_battery.py model=qwen3_8b
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
from graybox.battery_v2m import BATTERY_V2M  # noqa: E402
from graybox.batch import batch_block, parse_batch, transcript_prompt  # noqa: E402

run12 = importlib.import_module("12_l2_run")
LB = Path("/data/models/graybox_ws/outputs/livebench")
OUT = Path("/data/models/graybox_ws/outputs/livebench_transcript")
D = OmegaConf.create({"model": "qwen3_8b", "max_len": 8192, "n": 4, "limit": 0, "out": ""})


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    m = run12.MODELS[cfg.model]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    path = run12.path_of(m["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    items = pd.read_parquet(LB / "items.parquet")
    if cfg.limit:                                                  # smoke test on the first items only
        items = items.head(int(cfg.limit))
    out_root = Path(cfg.out) if cfg.out else OUT
    llm = LLM(model=path, enable_prefix_caching=True, max_model_len=cfg.max_len, gpu_memory_utilization=0.85,
              trust_remote_code=True, limit_mm_per_prompt={"image": 0} if "gemma3" in cfg.model else None)
    gen = SamplingParams(max_tokens=200, temperature=0.0)
    ask = SamplingParams(max_tokens=12 * len(BATTERY_V2M), temperature=1.0, n=cfg.n, seed=0)

    def render(msgs, prefill=""):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **m["kw"]) + prefill

    def enc(ps):
        out = []
        for i in range(0, len(ps), 4096):
            out += [{"prompt_token_ids": ids} for ids in tok(ps[i:i + 4096], add_special_tokens=False)["input_ids"]]
        return out

    t0 = time.time()
    convs = [[] for _ in range(len(items))]
    for k in range(int(items.turns.map(len).max())):             # the model's own replies to earlier turns (as 36)
        idx = [i for i, t in enumerate(items.turns) if len(t) > k + 1]
        for i in idx:
            convs[i].append({"role": "user", "content": items.turns.iloc[i][k]})
        if idx:
            outs = llm.generate(enc([render(convs[i], m["prefill"] or "") for i in idx]), gen)
            for i, o in zip(idx, outs):
                convs[i].append({"role": "assistant", "content": o.outputs[0].text.strip()})
    block = batch_block(BATTERY_V2M)
    prompts = [render([{"role": "user", "content": transcript_prompt(c + [{"role": "user", "content": t[-1]}], block)}], m["prefill"] or "")
               for c, t in zip(convs, items.turns)]
    outs = llm.generate(enc(prompts), ask)
    K = len(BATTERY_V2M)
    C = np.zeros((len(items), K, 3), np.int16)                    # yes, no, missing
    for i, o in enumerate(outs):
        for s in o.outputs:
            C[i] += parse_batch(s.text, K)
    q = (C[..., 0] + 0.5) / (C[..., 0] + C[..., 1] + 1.0)
    d = out_root / cfg.model
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "psi.npz", p_yes=q.astype(np.float32), p_no=(1 - q).astype(np.float32), counts=C,
                        items=np.array([a for a, _ in BATTERY_V2M]))
    for name, src in (("refusals.parquet", LB / cfg.model / "refusals.parquet"),):
        if not (d / name).exists():
            os.symlink(src, d / name)
    for name in ("items.parquet", "moderation.parquet"):
        if not (out_root / name).exists():
            os.symlink(LB / name, out_root / name)
    answered = C[..., :2].sum() / C.sum()
    print(f"{cfg.model}: done {time.time()-t0:.0f}s; answered {answered:.3%} of item-samples; "
          f"mean answers per sample {C[..., :2].sum() / (len(items) * cfg.n):.1f}/{K}", flush=True)
    print("example output:", outs[0].outputs[0].text[:600].replace("\n", " | "), flush=True)


if __name__ == "__main__":
    main()
