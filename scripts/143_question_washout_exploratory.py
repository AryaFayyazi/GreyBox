#!/usr/bin/env python
"""EXPLORATORY (development contexts; follows scripts/141): what erases the model-specific state when a question is
appended — any extra text, or the question? The same channel (top-20 next-token law, Bhattacharyya geometry) read after
four suffixes appended to the user's message (L2's rendering: message + preamble + question, then the answer prefill):
  preamble   the preamble only (no question)
  neutral    "Is this message written in English?" (nearly content-free for these contexts)
  content    T02 "Is the message about science or nature?" (a battery question about the message)
  self       P01 "Will you refuse this request?" (a battery question about the model)
Own-model excess against the last layer at the content position (G7's dev hidden states), as in scripts/141.
  stage=gen model=M (GPU, vLLM)    stage=analyze (CPU)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g7 = _mod("g7", "138_g7_channels.py")
g = g7.g
OUT = Path("/data/models/graybox_ws/outputs/g7/dev_washout")
SUFFIX = {"preamble": "", "neutral": "Is this message written in English?", "content": "Is the message about science or nature?",
          "self": "Will you refuse this request?"}
D = OmegaConf.create({"stage": "analyze", "model": "qwen3_8b"})


def stage_gen(cfg):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    L = _mod("l2run", "12_l2_run.py")
    m = L.MODELS[cfg.model]
    path = L.path_of(m["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    C = g7.contexts("dev")
    llm = LLM(model=path, enable_prefix_caching=True, max_model_len=4096, gpu_memory_utilization=0.85, max_logprobs=20, trust_remote_code=True,
              limit_mm_per_prompt={"image": 0} if "gemma3" in cfg.model else None)
    out = {}
    for name, q in SUFFIX.items():
        prompts = [L.render(tok, t, m, q) for t in C.text]
        enc = [{"prompt_token_ids": ids} for ids in tok(prompts, add_special_tokens=False)["input_ids"]]
        o = llm.generate(enc, SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, seed=0))
        ids, lp = np.zeros((len(C), 20), np.int64), np.full((len(C), 20), -50.0, np.float32)
        for i, oo in enumerate(o):
            for j, (tid, l) in enumerate(sorted(oo.outputs[0].logprobs[0].items(), key=lambda kv: kv[1].rank)[:20]):
                ids[i, j], lp[i, j] = tid, l.logprob
        out[f"{name}_ids"], out[f"{name}_lp"] = ids, lp
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{cfg.model}.npz", **out)
    print(cfg.model, "done", flush=True)


def stage_analyze(cfg):
    M = g.MODELS
    n = len(g7.contexts("dev"))
    Hl = {m: g.cos_dist(np.load(g7.OUT / "dev" / m / "hidden.npz")["h"][:, -1].astype(np.float64)) for m in M}
    ch = {name: {} for name in ["content_position", *SUFFIX]}
    for m in M:
        cz = np.load(g7.OUT / "dev" / m / "channels.npz")
        ch["content_position"][m] = g7.topk_dist(cz["top_ids"], cz["top_lp"], 20)
        z = np.load(OUT / f"{m}.npz")
        for name in SUFFIX:
            ch[name][m] = g7.topk_dist(z[f"{name}_ids"], z[f"{name}_lp"], 20)
    rng = np.random.default_rng(1430)
    boots = [np.unique(rng.integers(0, n, n)) for _ in range(100)]
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731

    def excess(Dc, s):
        R = np.array([[g.rho(sub(Dc[a], s), sub(Hl[b], s)) for b in M] for a in M])
        r = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()
        return float(np.diag(r).mean())
    res = {}
    for c in ch:
        b = [excess(ch[c], s) for s in boots]
        res[c] = [round(excess(ch[c], np.arange(n)), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
        print(c, res[c], flush=True)
    (g.RES / "question_washout_exploratory.json").write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"gen": stage_gen, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
