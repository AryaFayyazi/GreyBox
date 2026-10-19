#!/usr/bin/env python
"""EXPLORATORY (development contexts only; after the lock of G7): why do yes/no answers carry no model-specific geometry
while one next-token probability does? Two explanations:
  (a) the yes/no PROJECTION discards it: the answer position's full distribution would still carry it;
  (b) the question's COMPUTATION discards it: even the answer position's full distribution carries only shared geometry.
For 8 battery questions (the first 8 of Claude's 24) on G7's 600 development contexts, each model's top-20 next-token
log-probabilities at the ANSWER position (after the question and the answer prefill) are read with vLLM. Geometries:
  ans_yesno    Fisher-Rao of P(yes | yes or no) over the 8 questions (as G1)
  ans_top20    Bhattacharyya geometry of the top-20 law at the answer position, summed over the 8 questions
Compared with G7's dev channels (top-20 at the content position). Own-model excess as in G7, against the last layer.
  stage=gen model=M   (GPU, vLLM)      stage=analyze (CPU)
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
OUT = Path("/data/models/graybox_ws/outputs/g7/dev_answerpos")
D = OmegaConf.create({"stage": "analyze", "model": "qwen3_8b"})
ITEMS = ["T02", "T03", "T04", "T05", "T06", "T09", "T10", "T11"]


def stage_gen(cfg):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from graybox.battery_v2 import BATTERY_V2, PREAMBLE
    L = _mod("l2run", "12_l2_run.py")
    m = L.MODELS[cfg.model]
    path = L.path_of(m["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    C = g7.contexts("dev")
    qs = [q for k, q in BATTERY_V2 if k in ITEMS]
    prompts = [L.render(tok, t, m, q) for t in C.text for q in qs]
    llm = LLM(model=path, enable_prefix_caching=True, max_model_len=4096, gpu_memory_utilization=0.85, max_logprobs=20, trust_remote_code=True,
              limit_mm_per_prompt={"image": 0} if "gemma3" in cfg.model else None)
    enc = [{"prompt_token_ids": ids} for ids in tok(prompts, add_special_tokens=False)["input_ids"]]
    o = llm.generate(enc, SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, seed=0))
    n, k = len(C), len(qs)
    ids, lp, py, pn = np.zeros((n, k, 20), np.int64), np.full((n, k, 20), -50.0, np.float32), np.zeros((n, k)), np.zeros((n, k))
    for i, oo in enumerate(o):
        d = sorted(oo.outputs[0].logprobs[0].items(), key=lambda kv: kv[1].rank)[:20]
        for j, (tid, l) in enumerate(d):
            ids[i // k, i % k, j], lp[i // k, i % k, j] = tid, l.logprob
        py[i // k, i % k], pn[i // k, i % k] = L.yes_no(oo.outputs[0].logprobs[0])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{cfg.model}.npz", ids=ids, lp=lp, p_yes=py, p_no=pn)
    print(cfg.model, "done", ids.shape, flush=True)


def stage_analyze(cfg):
    M = g.MODELS
    C = g7.contexts("dev")
    ch = {"ans_yesno": {}, "ans_top20": {}, "content_top20": {}}
    Hl = {}
    for m in M:
        z = np.load(OUT / f"{m}.npz")
        P = np.clip((z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6), 0, 1)
        ch["ans_yesno"][m] = np.sqrt(g.sq_dist(g.theta(P)))
        D2 = sum(g7.topk_dist(z["ids"][:, j], z["lp"][:, j], 20) ** 2 for j in range(z["ids"].shape[1]))
        ch["ans_top20"][m] = np.sqrt(D2)                              # product law over questions: squared distances add (intrinsic)
        cz = np.load(g7.OUT / "dev" / m / "channels.npz")
        ch["content_top20"][m] = g7.topk_dist(cz["top_ids"], cz["top_lp"], 20)
        Hl[m] = g.cos_dist(np.load(g7.OUT / "dev" / m / "hidden.npz")["h"][:, -1].astype(np.float64))
    res = {}
    rng = np.random.default_rng(1410)
    n = len(C)
    boots = [np.unique(rng.integers(0, n, n)) for _ in range(100)]
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731

    def excess(Dc, s):
        R = np.array([[g.rho(sub(Dc[a], s), sub(Hl[b], s)) for b in M] for a in M])
        r = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()
        return float(np.diag(r).mean())
    for c in ch:
        b = [excess(ch[c], s) for s in boots]
        res[c] = [round(excess(ch[c], np.arange(n)), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
        print(c, res[c], flush=True)
    (g.RES / "answer_position_exploratory.json").write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"gen": stage_gen, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
