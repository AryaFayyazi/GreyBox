#!/usr/bin/env python
"""EXPLORATORY (development contexts): does asking a question make different models' INTERNAL geometries agree more?
For G7's 600 development contexts, each model's hidden states at the last token (L2's 6 depths) after:
  content   the message only (the content position; G7's dev hidden states)
  preamble  the message + the battery preamble, no question
  q_content the message + preamble + T02 "Is the message about science or nature?" + answer prefill
  q_self    the message + preamble + P01 "Will you refuse this request?" + answer prefill
Cross-model agreement: mean Spearman between the cosine geometries of every pair of the 8 models, at relative depths 50%,
75% and the last layer; bootstrap over contexts. Prediction (from scripts/141): the question conditions agree more.
  stage=gen model=M (GPU, HF)    stage=analyze (CPU)
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
OUT = Path("/data/models/graybox_ws/outputs/g7/dev_align")
COND = {"preamble": "", "q_content": "Is the message about science or nature?", "q_self": "Will you refuse this request?"}
D = OmegaConf.create({"stage": "analyze", "model": "qwen3_8b", "set": "dev"})


def conditions(which):
    """dev: the three conditions above. g7 (confirmatory): the preamble and the first 8 of Claude's 24 battery questions."""
    if which == "dev":
        return COND
    from graybox.battery_v2 import BATTERY_V2
    keep = ["T02", "T03", "T04", "T05", "T06", "T09", "T10", "T11"]
    return {"preamble": "", **{f"q_{k}": q for k, q in BATTERY_V2 if k in keep}}


def outdir(which):
    return OUT if which == "dev" else OUT.parent / "g7_align"


def content_hidden(which, m):
    return np.load(g7.OUT / which / m / "hidden.npz")["h"]


def stage_gen(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    L = _mod("l2run", "12_l2_run.py")
    m = L.MODELS[cfg.model]
    path = L.path_of(m["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = [int(x) for x in np.load(g.L2 / cfg.model / "hidden.npz")["layers"]]
    C = g7.contexts(cfg.set)
    out = {}
    for name, q in conditions(cfg.set).items():
        H = []
        for t in C.text:
            ids = tok(L.render(tok, t, m, q), add_special_tokens=False, return_tensors="pt")["input_ids"][:, -4096:].cuda()
            with torch.no_grad():
                hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
            H.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in layers]))
        out[name] = np.stack(H).astype(np.float32)
    outdir(cfg.set).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir(cfg.set) / f"{cfg.model}.npz", layers=np.array(layers), **out)
    print(cfg.model, "done", flush=True)


def stage_analyze(cfg):
    M = g.MODELS
    n = len(g7.contexts(cfg.set))
    CND = conditions(cfg.set)
    geo = {c: {} for c in ["content", *CND]}
    for m in M:
        base = content_hidden(cfg.set, m)
        z = np.load(outdir(cfg.set) / f"{m}.npz")
        nl = int(np.load(g.L2 / m / "hidden.npz")["n_layers"])
        layers = [int(x) for x in z["layers"]]
        pick = {"d50": int(np.argmin([abs(l / nl - 0.5) for l in layers])), "d75": int(np.argmin([abs(l / nl - 0.75) for l in layers])), "last": len(layers) - 1}
        for c in geo:
            arr = base if c == "content" else z[c]
            geo[c][m] = {k: g.cos_dist(arr[:, j].astype(np.float64)) for k, j in pick.items()}
    rng = np.random.default_rng(1440)
    boots = [np.unique(rng.integers(0, n, n)) for _ in range(100)]
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731
    pairs = [(a, b) for i, a in enumerate(M) for b in M[i + 1:]]
    res = {}
    B_all = {}
    for depth in ("d50", "d75", "last"):
        res[depth] = {}
        B = {}
        B_all[depth] = B
        for c in geo:
            def agree(s):
                return float(np.mean([g.rho(sub(geo[c][a][depth], s), sub(geo[c][b][depth], s)) for a, b in pairs]))
            B[c] = np.array([agree(s) for s in boots])
            res[depth][c] = [round(agree(np.arange(n)), 4), round(float(np.percentile(B[c], 2.5)), 4), round(float(np.percentile(B[c], 97.5)), 4)]
        for c in CND:
            dd = B[c] - B["content"]
            res[depth][f"{c}_minus_content"] = [round(float(dd.mean()), 4), round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]
        print(depth, json.dumps(res[depth]), flush=True)
    if cfg.set != "dev":
        qs = [c for c in CND if c.startswith("q_")]
        for depth in ("d75", "last"):
            dq = np.mean([B_all[depth][c] for c in qs], 0) - B_all[depth]["content"]
            dp = np.mean([B_all[depth][c] for c in qs], 0) - B_all[depth]["preamble"]
            res[depth]["questions_minus_content"] = [round(float(dq.mean()), 4), round(float(np.percentile(dq, 2.5)), 4), round(float(np.percentile(dq, 97.5)), 4)]
            res[depth]["questions_minus_preamble"] = [round(float(dp.mean()), 4), round(float(np.percentile(dp, 2.5)), 4), round(float(np.percentile(dp, 97.5)), 4)]
    name = "question_alignment_exploratory.json" if cfg.set == "dev" else "g8_question_alignment.json"
    (g.RES / name).write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"gen": stage_gen, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
