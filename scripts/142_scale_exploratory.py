#!/usr/bin/env python
"""EXPLORATORY (after G6's lock): does the answer geometry stay shared, with no own-model excess, for larger models?
Adds Qwen3-14B, Qwen3-32B (the Qwen3 series 8B -> 14B -> 32B) and Gemma-4-26B-A4B to G6's 1,000 fresh contexts, with the
same code (scripts/132: answers via vLLM, hidden states via HF). New models have no L2 development data, so every model is
compared at fixed relative depths: 75% and the last layer (the same rule for all 11).
  stage=gen model=M     (GPU)    stage=hidden model=M (GPU)    stage=analyze (CPU)
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


g = _mod("g", "129_infogeom.py")
g4 = _mod("g4", "132_g4_shift.py")
g6 = _mod("g6", "137_g6_shared_geometry.py")
OUT = Path("/data/models/graybox_ws/outputs/g6")
NEW = {"qwen3_14b": {"hf": "Qwen/Qwen3-14B", "kw": {"enable_thinking": False}, "prefill": ""},
       "qwen3_32b": {"hf": "Qwen/Qwen3-32B", "kw": {"enable_thinking": False}, "prefill": ""},
       "gemma4_26b": {"hf": "google/gemma-4-26B-A4B-it", "kw": {}, "prefill": ""}}
D = OmegaConf.create({"stage": "analyze", "model": "qwen3_14b", "n_boot": 200})


def _patch():
    L = g4.l2mod()
    L.MODELS.update(NEW)
    g4.l2mod = lambda: L                                           # every call sees the extended model table
    g4.CONDITIONS = {"none": None}
    g4.contexts = lambda split, per_source: g6.ctx_frame()
    g4.OUT = OUT


def stage_gen(cfg):
    _patch()
    g4.stage_gen(OmegaConf.merge(g4.D, {"model": cfg.model, "split": "run"}))


def stage_hidden(cfg):
    """Hidden states at 6 relative depths (L2's fractions) at the last prompt token; the new models have no L2 file."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _patch()
    L = g4.l2mod()
    m = L.MODELS[cfg.model]
    path = L.path_of(m["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    nl = model.config.get_text_config().num_hidden_layers
    layers = sorted({max(1, int(round(nl * f))) for f in (0.25, 0.5, 0.62, 0.75, 0.9, 1.0)})
    C = g6.ctx_frame()
    H = []
    for t in C.text:
        ids = tok(g4.render(tok, t, m, None), add_special_tokens=False, return_tensors="pt")["input_ids"][:, -4096:].cuda()
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        H.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in layers]))
    out = OUT / "run" / cfg.model
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "hidden_none.npz", h=np.stack(H).astype(np.float32), layers=np.array(layers), n_layers=nl)
    print(cfg.model, "hidden", np.stack(H).shape, flush=True)


def depth_index(layers, n_layers, frac):
    return int(np.argmin([abs(l / n_layers - frac) for l in layers]))


def stage_analyze(cfg):
    import importlib.util as iu
    models = g.MODELS + list(NEW)
    Dh75, DhL, Da = {}, {}, {}
    for m in models:
        d = OUT / "run" / m
        hz = np.load(d / "hidden_none.npz")
        layers = [int(x) for x in hz["layers"]]
        nl = int(hz["n_layers"]) if "n_layers" in hz.files else int(np.load(g.L2 / m / "hidden.npz")["n_layers"])
        Dh75[m] = g.cos_dist(hz["h"][:, depth_index(layers, nl, 0.75)].astype(np.float64))
        DhL[m] = g.cos_dist(hz["h"][:, -1].astype(np.float64))
        Da[m] = np.sqrt(g.sq_dist(g.theta(g4.p_of(np.load(d / "psi_none.npz")))))
    n = len(next(iter(Da.values())))
    rng = np.random.default_rng(1420)
    boots = [np.unique(rng.integers(0, n, n)) for _ in range(cfg.n_boot)]
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731
    res = {"models": models}
    for name, Dh in (("depth75", Dh75), ("last", DhL)):
        def mat(s):
            return np.array([[g.rho(sub(Da[a], s), sub(Dh[b], s)) for b in models] for a in models])
        R = mat(np.arange(n))
        resid = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()
        per = {m: round(float(resid[i, i]), 4) for i, m in enumerate(models)}
        b = []
        for s in boots:
            Rs = mat(s)
            rs = Rs - Rs.mean(1, keepdims=True) - Rs.mean(0, keepdims=True) + Rs.mean()
            b.append(np.diag(rs))
        b = np.array(b)
        res[name] = {"excess_all": [round(float(np.diag(resid).mean()), 4), round(float(np.percentile(b.mean(1), 2.5)), 4), round(float(np.percentile(b.mean(1), 97.5)), 4)],
                     "excess_per_model": per,
                     "excess_per_model_ci": {m: [round(float(np.percentile(b[:, i], 2.5)), 4), round(float(np.percentile(b[:, i], 97.5)), 4)] for i, m in enumerate(models)},
                     "own_rho": {m: round(float(R[i, i]), 4) for i, m in enumerate(models)},
                     "hidden_vs_hidden_qwen_series": {f"{a}-{c}": round(g.rho(Dh[a], Dh[c]), 4) for a, c in (("qwen3_8b", "qwen3_14b"), ("qwen3_8b", "qwen3_32b"), ("qwen3_14b", "qwen3_32b"))}}
        print(name, json.dumps(res[name]["excess_all"]), json.dumps({k: res[name]["excess_per_model"][k] for k in ("qwen3_8b", "qwen3_14b", "qwen3_32b", "gemma4_26b")}), flush=True)
    (g.RES / "scale_exploratory.json").write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"gen": stage_gen, "hidden": stage_hidden, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
