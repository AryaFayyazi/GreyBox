#!/usr/bin/env python
"""Live v2 truth overlay for an OPEN model: decoder θ(Ψ_v2m) -> true hidden state, refit for the v2m battery.

The v1 overlay decoders (scripts/22) were fit on battery-v2 answers. Live v2 asks the v2m questions, so the overlay is
refit here on the Live benchmark's single-turn conversations (kinds single / toxicchat / xstest), whose context is
exactly [user: message] and can be reproduced:
  hidden  the model's residual stream at the generation-prompt token (the state the battery forks from; no
          question appended), at the same mid-depth layer as the v1 overlay
  frame   PCA (top 32) of the hidden states of TRAIN items; display = top 3
  decoder ridge from θ of the 88 v2m items (livebench psi.npz) to the 32 PCs, fit on TRAIN
  check   EVAL items: variance-weighted R² of the 32 PCs and per-axis correlation of the 3D display
Saved to /data/models/graybox_ws/models/live/open_v2m_<model>.npz (graybox.live uses it with a v2m bundle).

python scripts/40_live_open_v2m.py model=qwen3_8b
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
run12 = importlib.import_module("12_l2_run")
LB = Path("/data/models/graybox_ws/outputs/livebench")
L2 = Path("/data/models/graybox_ws/outputs/l2")
MOD = Path("/data/models/graybox_ws/models/live")
RES = Path("/home/arya/graybox/results/live")
D = OmegaConf.create({"model": "qwen3_8b"})


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    m = run12.MODELS[cfg.model]
    path = run12.path_of(m["hf"])
    layers = np.load(L2 / cfg.model / "hidden.npz")["layers"]
    layer = int(layers[len(layers) // 2])                              # the v1 overlay's layer
    items = pd.read_parquet(LB / "items.parquet")
    one = np.flatnonzero((items.kind != "multiturn").values & (items.turns.map(len) == 1).values)
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    H = np.zeros((len(one), model.config.get_text_config().hidden_size), np.float32)
    with torch.no_grad():
        for r, i in enumerate(one):
            s = tok.apply_chat_template([{"role": "user", "content": items.turns.iloc[i][-1]}], tokenize=False,
                                        add_generation_prompt=True, **m["kw"])
            ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][:, -4096:].cuda()
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
            H[r] = hs[layer][0, -1].float().cpu().numpy()
    assert np.isfinite(H).all(), "non-finite hidden states"
    z = np.load(LB / cfg.model / "psi.npz")
    q = (z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6)
    X = 2 * np.arcsin(np.sqrt(q[one]))
    tr = (items.split.values[one] == "train")
    te = ~tr
    mu = H[tr].mean(0)
    _, S, Vt = np.linalg.svd(H[tr] - mu, full_matrices=False)
    comps, w = Vt[:32], S[:32] ** 2 / (S ** 2).sum()
    P = (H - mu) @ comps.T
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((X[tr] - xm) / xs, P[tr])
    Ph = reg.predict((X[te] - xm) / xs)
    r2 = 1 - ((P[te] - Ph) ** 2).sum(0) / ((P[te] - P[te].mean(0)) ** 2).sum(0)
    scale = 1.0 / np.percentile(np.linalg.norm(P[tr][:, :3], axis=1), 97)
    corr = [round(float(np.corrcoef(P[te][:, k], Ph[:, k])[0, 1]), 4) for k in range(3)]
    rep = {"layer": layer, "battery": "v2m", "R2_top32_weighted": round(float((r2 * w).sum() / w.sum()), 4),
           "display_axis_corr": corr, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    MOD.mkdir(parents=True, exist_ok=True)
    np.savez(MOD / f"open_v2m_{cfg.model}.npz", items=z["items"], xm=xm, xs=xs, coef=reg.coef_, intercept=reg.intercept_,
             mu=mu, comps=comps, scale=scale, layer=layer, check=json.dumps(rep))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"open_v2m_{cfg.model}.json").write_text(json.dumps(rep, indent=1))
    print(cfg.model, rep, flush=True)


if __name__ == "__main__":
    main()
