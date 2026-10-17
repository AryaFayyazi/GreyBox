#!/usr/bin/env python
"""Ground truth for the grey-box study: the actor's residual stream at the TURN-START token of each unit
(the last token of the turn-start prompt rendered by scripts/01, i.e. the state from which the agent begins its turn).

Layers: CAPE's depth grid (0.5, 0.62, 0.72, 0.82, 0.9 of depth) plus the final layer, so that the white-box comparator
matches CAPE and the output-layer state (which Tier-1 theory addresses) is available.

python scripts/02_hidden_turnstart.py actor=gemma4 tag=pilot
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.cape_bridge import ACT  # noqa: E402

OUT = Path("/data/models/graybox_ws/outputs/interrogate")
D = OmegaConf.create({"actor": "gemma4", "tag": "pilot", "reader": None})   # reader: hidden states of a READER model on the actor's transcript (Model Hermeneutics baseline)


def layers_for(n):
    return sorted({int(round(n * f)) - 1 for f in (0.5, 0.62, 0.72, 0.82, 0.9)} | {n - 1})


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    import torch
    from cape.models.load import discover_transformer_blocks, model_path
    from cape.utils.config import resolve_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    torch.backends.cuda.enable_cudnn_sdp(False)
    path = model_path(resolve_model(ACT[cfg.reader or cfg.actor]["model"]))
    tok = AutoTokenizer.from_pretrained(path)
    stem = OUT / (f"{cfg.actor}_{cfg.tag}" + (f"_reader-{cfg.reader}" if cfg.reader else ""))
    prompts = pd.read_parquet(f"{stem}_turn_prompts.parquet").turn_prompt.tolist()
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    blocks, _ = discover_transformer_blocks(model)
    L = layers_for(len(blocks))
    print(f"loaded {cfg.actor} {time.time()-t0:.0f}s; {len(blocks)} blocks; layers {L}; {len(prompts)} units", flush=True)
    cap = {}

    def hook_for(l):
        def fn(mod, inp, outp):
            h = outp[0] if isinstance(outp, tuple) else outp
            cap[l] = h[0, -1].float().cpu().numpy()
        return fn
    hooks = [blocks[l].register_forward_hook(hook_for(l)) for l in L]
    feats = []
    for i, p in enumerate(prompts):
        ids = tok(p, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]).cuda(), logits_to_keep=1, use_cache=False)
        feats.append(np.stack([cap[l] for l in L]).astype(np.float16))
        if i % 250 == 0:
            print(f"{i}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    H = np.stack(feats)
    np.savez_compressed(f"{stem}_hidden.npz", h=H, layers=np.array(L), n_blocks=len(blocks))
    print(f"saved {H.shape} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
