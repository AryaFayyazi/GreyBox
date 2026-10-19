#!/usr/bin/env python
"""EXPLORATORY (second review, weakness 2): do the answers still beat the input text and the replies when the text
encoder is strong? Qwen3-Embedding-0.6B (last-token pooling) and all-mpnet-base-v2 (mean pooling), from graybox.embed,
embed (a) the input context and (b) each of the model's 8 sampled replies (mean of the 8 embeddings). Spearman with each
model's hidden geometry at G1's dev-chosen layer, beside the answer geometry, on G1's 1,440 confirm contexts (replies:
G5's confirm pass) and on G6's 1,000 fresh contexts. Bootstrap over contexts for answers − strongest embedding.
  stage=embed (GPU)    stage=analyze (CPU)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
OUT = Path("/data/models/graybox_ws/outputs/strong_embed")
SETS = {"confirm": (Path("/data/models/graybox_ws/outputs/g5/confirm"), None), "fresh": (Path("/data/models/graybox_ws/outputs/g6/run"), Path("/data/models/graybox_ws/outputs/g6/contexts.parquet"))}
D = OmegaConf.create({"stage": "analyze", "n_boot": 300})


def texts(which):
    if which == "confirm":
        C = g.contexts()
        return C[C.split == "confirm"].text.tolist()
    return pd.read_parquet(SETS[which][1]).text.tolist()


def stage_embed(cfg):
    from graybox.embed import Encoder
    OUT.mkdir(parents=True, exist_ok=True)
    for enc_name in ("qemb06", "mpnet"):
        enc = Encoder(enc_name, device="cuda", cache_dir="/data/models")
        for which, (d, _) in SETS.items():
            np.save(OUT / f"{which}_input_{enc_name}.npy", enc._embed(texts(which)))
            for m in g.MODELS:
                reps = pd.read_parquet(d / m / "replies_none.parquet").reply.tolist()
                E = enc._embed(reps).reshape(len(reps) // 8, 8, -1).mean(1)
                np.save(OUT / f"{which}_replies_{m}_{enc_name}.npy", E)
            print(enc_name, which, "done", flush=True)


def stage_analyze(cfg):
    C = g.contexts()
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    conf = np.where(C.split.values == "confirm")[0]
    rng = np.random.default_rng(1470)
    res = {}
    for which, (d, _) in SETS.items():
        res[which] = {}
        Ein = {e: g.cos_dist(np.load(OUT / f"{which}_input_{e}.npy").astype(np.float64)) for e in ("qemb06", "mpnet")}
        Emini = g.cos_dist(np.load(g.L2 / "minilm.npy").astype(np.float64)[conf]) if which == "confirm" else \
            g.cos_dist(np.load(Path("/data/models/graybox_ws/outputs/g6/run/contexts_minilm.npy")).astype(np.float64))
        for m in g.MODELS:
            k, _, _ = g.select_layer(m, dev_sel)
            if which == "confirm":
                H, _ = g.hidden(m)
                Dh = g.cos_dist(H[conf, k])
                p, _ = g.p_yes(m)
                Da = np.sqrt(g.sq_dist(g.theta(p[conf])))
            else:
                Dh = g.cos_dist(np.load(d / m / "hidden_none.npz")["h"][:, k].astype(np.float64))
                Da = np.sqrt(g.sq_dist(g.theta(g4.p_of(np.load(d / m / "psi_none.npz")))))
            Erep = {e: g.cos_dist(np.load(OUT / f"{which}_replies_{m}_{e}.npy").astype(np.float64)) for e in ("qemb06", "mpnet")}
            r = {"answers": g.rho(Da, Dh), "input_minilm": g.rho(Emini, Dh), **{f"input_{e}": g.rho(Ein[e], Dh) for e in Ein},
                 **{f"replies_{e}": g.rho(Erep[e], Dh) for e in Erep}}
            best = max((k_ for k_ in r if k_ != "answers"), key=r.get)
            Dbest = {"input_minilm": Emini, **{f"input_{e}": Ein[e] for e in Ein}, **{f"replies_{e}": Erep[e] for e in Erep}}[best]
            n = len(Da)
            sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731
            bb = [g.rho(sub(Da, s), sub(Dh, s)) - g.rho(sub(Dbest, s), sub(Dh, s)) for s in (np.unique(rng.integers(0, n, n)) for _ in range(cfg.n_boot))]
            r = {k_: round(v, 4) for k_, v in r.items()}
            r["strongest_text_baseline"] = best
            r["answers_minus_strongest"] = [round(float(np.mean(bb)), 4), round(float(np.percentile(bb, 2.5)), 4), round(float(np.percentile(bb, 97.5)), 4)]
            res[which][m] = r
            print(which, m, json.dumps(r), flush=True)
    (g.RES / "strong_embedders_exploratory.json").write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"embed": stage_embed, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
