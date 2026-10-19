#!/usr/bin/env python
"""G6: the observable geometry is shared across models. Fresh contexts (scripts/136): 1,000, of which 400 are of four
kinds never used in this project. Data per model in one pass (scripts/132's code, no system message): P(yes) on the 88
battery-v2 questions (logprob tier), 8 replies, hidden states at L2's 6 layers. Layer: G1's dev-chosen layer (fixed).
  stage=gen    model=M        (GPU)  answers + replies
  stage=hidden model=M        (GPU)
  stage=embed                 (CPU)  MiniLM of the contexts (text) and of the replies
  stage=analyze [data=DIR contexts=FILE tag=T]   (CPU)
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
OUT = Path("/data/models/graybox_ws/outputs/g6")
RES = ROOT / "results/t1"
READER = "qwen3_8b"
D = OmegaConf.create({"stage": "analyze", "model": "qwen3_8b", "data": str(OUT / "run"), "contexts": str(OUT / "contexts.parquet"),
                      "tag": "confirm", "n_boot": 500, "margin": 0.02})


def ctx_frame():
    C = pd.read_parquet(OUT / "contexts.parquet").reset_index(drop=True)
    return C


def _patch():
    g4.CONDITIONS = {"none": None}
    g4.contexts = lambda split, per_source: ctx_frame()
    g4.OUT = OUT


def stage_gen(cfg):
    _patch()
    g4.stage_gen(OmegaConf.merge(g4.D, {"model": cfg.model, "split": "run"}))


def stage_hidden(cfg):
    _patch()
    g4.stage_hidden(OmegaConf.merge(g4.D, {"model": cfg.model, "split": "run"}))


def stage_embed(cfg):
    _patch()
    g4.stage_embed(OmegaConf.merge(g4.D, {"split": "run"}))
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    p = snapshot_download("sentence-transformers/all-MiniLM-L6-v2", cache_dir="/data/models", local_files_only=True)
    tok, enc = AutoTokenizer.from_pretrained(p), AutoModel.from_pretrained(p).eval()
    texts, E = ctx_frame().text.tolist(), []
    with torch.no_grad():
        for i in range(0, len(texts), 64):              # the L2 recipe: max 256 tokens, masked mean, L2 normalisation
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = enc(**b).last_hidden_state
            mk = b["attention_mask"][..., None].float()
            E.append(torch.nn.functional.normalize((h * mk).sum(1) / mk.sum(1), dim=-1).numpy())
    np.save(OUT / "run" / "contexts_minilm.npy", np.concatenate(E).astype(np.float32))


def stage_analyze(cfg):
    data, C = Path(cfg.data), pd.read_parquet(cfg.contexts).reset_index(drop=True)
    kind, new = C["kind"].values, C["new_kind"].values.astype(bool)
    Call = g.contexts()
    dev = np.where(Call.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    M = g.MODELS
    Dh, Da, Dr, layer = {}, {}, {}, {}
    for m in M:
        k, layer[m], _ = g.select_layer(m, dev_sel)
        Dh[m] = g.cos_dist(np.load(data / m / "hidden_none.npz")["h"][:, k].astype(np.float64))
        Da[m] = np.sqrt(g.sq_dist(g.theta(g4.p_of(np.load(data / m / "psi_none.npz")))))
        f = data / m / "replies_none.minilm.npy"
        if f.exists():
            Dr[m] = g.cos_dist(np.load(f).reshape(len(C), 8, -1).astype(np.float64).mean(1))
    Dt = g.cos_dist(np.load(data / "contexts_minilm.npy").astype(np.float64))
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731

    def matrix(s):
        return np.array([[g.rho(sub(Da[a], s), sub(Dh[b], s)) for b in M] for a in M])

    def excess(R):
        res = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()
        return float(np.diag(res).mean())
    n = len(C)
    R = matrix(np.arange(n))
    res = {"tag": cfg.tag, "n_contexts": n, "n_new_kind": int(new.sum()), "layers": layer,
           "rho_answers_A_vs_hidden_B": {a: {b: round(float(R[i, j]), 4) for j, b in enumerate(M)} for i, a in enumerate(M)}}
    rng = np.random.default_rng(1360)
    bS1 = []
    for _ in range(cfg.n_boot):
        s = np.unique(rng.integers(0, n, n))
        bS1.append(excess(matrix(s)))
    res["S1_diagonal_excess"] = g.ci(np.array(bS1))
    res["S1_point"] = round(excess(R), 4)
    res["S1_pass"] = res["S1_diagonal_excess"][1] > -cfg.margin and res["S1_diagonal_excess"][2] < cfg.margin
    res["S2_reader_hidden_minus_own_answers"] = {}
    for i, a in enumerate(M):
        if a == READER:
            continue
        b = g.boot(lambda s: g.rho(sub(Dh[READER], s), sub(Dh[a], s)) - g.rho(sub(Da[a], s), sub(Dh[a], s)), n, cfg.n_boot, 1370 + i)
        res["S2_reader_hidden_minus_own_answers"][a] = g.ci(b)
    res["S2_pass"] = sum(v[1] > 0 for v in res["S2_reader_hidden_minus_own_answers"].values()) >= 5
    idx_new = np.where(new)[0]
    res["S3_new_kinds_answers_minus_text"] = {}
    for i, a in enumerate(M):
        A, H, T = sub(Da[a], idx_new), sub(Dh[a], idx_new), sub(Dt, idx_new)
        b = g.boot(lambda s: g.rho(sub(A, s), sub(H, s)) - g.rho(sub(T, s), sub(H, s)), len(idx_new), cfg.n_boot, 1380 + i)
        res["S3_new_kinds_answers_minus_text"][a] = g.ci(b)
    res["S3_pass"] = sum(v[1] > 0 for v in res["S3_new_kinds_answers_minus_text"].values()) >= 6
    # reported only
    rep = {}
    for a in M:
        r = {"answers": round(g.rho(Da[a], Dh[a]), 4), "text": round(g.rho(Dt, Dh[a]), 4),
             "reader_hidden": round(g.rho(Dh[READER], Dh[a]), 4) if a != READER else None,
             "answers_new_kinds": round(g.rho(sub(Da[a], idx_new), sub(Dh[a], idx_new)), 4),
             "text_new_kinds": round(g.rho(sub(Dt, idx_new), sub(Dh[a], idx_new)), 4)}
        if a in Dr:
            r["replies"] = round(g.rho(Dr[a], Dh[a]), 4)
        r["per_kind"] = {k: {"answers": round(g.rho(sub(Da[a], np.where(kind == k)[0]), sub(Dh[a], np.where(kind == k)[0])), 4),
                             "text": round(g.rho(sub(Dt, np.where(kind == k)[0]), sub(Dh[a], np.where(kind == k)[0])), 4)}
                         for k in sorted(set(kind))}
        rep[a] = r
    res["reported"] = rep
    res["reported_hidden_vs_hidden"] = {a: {b: round(g.rho(Dh[a], Dh[b]), 4) for b in M} for a in M}
    name = "g6_shared_geometry.json" if cfg.tag == "confirm" else f"g6_shared_geometry_DRYRUN_{cfg.tag}.json"
    (RES / name).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("S1_point", "S1_diagonal_excess", "S1_pass", "S2_pass", "S3_pass")}, indent=1))
    print("S2", json.dumps(res["S2_reader_hidden_minus_own_answers"]))
    print("S3", json.dumps(res["S3_new_kinds_answers_minus_text"]))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"gen": stage_gen, "hidden": stage_hidden, "embed": stage_embed, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
