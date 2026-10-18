#!/usr/bin/env python
"""G5: which outputs carry the state? A model's yes/no answers vs its own replies, as geometries of its hidden state.

Across contexts, G1 compared the answer geometry with the geometry of the input text. A model's replies are the other
output an API gives. G5 asks which output geometry matches the model's own hidden geometry, at stated costs:
  answers  P(yes) on the 88 battery-v2 questions, read exactly (logprob tier), or 1 sampled answer per question
           (sample tier: yes / no / other from the raw probabilities; 88 calls, 1 output token each)
  replies  8 replies sampled at temperature 1, 128 tokens, embedded with all-MiniLM-L6-v2: their mean (8 calls), or
           the first one (1 call)
  hidden   the residual stream at the last prompt token, at G1's dev-chosen layer (ground truth only)
Data: one fresh pass per model (answers, replies, hidden; no system message), so the three share one rendering
(gpt-oss's template inserts the date). Contexts: dev = G4's 480 dev contexts (the dry run); confirm = all 1,440 L2
confirm contexts.
  stage=gen    model=M split=confirm   (GPU)  -> answers + replies, then hidden states
  stage=embed  split=confirm
  stage=analyze split=dev|confirm      (CPU)
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
OUT = Path("/data/models/graybox_ws/outputs/g5")
RES = ROOT / "results/t1"
D = OmegaConf.create({"stage": "analyze", "model": "qwen3_8b", "split": "confirm", "n_boot": 500, "replies": 8,
                      "reply_tokens": 128})


def contexts(split):
    if split == "dev":
        return g4.contexts("dev", 80)
    C = g4.pd.read_parquet(g4.L2 / "contexts.parquet").reset_index().rename(columns={"index": "row"})
    return C[C.split == "confirm"].reset_index(drop=True)


def data_dir(split, m):
    return (g4.OUT / "dev" / m) if split == "dev" else (OUT / split / m)


def stage_gen(cfg):
    """G4's generation code with the 'none' condition only, on this split's contexts."""
    g4.CONDITIONS = {"none": None}
    g4.contexts = lambda split, per_source: contexts(split)
    g4.OUT = OUT
    c = OmegaConf.merge(g4.D, {"model": cfg.model, "split": cfg.split, "replies": cfg.replies, "reply_tokens": cfg.reply_tokens})
    g4.stage_gen(c)


def stage_hidden(cfg):
    g4.CONDITIONS = {"none": None}
    g4.contexts = lambda split, per_source: contexts(split)
    g4.OUT = OUT
    g4.stage_hidden(OmegaConf.merge(g4.D, {"model": cfg.model, "split": cfg.split}))


def stage_embed(cfg):
    g4.OUT = OUT
    g4.stage_embed(OmegaConf.merge(g4.D, {"split": cfg.split}))


def stage_analyze(cfg):
    C = contexts(cfg.split)
    Call = g.contexts()
    dev = np.where(Call.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    rng = np.random.default_rng(1330)
    res = {"split": cfg.split, "n_contexts": len(C), "models": {}}
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731
    for mi, m in enumerate(g.MODELS):
        d = data_dir(cfg.split, m)
        k, layer, _ = g.select_layer(m, dev_sel)                            # G1's layer (dev only)
        Dh = g.cos_dist(np.load(d / "hidden_none.npz")["h"][:, k].astype(np.float64))
        z = np.load(d / "psi_none.npz")
        P = g4.p_of(z)
        E = np.load(d / "replies_none.minilm.npy").reshape(len(C), cfg.replies, -1).astype(np.float64)
        Dx = np.sqrt(g.sq_dist(g.theta(P)))                                  # answers, exact (logprob tier)
        Ds = np.sqrt(g.sq_dist(g4.one_sample_angles(z, list(range(P.shape[1])), rng)))   # answers, 1 sample each
        R8, R1 = g.cos_dist(E.mean(1)), g.cos_dist(E[:, 0])                  # replies: mean of 8, first one
        Dt = g.cos_dist(np.load(g.L2 / "minilm.npy").astype(np.float64)[C.row.values])
        r = {"layer": layer, **{f"rho_{n}": round(g.rho(A, Dh), 4) for n, A in
                                (("answers_exact88", Dx), ("answers_1sample88", Ds), ("replies_mean8", R8), ("reply_1", R1), ("context_text", Dt))}}
        for name, (A, B), seed in (("H1_exact88_minus_replies8", (Dx, R8), 1331), ("H2_1sample88_minus_reply1", (Ds, R1), 1341),
                                   ("H3_replies8_minus_reply1", (R8, R1), 1351)):
            b = g.boot(lambda s: g.rho(sub(A, s), sub(Dh, s)) - g.rho(sub(B, s), sub(Dh, s)), len(C), cfg.n_boot, seed + mi)
            r[name] = g.ci(b)
        res["models"][m] = r
        print(m, json.dumps(r), flush=True)
    ms = res["models"].values()
    res["H1_pass"] = sum(x["H1_exact88_minus_replies8"][1] > 0 for x in ms) >= 7
    res["H2_pass"] = sum(x["H2_1sample88_minus_reply1"][1] > 0 for x in ms) >= 6
    res["H3_pass"] = sum(x["H3_replies8_minus_reply1"][2] < 0.05 for x in ms) >= 7
    name = "g5_answers_vs_replies.json" if cfg.split == "confirm" else "g5_answers_vs_replies_DRYRUN_dev.json"
    (RES / name).write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: res[k_] for k_ in ("H1_pass", "H2_pass", "H3_pass")}))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"gen": stage_gen, "hidden": stage_hidden, "embed": stage_embed, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
