#!/usr/bin/env python
"""G73 (docs/prereg_G73_claude_fresh_geometry.md): does a CLOSED model's answer geometry reveal the hidden geometry that
open models share, better than any text a monitor could embed, on fresh contexts? Claude Haiku 5.5 answers G2's 24
battery items (graybox.adapters.fork; 8 samples each; scripts/32's frame and parser) on 400 of G6's 1,000 fresh contexts
(numpy seed 7300), and gives 8 real replies to each unmodified context. Compared, for each of G6's 8 open models, with
that model's hidden geometry on the same contexts (G6's stored states, G1's development-chosen layer, cosine):
  Claude's answer geometry (Theorem 5's debiased Fisher-Rao distances)
  vs text baselines: the input under MiniLM, Qwen3-Embedding-0.6B and MPNet; Claude's own replies under Qwen3-Embedding
  and MPNet (mean of 8 replies, as for the open models' replies in scripts/147).
  stage=collect cap=.. [n_ctx=..] (paid) ; stage=embed (CPU) ; stage=analyze -> results/t1/g73_claude_fresh_geometry.json
"""
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
N_CTX, N_SAMP, N_REP, SEED, B_N, BOOT_SEED = 400, 8, 8, 7300, 500, 7301
OUT = Path("/data/models/graybox_ws/outputs/g73")
CTX = Path("/data/models/graybox_ws/outputs/g6/contexts.parquet")
RUN = Path("/data/models/graybox_ws/outputs/g6/run")
SE = Path("/data/models/graybox_ws/outputs/strong_embed")


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def picks():
    C = pd.read_parquet(CTX)
    idx = np.sort(np.random.default_rng(SEED).choice(len(C), N_CTX, replace=False))
    return C, idx


def items():
    from graybox.battery_v2 import BATTERY_V2
    keep = list(np.load("/data/models/graybox_ws/models/live/readouts.npz", allow_pickle=True)["items"])
    return [(k, q) for k, q in BATTERY_V2 if k in keep]


def stage_collect(cap, n_ctx=None):
    import anthropic
    from graybox.adapters import fork
    from graybox.budget import HAIKU55, Ledger
    L3 = _mod("l3", "32_claude_l3.py")
    led = Ledger(Path("/data/models/graybox_ws/outputs/l3/spend.json"), float(cap), *HAIKU55)
    C, idx = picks()
    idx = idx[:n_ctx] if n_ctx else idx
    its = items()
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / ("claude.jsonl" if not n_ctx else f"claude_smoke{n_ctx}.jsonl")
    done = {json.loads(x)["i"] for x in open(fn)} if fn.exists() else set()
    client = anthropic.AsyncAnthropic(api_key=L3.KEY.read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(24)

    async def call(msgs, mt):
        led.check()
        async with sem:
            r = await client.messages.create(model=L3.MODEL, max_tokens=mt, messages=msgs, thinking={"type": "disabled"}, output_config={"effort": "low"})
        led.charge(r.usage.input_tokens, r.usage.output_tokens)
        return "".join(b.text for b in r.content if b.type == "text")

    async def one(i):
        text = C.text.values[i]
        counts = []
        for _, q in its:
            labs = [L3.classify(o) for o in await asyncio.gather(*[call(fork(text, q), 3) for _ in range(N_SAMP)])]
            counts.append([sum(x == "yes" for x in labs), sum(x == "no" for x in labs), sum(x is None for x in labs)])
        reps = await asyncio.gather(*[call([{"role": "user", "content": text}], 400) for _ in range(N_REP)])
        return {"i": int(i), "counts": counts, "replies": reps}

    async def go():
        t0 = time.time()
        todo = [i for i in idx if int(i) not in done]
        for j in range(0, len(todo), 8):
            res = await asyncio.gather(*[one(i) for i in todo[j:j + 8]])
            with open(fn, "a") as f:
                for r in res:
                    f.write(json.dumps(r) + "\n")
            print(f"{j + len(res)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def stage_embed():
    from graybox.embed import Encoder
    R = [json.loads(x) for x in open(OUT / "claude.jsonl")]
    R.sort(key=lambda r: r["i"])
    for enc_name in ("qemb06", "mpnet"):
        enc = Encoder(enc_name, device="cpu", cache_dir="/data/models")
        reps = [x for r in R for x in r["replies"]]
        E = enc._embed(reps).reshape(len(R), N_REP, -1).mean(1)
        np.save(OUT / f"claude_replies_{enc_name}.npy", E)
        print(enc_name, E.shape, flush=True)


def stage_analyze():
    g = _mod("g", "129_infogeom.py")
    g4 = _mod("g4", "132_g4_shift.py")
    R = sorted((json.loads(x) for x in open(OUT / "claude.jsonl")), key=lambda r: r["i"])
    ix = np.array([r["i"] for r in R])
    cnt = np.array([[[c[0], c[0] + c[1]] for c in r["counts"]] for r in R], float)      # [yes, parsed]
    k, n = cnt[..., 0], np.maximum(cnt[..., 1], 1)
    th = 2 * np.arcsin(np.sqrt((k + 3 / 8) / (n + 3 / 4)))
    var = 1 / (n + 0.5)
    G = th @ th.T
    nr = np.diag(G)
    Dc = np.sqrt(np.maximum(nr[:, None] + nr[None, :] - 2 * G - (var.sum(1)[:, None] + var.sum(1)[None, :]), 0))
    np.fill_diagonal(Dc, 0)
    sub = lambda A: A[np.ix_(ix, ix)]  # noqa: E731
    base = {"input_minilm": sub(g.cos_dist(np.load(RUN / "contexts_minilm.npy").astype(np.float64))),
            "input_qemb06": sub(g.cos_dist(np.load(SE / "fresh_input_qemb06.npy").astype(np.float64))),
            "input_mpnet": sub(g.cos_dist(np.load(SE / "fresh_input_mpnet.npy").astype(np.float64))),
            "claude_replies_qemb06": g.cos_dist(np.load(OUT / "claude_replies_qemb06.npy").astype(np.float64)),
            "claude_replies_mpnet": g.cos_dist(np.load(OUT / "claude_replies_mpnet.npy").astype(np.float64))}
    C = g.contexts()
    dev = np.where(C.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)                 # scripts/147's layer choice
    rng, res = np.random.default_rng(BOOT_SEED), {"contexts": int(len(ix)), "parsed_share": round(float(cnt[..., 1].sum() / (len(R) * cnt.shape[1] * 8)), 4), "models": {}}
    for m in g.MODELS:
        kk, _, _ = g.select_layer(m, dev_sel)
        Dh = sub(g.cos_dist(np.load(RUN / m / "hidden_none.npz")["h"][:, kk].astype(np.float64)))
        Da_open = sub(np.sqrt(g.sq_dist(g.theta(g4.p_of(np.load(RUN / m / "psi_none.npz"))))))
        r = {"claude_answers": g.rho(Dc, Dh), "open_own_answers": g.rho(Da_open, Dh), **{b: g.rho(D, Dh) for b, D in base.items()}}
        best = max(base, key=lambda b: r[b])
        nn = len(ix)
        bb = []
        for _ in range(B_N):
            s = np.unique(rng.integers(0, nn, nn))
            q = lambda A: A[np.ix_(s, s)]  # noqa: E731
            bb.append(g.rho(q(Dc), q(Dh)) - g.rho(q(base[best]), q(Dh)))
        res["models"][m] = {**{x: round(float(v), 4) for x, v in r.items()}, "strongest_text_baseline": best,
                            "claude_minus_strongest": [round(float(np.mean(bb)), 4), round(float(np.percentile(bb, 2.5)), 4), round(float(np.percentile(bb, 97.5)), 4)]}
        print(m, json.dumps(res["models"][m]), flush=True)
    wins = int(sum(v["claude_minus_strongest"][1] > 0 for v in res["models"].values()))
    res["E1_models_claude_beats_strongest_text"] = wins
    res["endpoints"] = {"E1_at_least_6_of_8": wins >= 6}
    (ROOT / "results/t1/g73_claude_fresh_geometry.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    st = kw.get("stage", "analyze")
    if st == "collect":
        stage_collect(kw["cap"], int(kw["n_ctx"]) if "n_ctx" in kw else None)
    elif st == "embed":
        stage_embed()
    else:
        stage_analyze()
