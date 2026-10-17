#!/usr/bin/env python
"""Grey-box interrogation state Ψ at the turn start of CAPE decision units, with the actor served as a black box.

For each unit and each battery item, one prompt = turn-start context + user question + answer prefill. The simulated
API returns the top-20 next-token logprobs (what OpenAI/DeepSeek-style APIs expose). From them we keep
log P(yes-variants) and log P(no-variants). The text-only tier is simulated later by sampling answers from these
probabilities, which is exactly what a sampling-only API would return for a one-word answer.

python scripts/01_interrogate.py actor=gemma4 n_xother=2000 n_xplus=2000 tag=pilot
python scripts/01_interrogate.py actor=gemma4 stage=check     # CPU: render 2 units, print prompts
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.battery import BATTERY_V1, PREAMBLE, battery_sha  # noqa: E402
from graybox.cape_bridge import ACT, load_units, question_prompt, render, rollouts, specs, turn_start_messages  # noqa: E402

OUT = Path("/data/models/graybox_ws/outputs/interrogate")
D = OmegaConf.create({"actor": "gemma4", "stage": "run", "n_viol": None, "n_xother": 2000, "n_xplus": 2000,
                      "seed": 0, "tag": "pilot", "max_len": 32768, "units": None,
                      "reader": None,
                      "max_num_seqs": None})  # hybrid-attention models (Qwen3.8: Mamba-style cache) need <= cache blocks  # reader: another model placed in the agent's shoes (Tier-3 control)


def pick_units(cfg):
    if cfg.units:
        return pd.read_parquet(cfg.units)
    u = load_units(cfg.actor)
    rng = np.random.default_rng(cfg.seed)
    parts = []
    for cls, n in (("VIOL", cfg.n_viol), ("XOTHER", cfg.n_xother), ("XPLUS", cfg.n_xplus)):
        g = u[u.cls == cls]
        parts.append(g if n is None or len(g) <= n else g.iloc[rng.choice(len(g), int(n), replace=False)])
    return pd.concat(parts).sort_values(["set", "episode_id", "event_id"]).reset_index(drop=True)


def build(cfg, tok, units):
    who = cfg.reader or cfg.actor
    a = ACT[who]
    turn, qprompts = [], []
    for r in units.itertuples():
        ep = rollouts(r.set).loc[r.episode_id]
        msgs = turn_start_messages(specs(r.bench)[ep.instance_id], list(ep.events), int(r.event_id), a["suffix"])
        turn.append(render(tok, msgs, a["kwargs"], gen=True))
        qprompts.append([question_prompt(tok, msgs, who, q, PREAMBLE) for _, q in BATTERY_V1])
    return turn, qprompts


def yes_no(lp_dict):
    py = pn = 0.0
    top = None
    for tid, lp in lp_dict.items():
        s = (lp.decoded_token or "").strip().strip(".").lower()
        p = float(np.exp(lp.logprob))
        if top is None or lp.rank == 1:
            top = lp.decoded_token if lp.rank == 1 else top
        if s == "yes":
            py += p
        elif s == "no":
            pn += p
    return py, pn, top


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    a = ACT[cfg.reader or cfg.actor]
    from cape.models.load import model_path
    from cape.utils.config import resolve_model
    from transformers import AutoTokenizer
    path = model_path(resolve_model(a["model"]))
    tok = AutoTokenizer.from_pretrained(path)
    units = pick_units(cfg)
    print(f"{cfg.actor}: {len(units)} units {units.cls.value_counts().to_dict()} battery {battery_sha()[:12]}", flush=True)
    if cfg.stage == "prompts":   # CPU only: render and save the turn-start prompts (for hidden-state extraction alone)
        turn, _ = build(cfg, tok, units)
        stem = OUT / (f"{cfg.actor}_{cfg.tag}" + (f"_reader-{cfg.reader}" if cfg.reader else ""))
        OUT.mkdir(parents=True, exist_ok=True)
        units.to_parquet(f"{stem}_units.parquet")
        pd.Series(turn).to_frame("turn_prompt").to_parquet(f"{stem}_turn_prompts.parquet")
        print("saved", len(turn), "turn prompts to", stem)
        return
    if cfg.stage == "check":
        units = units.groupby("cls").head(1)
        turn, qp = build(cfg, tok, units)
        for t, q in zip(turn, qp):
            print("=== TURN START (tail)\n", t[-500:], "\n=== Q0 (tail)\n", q[0][-400:])
        return
    t0 = time.time()
    turn, qp = build(cfg, tok, units)
    lens = np.array([len(x) for x in tok(turn, add_special_tokens=False)["input_ids"]])
    keep = lens + 200 < cfg.max_len
    print(f"rendered in {time.time()-t0:.0f}s; tokens median {np.median(lens):.0f} max {lens.max()}; "
          f"dropping {(~keep).sum()} over-long", flush=True)
    units, turn, qp, lens = units[keep].reset_index(drop=True), [t for t, k in zip(turn, keep) if k], \
        [q for q, k in zip(qp, keep) if k], lens[keep]
    from vllm import LLM, SamplingParams
    extra = {"max_num_seqs": int(cfg.max_num_seqs)} if cfg.max_num_seqs else {}
    llm = LLM(model=path, enable_prefix_caching=True, max_model_len=cfg.max_len, gpu_memory_utilization=0.88,
              trust_remote_code=True, max_logprobs=20, **extra)
    sp = SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, seed=0)
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"      # batch encoding in the Rust backend (was ~25 min one by one)
    allp = [p for qs in qp for p in qs]
    flat = []
    for i in range(0, len(allp), 4096):
        flat += [{"prompt_token_ids": ids} for ids in tok(allp[i:i + 4096], add_special_tokens=False)["input_ids"]]
    print(f"vLLM ready {time.time()-t0:.0f}s; {len(flat)} prompts", flush=True)
    outs = llm.generate(flat, sp)
    m = len(BATTERY_V1)
    PY = np.zeros((len(units), m), np.float32)
    PN = np.zeros((len(units), m), np.float32)
    tops = []
    for i, o in enumerate(outs):
        py, pn, top = yes_no(o.outputs[0].logprobs[0])
        PY[i // m, i % m], PN[i // m, i % m] = py, pn
        tops.append(top)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / (f"{cfg.actor}_{cfg.tag}" + (f"_reader-{cfg.reader}" if cfg.reader else ""))
    units = units.assign(n_ctx=lens)
    units.to_parquet(f"{stem}_units.parquet")
    np.savez_compressed(f"{stem}.npz", p_yes=PY, p_no=PN, top=np.array(tops, dtype=object).reshape(len(units), m),
                        items=np.array([k for k, _ in BATTERY_V1]), battery_sha=battery_sha())
    pd.Series(turn).to_frame("turn_prompt").to_parquet(f"{stem}_turn_prompts.parquet")
    cov = (PY + PN)
    print(f"done {time.time()-t0:.0f}s; yes+no mass median {np.median(cov):.3f}, <0.5 in {(cov < 0.5).mean():.3%}",
          flush=True)
    print("top tokens:", pd.Series(tops).value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
