#!/usr/bin/env python
"""Exploratory (not preregistered): T6's monitoring comparison on the two architectures that T17 added for replication,
Qwen3.8-27B and Muse-Glimmer-30B, so the paper's monitoring table covers all four open agents.

Units: each agent's held-out-scenario actions of T17 (the split on which T17's internals probe is scored).
Monitors, as in T6 and T9:
  READER    the reader probe of T6 (Qwen3-8B layer 32, fit on Gemma-4's clean units only), on T17's reader features
  TXT       T9's TF-IDF (fit on Gemma-4 dev + fresh steps), on the same transcripts
  J_act     the zero-shot 70B judge of T3/T6 (Llama-3.3-70B, fp8, P(yes)), on the same transcripts
  INTERNALS the probe trained on the agent's OWN decision states and labels (T17, development scenarios)
Within-template AUROC; cluster bootstrap over scenarios (1,000) for reader - judge and reader - internals.
  stage=judge   (GPU, vLLM)   stage=analyze (CPU)
python scripts/127_more_agents_monitor_exploratory.py stage=judge|analyze
"""
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

t17 = importlib.import_module("96_t17_replication_agents")
OUT = Path("/data/models/graybox_ws/outputs/t17")
RES = ROOT / "results/t1"
AGENTS = ("qwen38", "museglimmer")
D = OmegaConf.create({"stage": "analyze", "n_boot": 1000})


def heldout(actor):
    U = pd.read_parquet(t17.units_path(actor))
    ok = np.load(OUT / f"{actor}_decision_hidden.npz")["ok"].astype(bool)
    f = ok & ~t17.is_dev(U.scenario_id.values)
    return U, f


def stage_judge(cfg):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    t3 = importlib.import_module("68_t3_deployable")
    p = snapshot_download("meta-llama/Llama-3.3-70B-Instruct", cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(p)
    llm = LLM(model=p, quantization="fp8", max_model_len=12288, gpu_memory_utilization=0.92, max_logprobs=20, enable_prefix_caching=True)

    def pyes(lp):
        y = sum(math.exp(t.logprob) for t in lp.values() if (t.decoded_token or "").strip().lower() == "yes")
        n = sum(math.exp(t.logprob) for t in lp.values() if (t.decoded_token or "").strip().lower() == "no")
        return y / max(y + n, 1e-9)
    for actor in AGENTS:
        U, f = heldout(actor)
        T = t17.texts(actor)[f]
        ps = [tok.apply_chat_template([{"role": "user", "content": t3.JUDGE.format(ctx=c, act=a)}], tokenize=False, add_generation_prompt=True)
              for c, a in zip(T.ctx, T.act)]
        outs = llm.generate(ps, SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, seed=0))
        pd.DataFrame({"p_unsafe": [pyes(o.outputs[0].logprobs[0]) for o in outs]}).to_parquet(OUT / f"{actor}_judge_heldout.parquet")
        print(actor, "judge scored", int(f.sum()), flush=True)


def stage_analyze(cfg):
    t6 = importlib.import_module("71_t6_cross_agent")
    t9 = importlib.import_module("77_t9_text_baseline")
    I, T4 = t6.I, t6.T4
    Ud = pd.read_parquet(I / "gemma4_clean_units.parquet")
    Hd = np.load(T4 / "clean_act.npz")
    Xd = Hd["h"][:, [int(x) for x in Hd["layers"]].index(32)].astype(np.float32)
    reader = t6.fit(Xd, Ud.y.values.astype(int))                       # T6's reader probe, Gemma-4 clean units only
    Uf = pd.read_parquet(I / "gemma4_fresh_units.parquet")
    Td, Tf = t9.gemma_texts("clean"), t9.gemma_texts("fresh")
    txt = t9.TXT().fit(pd.concat([Td.ctx, Tf.ctx]).values, pd.concat([Td.act, Tf.act]).values,
                       np.concatenate([Ud.y.values, Uf.y.values]).astype(int), np.concatenate([Ud.scenario_id.values, Uf.scenario_id.values]))
    res = {}
    for actor in AGENTS:
        U, f = heldout(actor)
        y, st, g = U.y.values[f].astype(int), t17.stratum(U.template.values[f]), U.scenario_id.values[f]
        T = t17.texts(actor)[f]
        z = np.load(ROOT / "results/units" / f"t17_{actor}_replication.npz")
        assert len(z["y"]) == f.sum() and (z["y"].astype(int) == y).all()
        S = {"READER": reader.predict_proba(t17.reader_features(actor)[f])[:, 1],
             "TXT": np.asarray(txt.score(T.ctx.values, T.act.values), float),
             "J_act": pd.read_parquet(OUT / f"{actor}_judge_heldout.parquet").p_unsafe.values,
             "INTERNALS": z["probe_true"].astype(float)}
        r = {"n": int(len(y)), "n_unsafe": int(y.sum()), "auroc_within_template": {k: round(within_auroc(y, v, st), 4) for k, v in S.items()}}
        for k, seed in (("J_act", 1271), ("INTERNALS", 1272)):
            d = cluster_boot(lambda i, k=k: within_auroc(y[i], S["READER"][i], st[i]) - within_auroc(y[i], S[k][i], st[i]), g, cfg.n_boot, seed=seed)
            r[f"READER_minus_{k}"] = [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
        res[actor] = r
    (RES / "more_agents_monitor_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"judge": stage_judge, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
