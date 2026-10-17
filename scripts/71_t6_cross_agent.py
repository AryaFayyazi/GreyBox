#!/usr/bin/env python
"""T6: train the transcript reader on ONE agent, monitor ANOTHER (docs/prereg_T6_cross_agent.md).

The T4 reader probe (Qwen3-8B reading the transcript + proposed action; linear probe at the layer T4 chose on Gemma's
development units by grouped CV, L32) is trained on Gemma-4 development units only and applied unchanged to
gpt-oss-20b's clean units (another agent family; 5,229 units, 729 unsafe). Baseline: the zero-shot judge of T3
(Llama-3.3-70B, same prompt) on gpt-oss's actual actions.
  stage=hidden  reader states for the gpt-oss units (HF)         -> outputs/t6/gptoss_act.npz
  stage=judge   zero-shot judge on gpt-oss actual actions (vLLM) -> outputs/t6/gptoss_judge.parquet
  stage=analyze Q1 + reported: label-efficiency curve on Gemma (fresh test) and reader latency
python scripts/71_t6_cross_agent.py stage=hidden|judge|analyze
"""
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox import cape_bridge as cb  # noqa: E402
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

I = Path("/data/models/graybox_ws/outputs/interrogate")
T4 = Path("/data/models/graybox_ws/outputs/t4")
OUT = Path("/data/models/graybox_ws/outputs/t6")
RES = ROOT / "results/t1"
D = OmegaConf.create({"stage": "analyze", "n_boot": 1000, "layer": 32})


def gptoss_prompts():
    t3, t4 = importlib.import_module("68_t3_deployable"), importlib.import_module("69_t4_reader_action")
    U = pd.read_parquet(I / "gptoss20b_clean_units.parquet")
    ctxs, acts = [], []
    for r in U.itertuples():
        ep = cb.rollouts(r.set).loc[r.episode_id]
        evs = list(ep.events)
        ev = {int(x["event_id"]): x for x in evs}[int(r.event_id)]
        ctxs.append(t3.ctx_text(cb.specs(r.bench)[ep.instance_id], evs, int(r.event_id)))
        acts.append(t3.act_text(ev["tool"], json.loads(ev["args_json"] or "{}")))
    return U, ctxs, acts, t3, t4


def stage_hidden(cfg):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    U, ctxs, acts, _, t4 = gptoss_prompts()
    p = snapshot_download("Qwen/Qwen3-8B", cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda").eval()
    L = [int(x) for x in np.load(T4 / "clean_act.npz")["layers"]]
    H = np.zeros((len(U), len(L), model.config.hidden_size), np.float16)
    lat = []
    for i, (c, a) in enumerate(zip(ctxs, acts)):
        s = tok.apply_chat_template([{"role": "user", "content": t4.Q.format(ctx=c, cot="", act=a)}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][:, -12000:].cuda()
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        torch.cuda.synchronize(); lat.append(time.time() - t0)
        H[i] = np.stack([hs[l][0, -1].float().cpu().numpy() for l in L]).astype(np.float16)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "gptoss_act.npz", h=H, layers=np.array(L), latency_s=np.array(lat))
    print("saved", H.shape, "median latency", round(float(np.median(lat)), 4), flush=True)


def stage_judge(cfg):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    U, ctxs, acts, t3, _ = gptoss_prompts()
    p = snapshot_download("meta-llama/Llama-3.3-70B-Instruct", cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(p)
    llm = LLM(model=p, quantization="fp8", max_model_len=12288, gpu_memory_utilization=0.92, max_logprobs=20, enable_prefix_caching=True)
    ps = [tok.apply_chat_template([{"role": "user", "content": t3.JUDGE.format(ctx=c, act=a)}], tokenize=False, add_generation_prompt=True) for c, a in zip(ctxs, acts)]
    outs = llm.generate(ps, SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, seed=0))

    def pyes(lp):
        y = sum(math.exp(t.logprob) for t in lp.values() if (t.decoded_token or "").strip().lower() == "yes")
        n = sum(math.exp(t.logprob) for t in lp.values() if (t.decoded_token or "").strip().lower() == "no")
        return y / max(y + n, 1e-9)
    pd.DataFrame({"episode_id": U.episode_id, "event_id": U.event_id, "p_unsafe": [pyes(o.outputs[0].logprobs[0]) for o in outs]}).to_parquet(OUT / "gptoss_judge.parquet")
    print("saved judge", len(U), flush=True)


def fit(X, y):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(min(256, len(y) - 1), random_state=0), LogisticRegression(C=0.05, max_iter=4000)).fit(X, y)


def stage_analyze(cfg):
    Ud = pd.read_parquet(I / "gemma4_clean_units.parquet")
    Uf = pd.read_parquet(I / "gemma4_fresh_units.parquet")
    Uo = pd.read_parquet(I / "gptoss20b_clean_units.parquet")
    Hd, Hf, Ho = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz"), np.load(OUT / "gptoss_act.npz")
    j = [int(x) for x in Hd["layers"]].index(cfg.layer)
    Xd, Xf, Xo = (H["h"][:, j].astype(np.float32) for H in (Hd, Hf, Ho))
    yd, yf, yo = (U.y.values.astype(int) for U in (Ud, Uf, Uo))
    m = fit(Xd, yd)
    S = {"READER_from_gemma": m.predict_proba(Xo)[:, 1],
         "J_act": pd.read_parquet(OUT / "gptoss_judge.parquet").p_unsafe.values}
    st, g = Uo.template.values, Uo.scenario_id.values
    res = {"n": int(len(yo)), "n_unsafe": int(yo.sum()), "layer": cfg.layer,
           "auroc_gptoss": {k: round(within_auroc(yo, v, st), 4) for k, v in S.items()}}
    d = cluster_boot(lambda i: within_auroc(yo[i], S["READER_from_gemma"][i], st[i]) - within_auroc(yo[i], S["J_act"][i], st[i]), g, cfg.n_boot, seed=61)
    res["Q1_READER_minus_Jact_gptoss"] = [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    res["Q1_pass"] = res["Q1_READER_minus_Jact_gptoss"][1] > 0
    rng = np.random.default_rng(0)
    scen = Ud.scenario_id.values
    us = np.unique(scen)
    curve = {}
    for n_s in (10, 25, 50, 100, 200, len(us)):
        aucs = []
        for rep in range(5 if n_s < len(us) else 1):
            pick = rng.choice(us, n_s, replace=False) if n_s < len(us) else us
            tr = np.isin(scen, pick)
            if yd[tr].sum() < 3:
                continue
            aucs.append(within_auroc(yf, fit(Xd[tr], yd[tr]).predict_proba(Xf)[:, 1], Uf.template.values))
        curve[int(n_s)] = {"train_units_mean": int(np.mean([np.isin(scen, rng.choice(us, n_s, replace=False)).sum() for _ in range(3)])) if n_s < len(us) else int(len(yd)),
                           "fresh_gemma_auroc_mean": round(float(np.mean(aucs)), 4) if aucs else None, "reps": len(aucs)}
    res["reported_label_efficiency_by_training_scenarios"] = curve
    res["reported_reader_latency_s_median"] = round(float(np.median(Ho["latency_s"])), 4)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "t6_cross_agent.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"hidden": stage_hidden, "judge": stage_judge, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
