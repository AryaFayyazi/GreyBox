#!/usr/bin/env python
"""When do internals beat the black box? The same 12,130 fresh CAPE units read at two moments (docs/prereg_T2_time.md).

Turn start (the last point a black box can be forked before the turn; already measured in the T1 confirmation):
  R16_ts    16 API resamples of the whole turn                (black box)
  WB_ts     frozen T1 probe on the turn-start residual        (white box)
  TXT_ts    frozen T1 text monitor on the transcript
Decision point (after the agent's own reasoning for this turn, the last token before its tool call; CAPE's point):
  I_dec     CAPE's frozen gate (outputs/v54/gate_gemma4.npz, layers 14/18/21/24/26), applied unchanged (white box)
  R16_dec   16 resamples of ONLY the action, given the reasoning the agent actually wrote (black box: needs the
            reasoning text, which some APIs return)
  COT_dec   TF-IDF on transcript + this turn's reasoning, out of fold by scenario (a supervised CoT text monitor)
Stages: hidden (HF, GPU), roll (vLLM, GPU), analyze (CPU).
python scripts/60_decision_point.py stage=hidden|roll|analyze
"""
import copy
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CAPE = Path("/home/arya/new_safety/cape")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(CAPE / "src"))
sys.path.insert(0, str(CAPE / "scripts"))
from graybox import cape_bridge as cb  # noqa: E402
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

I = Path("/data/models/graybox_ws/outputs/interrogate")
GATE = Path("/data/models/cape_ws/outputs/v54/gate_gemma4.npz")
RES = ROOT / "results/t1"
TAG = re.compile(r"<\|[^|<>]{0,40}\|>")
D = OmegaConf.create({"stage": "analyze", "K": 16, "limit": 0, "n_boot": 1000})
CAPE_LAYERS = [14, 18, 21, 24, 26]


def units():
    U = pd.read_parquet(I / "gemma4_fresh_units.parquet")
    P = pd.read_parquet(I / "gemma4_fresh_turn_prompts.parquet").turn_prompt.values
    return U, P


def reasoning_of(U):
    out = []
    for s, e, t in zip(U.set, U.episode_id, U.event_id):
        ev = {int(x["event_id"]): x for x in cb.rollouts(s).loc[e].events}[int(t)]
        out.append(TAG.sub("", ev["reasoning_text"] or ""))
    return out


def contexts(U, P):
    R = reasoning_of(U)
    return [p + r + "\n<channel|>" if r.strip() else None for p, r in zip(P, R)], R


def stage_hidden(cfg):
    import torch
    from cape.models.load import discover_transformer_blocks, model_path
    from cape.utils.config import resolve_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    torch.backends.cuda.enable_cudnn_sdp(False)
    U, P = units()
    ctx, _ = contexts(U, P)
    if cfg.limit:
        ctx = ctx[: int(cfg.limit)]
    path = model_path(resolve_model("google/gemma-4-26B-A4B-it"))
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    blocks, _ = discover_transformer_blocks(model)
    cap = {}

    def hook_for(l):
        def fn(mod, inp, outp):
            h = outp[0] if isinstance(outp, tuple) else outp
            cap[l] = h[0, -1].float().cpu().numpy()
        return fn
    hooks = [blocks[l].register_forward_hook(hook_for(l)) for l in CAPE_LAYERS]
    X = np.zeros((len(ctx), len(CAPE_LAYERS), 2816), np.float16)
    ok = np.zeros(len(ctx), bool)
    t0 = time.time()
    for i, c in enumerate(ctx):
        if c is None:
            continue
        ids = tok(c, add_special_tokens=False)["input_ids"]
        assert ids[-1] == 101, "the decision point must be the <channel|> token"
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]).cuda(), logits_to_keep=1, use_cache=False)
        X[i] = np.stack([cap[l] for l in CAPE_LAYERS]).astype(np.float16)
        ok[i] = True
        if i % 500 == 0:
            print(f"{i}/{len(ctx)} {time.time()-t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    G = np.load(GATE)
    g154 = importlib.import_module("154_v54_gate")
    I_dec = np.where(ok, g154.gate_score(G, X.astype(np.float32)), np.nan)
    np.savez_compressed(I / "gemma4_fresh_decision_hidden.npz", h=X, ok=ok, I_dec=I_dec, layers=np.array(CAPE_LAYERS))
    print(f"saved; ok {ok.mean():.4f}; {time.time()-t0:.0f}s", flush=True)


def stage_roll(cfg):
    from cape.models.load import model_path
    from cape.utils.config import resolve_model
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    r04 = importlib.import_module("04_rollouts")
    s144 = importlib.import_module("144_v54_steer")
    U, P = units()
    ctx, _ = contexts(U, P)
    if cfg.limit:
        U, ctx = U.head(int(cfg.limit)), ctx[: int(cfg.limit)]
    path = model_path(resolve_model("google/gemma-4-26B-A4B-it"))
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(model=path, max_model_len=36000, gpu_memory_utilization=0.88, enable_prefix_caching=True, language_model_only=True)
    sp = SamplingParams(n=cfg.K, temperature=1.0, top_p=0.95, top_k=64, max_tokens=1024, seed=0,
                        stop_token_ids=[1, 106, 50], skip_special_tokens=False)
    idx = [i for i, c in enumerate(ctx) if c is not None]
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": tok(ctx[i], add_special_tokens=False)["input_ids"]} for i in idx], sp)
    print(f"generated {len(idx)} x {cfg.K} in {time.time()-t0:.0f}s", flush=True)
    a = r04.srv.ACTORS["gemma4"]
    rows = [{"unsafe": [], "kind": []} for _ in range(len(ctx))]
    for i, o in zip(idx, outs):
        r = U.iloc[i]
        ep = cb.rollouts(r.set).loc[r.episode_id]
        spec = cb.specs(r.bench)[ep.instance_id]
        _, env = r04.openai_msgs(a, spec, list(ep.events), int(r.event_id))
        for c in o.outputs:
            tool, args = s144.parse_call("gemma4", c.text)
            if tool == "__final__":
                lab = {"kind": "final", "unsafe": 0}
            elif tool is None:
                lab = {"kind": "parse_error", "unsafe": 0}
            else:
                choice = {"finish_reason": c.finish_reason, "message": {"tool_calls": [
                    {"function": {"name": tool, "arguments": json.dumps(args)}}]}}
                lab = r04.label(env, choice)
            rows[i]["unsafe"].append(int(lab["unsafe"]))
            rows[i]["kind"].append(lab["kind"])
    out = U[["set", "episode_id", "event_id", "cls", "y"]].copy()
    out["R_unsafe"] = [json.dumps(x["unsafe"]) for x in rows]
    out["R_kind"] = [json.dumps(x["kind"]) for x in rows]
    out["R"] = [np.mean(x["unsafe"]) if x["unsafe"] else np.nan for x in rows]
    out.to_parquet(I / f"gemma4_fresh_decision_rollouts_K{cfg.K}.parquet")
    print(out.groupby("cls").R.describe().to_string(), flush=True)


def stage_analyze(cfg):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    a25 = importlib.import_module("25_t1_confirm")
    import pickle
    U, P = units()
    y, g, st = U.y.values.astype(int), U.scenario_id.values, U.template.values
    stem = I / "gemma4_fresh"
    with open(a25.MOD / "frozen_monitors.pkl", "rb") as f:
        M = pickle.load(f)
    Rts = pd.read_parquet(I / "gemma4_fresh_rollouts_K16.parquet")
    k = np.array([np.sum(json.loads(s)) for s in Rts.R_unsafe]); n = np.array([len(json.loads(s)) for s in Rts.R_unsafe])
    Hd = np.load(I / "gemma4_fresh_decision_hidden.npz")
    Rd = pd.read_parquet(I / f"gemma4_fresh_decision_rollouts_K{cfg.K}.parquet")
    kd = np.array([np.sum(json.loads(s)) for s in Rd.R_unsafe]); nd = np.array([len(json.loads(s)) for s in Rd.R_unsafe])
    S = {"R16_ts": (k + 0.5) / (n + 1), "WB_ts": M["wb"].predict_proba(a25.hidden(stem, M["wb_layer"]))[:, 1],
         "TXT_ts": M["txt"].predict_proba(a25.texts(stem))[:, 1], "I_dec": Hd["I_dec"], "R16_dec": (kd + 0.5) / (nd + 1)}
    _, R = contexts(U, P)
    docs = [t + "\n" + r for t, r in zip(a25.texts(stem), R)]
    cot = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(docs, y, g):
        v = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=200_000, sublinear_tf=True)
        clf = LogisticRegression(C=4.0, max_iter=4000).fit(v.fit_transform([docs[i] for i in tr]), y[tr])
        cot[te] = clf.predict_proba(v.transform([docs[i] for i in te]))[:, 1]
    S["COT_dec_oof"] = cot
    ok = np.isfinite(S["I_dec"]) & np.isfinite(S["R16_dec"]) & (nd > 0)
    res = {"n": int(ok.sum()), "n_dropped_no_reasoning_or_rollout": int((~ok).sum()), "auroc": {}}
    for name, s in S.items():
        res["auroc"][name] = round(within_auroc(y[ok], s[ok], st[ok]), 4)
    a = lambda key, i: within_auroc(y[i], S[key][i], st[i])  # noqa: E731
    gi = np.flatnonzero(ok)

    def diff(k1, k2, seed):
        d = cluster_boot(lambda i: a(k1, gi[i]) - a(k2, gi[i]), g[ok], cfg.n_boot, seed=seed)
        return [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    res["G1_Idec_minus_R16ts"] = diff("I_dec", "R16_ts", 21)
    res["G2_R16dec_minus_Idec"] = diff("R16_dec", "I_dec", 22)
    res["G3_Idec_minus_WBts"] = diff("I_dec", "WB_ts", 23)
    res["G1_pass"] = res["G1_Idec_minus_R16ts"][1] > 0
    res["G2_pass"] = res["G2_R16dec_minus_Idec"][1] > -0.02
    res["G3_pass"] = res["G3_Idec_minus_WBts"][1] > 0
    res["reported_COTdec_minus_Idec"] = diff("COT_dec_oof", "I_dec", 24)
    res["gate_fire_rate"] = {c: round(float((S["I_dec"][ok & (U.cls.values == c)] > float(np.load(GATE)["tau"])).mean()), 4)
                             for c in ("XUNSAFE", "XSAFE", "XPLUS")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "t2_time.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"hidden": stage_hidden, "roll": stage_roll, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
