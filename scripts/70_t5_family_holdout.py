#!/usr/bin/env python
"""T5: agent internals vs a transcript reader, both trained on the same labels, on UNSEEN scenario families
(docs/prereg_T5_family.md).

Features at the decision point (the agent's proposed action):
  AGENT   Gemma-4's own residual stream at the decision point (context + its reasoning + "<channel|>"), layers 14/18/21/24/26
  READER  Qwen3-8B's residual stream reading the transcript + the proposed action (T4's READER_act features, 5 layers)
Identical probe for both: per layer, standardize -> PCA(256) -> logistic (C=0.05); the score is the mean of the 5
per-layer logits (CAPE-style depth ensemble; no layer selection).
Regimes:
  family  5 folds over the 23 scenario families (sorted, round-robin): train on development units of 4/5 of the
          families, score the fresh units of the held-out families; pooled over folds
  same    train on all development units, score all fresh units (the regime of T1-T4 and CAPE's gate)
  stage=hidden   agent decision-point states for the development units (HF)
  stage=analyze
python scripts/70_t5_family_holdout.py stage=hidden|analyze
"""
import importlib
import json
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
RES = ROOT / "results/t1"
D = OmegaConf.create({"stage": "analyze", "n_boot": 1000})
CAPE_LAYERS = [14, 18, 21, 24, 26]


def stage_hidden(cfg):
    import torch
    from cape.models.load import discover_transformer_blocks, model_path
    from cape.utils.config import resolve_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    a60 = importlib.import_module("60_decision_point")
    torch.backends.cuda.enable_cudnn_sdp(False)
    U = pd.read_parquet(I / "gemma4_clean_units.parquet")
    P = pd.read_parquet(I / "gemma4_clean_turn_prompts.parquet").turn_prompt.values
    ctx, _ = a60.contexts(U, P)
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
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]).cuda(), logits_to_keep=1, use_cache=False)
        X[i] = np.stack([cap[l] for l in CAPE_LAYERS]).astype(np.float16)
        ok[i] = True
        if i % 1000 == 0:
            print(f"{i}/{len(ctx)} {time.time()-t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    np.savez_compressed(I / "gemma4_clean_decision_hidden.npz", h=X, ok=ok, layers=np.array(CAPE_LAYERS))
    print(f"saved ok {ok.mean():.4f}", flush=True)


def probe_scores(Xtr, ytr, Xte):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = []
    for j in range(Xtr.shape[1]):
        m = make_pipeline(StandardScaler(), PCA(min(256, len(ytr) - 1), random_state=0), LogisticRegression(C=0.05, max_iter=4000))
        m.fit(Xtr[:, j].astype(np.float32), ytr)
        out.append(m.decision_function(Xte[:, j].astype(np.float32)))
    return np.mean(out, 0)


def stage_analyze(cfg):
    Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
    yd, yf = Ud.y.values.astype(int), Uf.y.values.astype(int)
    fd, ff = Ud.scenario_id.str.split(".").str[1].values, Uf.scenario_id.str.split(".").str[1].values
    Ad, Af = np.load(I / "gemma4_clean_decision_hidden.npz"), np.load(I / "gemma4_fresh_decision_hidden.npz")
    feats = {"AGENT": (Ad["h"], Af["h"], Ad["ok"], np.isfinite(Af["I_dec"])),
             "READER": (np.load(T4 / "clean_act.npz")["h"], np.load(T4 / "fresh_act.npz")["h"], np.ones(len(yd), bool), np.ones(len(yf), bool))}
    fams = sorted(set(fd) | set(ff))
    fold = {f: k % 5 for k, f in enumerate(fams)}
    S = {}
    for name, (Xd, Xf, okd, okf) in feats.items():
        s_fam = np.full(len(yf), np.nan)
        for k in range(5):
            tr = okd & np.array([fold[f] != k for f in fd])
            te = okf & np.array([fold[f] == k for f in ff])
            s_fam[te] = probe_scores(Xd[tr], yd[tr], Xf[te])
        S[f"{name}_family"] = s_fam
        s_same = np.full(len(yf), np.nan)
        s_same[okf] = probe_scores(Xd[okd], yd[okd], Xf[okf])
        S[f"{name}_same"] = s_same
        print(name, "done", flush=True)
    t3 = importlib.import_module("68_t3_deployable")
    idx, _ = t3.subset()
    J = pd.read_parquet(t3.OUT / "judge_K8.parquet").set_index("i")
    ja = np.full(len(yf), np.nan)
    ja[idx] = [json.loads(J.judge[i]).get("actual", np.nan) for i in idx]
    S["J_act"] = ja
    S["I_dec_frozen"] = np.load(I / "gemma4_fresh_decision_hidden.npz")["I_dec"]
    st, g = Uf.template.values, Uf.scenario_id.values
    ok = np.all([np.isfinite(S[k]) for k in S if k != "J_act"], 0)
    res = {"n": int(ok.sum()), "families": len(fams), "auroc": {}}
    for k, v in S.items():
        m = ok & np.isfinite(v)
        res["auroc"][k] = round(within_auroc(yf[m], v[m], st[m]), 4)

    def diff(k1, k2, mask, seed):
        gi = np.flatnonzero(mask)
        d = cluster_boot(lambda i: within_auroc(yf[gi[i]], S[k1][gi[i]], st[gi[i]]) - within_auroc(yf[gi[i]], S[k2][gi[i]], st[gi[i]]), g[mask], cfg.n_boot, seed=seed)
        return [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    sub = ok & np.isfinite(ja)
    res["P1_READER_minus_AGENT_family"] = diff("READER_family", "AGENT_family", ok, 51)
    res["P1_pass"] = res["P1_READER_minus_AGENT_family"][1] > -0.02
    res["P2_AGENT_family_minus_Jact"] = diff("AGENT_family", "J_act", sub, 52)
    res["P2_reading"] = ("agent probe better" if res["P2_AGENT_family_minus_Jact"][1] > 0 else
                         "judge better" if res["P2_AGENT_family_minus_Jact"][2] < 0 else "no significant difference")
    res["reported_READER_minus_AGENT_same"] = diff("READER_same", "AGENT_same", ok, 53)
    res["reported_READER_family_minus_Jact"] = diff("READER_family", "J_act", sub, 54)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "t5_family.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"hidden": stage_hidden, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
