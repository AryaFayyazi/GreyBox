#!/usr/bin/env python
"""When does reading an agent's chain of thought help a monitor? Corollary 1's boundary, tested.
Corollary 1: for a label decided by the action and visible facts, the agent's internals (and its hidden reasoning, a
function of them) add nothing to the transcript + action. For a label NOT decided by the current action, such as a
violation LATER in the episode, the reasoning can add information (it may state plans).
Monitors: the shipped reader (Qwen3-8B, layer 32) on the transcript + proposed action ("act"), or on the transcript +
the agent's reasoning for that turn + the action ("cot"; T4's rendering, reasoning cut to its last 4,000 characters);
probe = the shipped recipe (standardise, PCA 256, logistic C 0.05).
Labels:
  action   y: this action violates the policy (all decision steps)
  future   y_future: a LATER tool call in the same episode violates (only steps whose own action is safe)
  stage=labels                    (CPU) future labels for every unit of the four open agents
  stage=features agent=A split=S  (GPU) reader features with reasoning for T17's agents (Gemma-4's exist from T4)
  stage=dryrun                    (CPU) Gemma-4 development units, grouped 5-fold CV by scenario
  stage=dryrun_agents             (CPU) T17 agents' development units (grouped CV, and transfer from Gemma-4)
  stage=analyze                   (CPU) the preregistered test G8 (docs/prereg_G8_cot_boundary.md), held-out scenarios
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import within_auroc  # noqa: E402

I = Path("/data/models/graybox_ws/outputs/interrogate")
T4 = Path("/data/models/graybox_ws/outputs/t4")
OUT = Path("/data/models/graybox_ws/outputs/cot_boundary")
RES = ROOT / "results/t1"
D = OmegaConf.create({"stage": "dryrun", "agent": "gptoss20b", "split": "held", "n_boot": 1000})


def future_labels(U):
    from graybox import cape_bridge as cb
    fut = []
    for r in U.itertuples():
        evs = list(cb.rollouts(r.set).loc[r.episode_id].events)
        fut.append(int(any(bool(e.get("unsafe_action")) for e in evs if int(e["event_id"]) > int(r.event_id) and e.get("is_tool_call"))))
    return np.array(fut, np.int8)


def stage_labels(cfg):
    t17 = importlib.import_module("96_t17_replication_agents")
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in ("clean", "fresh"):
        U = pd.read_parquet(I / f"gemma4_{tag}_units.parquet")
        np.save(OUT / f"future_gemma4_{tag}.npy", future_labels(U))
    for a in t17.AGENTS:
        np.save(OUT / f"future_{a}.npy", future_labels(pd.read_parquet(t17.units_path(a))))
    print("labels saved", flush=True)


def stage_features(cfg):
    """Reader (Qwen3-8B, layer 32, T4's rendering) on transcript + reasoning + action, for one T17 agent and split."""
    import re
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from graybox import cape_bridge as cb
    t4 = importlib.import_module("69_t4_reader_action")
    t17 = importlib.import_module("96_t17_replication_agents")
    TAG = re.compile(r"<\|[^|<>]{0,40}\|>")
    U = pd.read_parquet(t17.units_path(cfg.agent))
    T = t17.texts(cfg.agent)
    dev = t17.is_dev(U.scenario_id.values)
    rows = np.where(dev if cfg.split == "dev" else ~dev)[0]                 # held-out features only after the lock
    p = snapshot_download("Qwen/Qwen3-8B", cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda").eval()
    H = np.zeros((len(rows), model.config.hidden_size), np.float16)
    for k, i in enumerate(rows):
        r = U.iloc[i]
        ev = {int(e["event_id"]): e for e in cb.rollouts(r.set).loc[r.episode_id].events}[int(r.event_id)]
        cot = TAG.sub("", ev.get("reasoning_text") or "")[-4000:]
        pr = t4.Q.format(ctx=T.ctx.values[i], cot=f"<agent_reasoning>\n{cot}\n</agent_reasoning>\n\n", act=T.act.values[i])
        s_ = tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(s_, add_special_tokens=False, return_tensors="pt")["input_ids"][:, -12000:].cuda()
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        H[k] = hs[32][0, -1].float().cpu().numpy().astype(np.float16)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{cfg.agent}_cot_{cfg.split}.npz", h=H, rows=rows)
    print(cfg.agent, cfg.split, H.shape, flush=True)


def probe():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=4000))


def gemma_feats(tag, var):
    H = np.load(T4 / f"{tag}_{var}.npz")
    return H["h"][:, [int(x) for x in H["layers"]].index(32)].astype(np.float32)


def stage_dryrun(cfg):
    from sklearn.model_selection import GroupKFold
    U = pd.read_parquet(I / "gemma4_clean_units.parquet")
    y, g, st = U.y.values.astype(int), U.scenario_id.values, U.template.values
    fut = np.load(OUT / "future_gemma4_clean.npy").astype(int)
    X = {v: gemma_feats("clean", v) for v in ("act", "cot")}
    res = {}
    for lab, yy, mask in (("action", y, np.ones(len(y), bool)), ("future", fut, y == 0)):
        idx = np.where(mask)[0]
        oof = {v: np.zeros(len(idx)) for v in X}
        for tr, te in GroupKFold(5).split(idx, yy[idx], g[idx]):
            for v in X:
                oof[v][te] = probe().fit(X[v][idx[tr]], yy[idx[tr]]).predict_proba(X[v][idx[te]])[:, 1]
        a = {v: within_auroc(yy[idx], oof[v], st[idx]) for v in X}
        res[lab] = {"n": int(len(idx)), "positives": int(yy[idx].sum()), "auroc_act": round(a["act"], 4), "auroc_cot": round(a["cot"], 4),
                    "cot_minus_act": round(a["cot"] - a["act"], 4)}
        print(lab, res[lab], flush=True)
    (RES / "cot_boundary_DRYRUN_dev.json").write_text(json.dumps(res, indent=1))


def stage_dryrun_agents(cfg):
    """T17 agents' DEVELOPMENT scenarios: (i) within-agent grouped 5-fold CV (what the reasoning contains), act vs cot,
    for both labels; (ii) transfer: probes trained on Gemma-4's clean units, applied to the agent."""
    from sklearn.model_selection import GroupKFold
    t17 = importlib.import_module("96_t17_replication_agents")
    Ug = pd.read_parquet(I / "gemma4_clean_units.parquet")
    yg, futg = Ug.y.values.astype(int), np.load(OUT / "future_gemma4_clean.npy").astype(int)
    Xg = {v: gemma_feats("clean", v) for v in ("act", "cot")}
    gem = {("action", v): probe().fit(Xg[v], yg) for v in Xg}
    gem.update({("future", v): probe().fit(Xg[v][yg == 0], futg[yg == 0]) for v in Xg})
    res = {}
    for a in t17.AGENTS:
        U = pd.read_parquet(t17.units_path(a))
        z = np.load(OUT / f"{a}_cot_dev.npz")
        rows = z["rows"]
        X = {"act": t17.reader_features(a)[rows].astype(np.float32), "cot": z["h"].astype(np.float32)}
        y, fut = U.y.values[rows].astype(int), np.load(OUT / f"future_{a}.npy")[rows].astype(int)
        g, st = U.scenario_id.values[rows], t17.stratum(U.template.values[rows])
        res[a] = {}
        for lab, yy, mask in (("action", y, np.ones(len(y), bool)), ("future", fut, y == 0)):
            idx = np.where(mask)[0]
            oof = {v: np.zeros(len(idx)) for v in X}
            for tr, te in GroupKFold(5).split(idx, yy[idx], g[idx]):
                for v in X:
                    oof[v][te] = probe().fit(X[v][idx[tr]], yy[idx[tr]]).predict_proba(X[v][idx[te]])[:, 1]
            wi = {v: within_auroc(yy[idx], oof[v], st[idx]) for v in X}
            tf = {v: within_auroc(yy[idx], gem[(lab, v)].predict_proba(X[v][idx])[:, 1], st[idx]) for v in X}
            res[a][lab] = {"n": int(len(idx)), "positives": int(yy[idx].sum()),
                           "within_act": round(wi["act"], 4), "within_cot": round(wi["cot"], 4), "within_cot_minus_act": round(wi["cot"] - wi["act"], 4),
                           "transfer_act": round(tf["act"], 4), "transfer_cot": round(tf["cot"], 4), "transfer_cot_minus_act": round(tf["cot"] - tf["act"], 4)}
            print(a, lab, res[a][lab], flush=True)
    (RES / "cot_boundary_DRYRUN_agents_dev.json").write_text(json.dumps(res, indent=1))


def stage_analyze(cfg):
    """G8 on T17's HELD-OUT scenarios. Two monitors per label, each with and without the reasoning:
    in-agent  probe trained on the agent's own development units, scored on its held-out units
    transfer  probe trained on Gemma-4's clean units only (the deployable case: no labels from the agent)
    Bootstrap: held-out scenarios resampled within each agent (independently), 1,000 times, seed 1490; pooled = mean of
    the three agents' differences."""
    t17 = importlib.import_module("96_t17_replication_agents")
    Ug = pd.read_parquet(I / "gemma4_clean_units.parquet")
    yg, futg = Ug.y.values.astype(int), np.load(OUT / "future_gemma4_clean.npy").astype(int)
    Xg = {v: gemma_feats("clean", v) for v in ("act", "cot")}
    gem = {("action", v): probe().fit(Xg[v], yg) for v in Xg}
    gem.update({("future", v): probe().fit(Xg[v][yg == 0], futg[yg == 0]) for v in Xg})
    rng = np.random.default_rng(1490)
    res, boots = {"agents": {}}, {}
    for a in t17.AGENTS:
        U = pd.read_parquet(t17.units_path(a))
        zd, zh = np.load(OUT / f"{a}_cot_dev.npz"), np.load(OUT / f"{a}_cot_held.npz")
        rd, rh = zd["rows"], zh["rows"]
        R = t17.reader_features(a).astype(np.float32)
        Xd, Xh = {"act": R[rd], "cot": zd["h"].astype(np.float32)}, {"act": R[rh], "cot": zh["h"].astype(np.float32)}
        y, fut = U.y.values.astype(int), np.load(OUT / f"future_{a}.npy").astype(int)
        st = t17.stratum(U.template.values)
        res["agents"][a] = {}
        for lab in ("action", "future"):
            yy = y if lab == "action" else fut
            md = np.ones(len(rd), bool) if lab == "action" else y[rd] == 0
            mh = np.ones(len(rh), bool) if lab == "action" else y[rh] == 0
            h = rh[mh]
            sc = {}
            for v in ("act", "cot"):
                sc[("inagent", v)] = probe().fit(Xd[v][md], yy[rd][md]).predict_proba(Xh[v][mh])[:, 1]
                sc[("transfer", v)] = gem[(lab, v)].predict_proba(Xh[v][mh])[:, 1]
            yh, sh, gh = yy[h], st[h], U.scenario_id.values[h]
            out = {"n": int(len(h)), "positives": int(yh.sum())}
            scen = np.unique(gh)
            members = {s_: np.where(gh == s_)[0] for s_ in scen}
            draws = [np.concatenate([members[s_] for s_ in rng.choice(scen, len(scen))]) for _ in range(cfg.n_boot)]
            for m in ("inagent", "transfer"):
                aa = {v: within_auroc(yh, sc[(m, v)], sh) for v in ("act", "cot")}
                bd = np.array([within_auroc(yh[d], sc[(m, "cot")][d], sh[d]) - within_auroc(yh[d], sc[(m, "act")][d], sh[d]) for d in draws])
                boots[(a, lab, m)] = bd
                out[m] = {"act": round(float(aa["act"]), 4), "cot": round(float(aa["cot"]), 4), "cot_minus_act": round(float(aa["cot"] - aa["act"]), 4),
                          "ci": [round(float(np.nanpercentile(bd, 2.5)), 4), round(float(np.nanpercentile(bd, 97.5)), 4)]}
            res["agents"][a][lab] = out
            print(a, lab, json.dumps(out), flush=True)
    pooled = {}
    for lab in ("action", "future"):
        for m in ("inagent", "transfer"):
            pt = float(np.mean([res["agents"][a][lab][m]["cot_minus_act"] for a in t17.AGENTS]))
            bd = np.nanmean(np.vstack([boots[(a, lab, m)] for a in t17.AGENTS]), axis=0)
            pooled[f"{lab}_{m}"] = [round(pt, 4), round(float(np.nanpercentile(bd, 2.5)), 4), round(float(np.nanpercentile(bd, 97.5)), 4)]
    res["pooled_cot_minus_act"] = pooled
    E = {"C1_inagent_action_no_gain": pooled["action_inagent"][2] < 0.01,
         "C2_transfer_action_reasoning_hurts": pooled["action_transfer"][2] < 0,
         "C3_transfer_action_every_agent": all(res["agents"][a]["action"]["transfer"]["cot_minus_act"] < 0 for a in t17.AGENTS)}
    f = pooled["future_inagent"]
    E["C4_future_reading"] = "reasoning_helps" if f[1] > 0 else ("equivalent" if -0.05 < f[1] and f[2] < 0.05 else "inconclusive")
    res["endpoints"] = E
    print(json.dumps({"pooled": pooled, "endpoints": E}, indent=1), flush=True)
    (RES / "g8_cot_boundary.json").write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"labels": stage_labels, "features": stage_features, "dryrun": stage_dryrun, "dryrun_agents": stage_dryrun_agents, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
