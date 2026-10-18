#!/usr/bin/env python
"""T17: does the grey box replicate the hidden state of other open agents? (docs/prereg_T17_replication_agents.md)

T15 (scripts/88) on three more open agents of different architectures, with a stricter split:
  gptoss20b    openai/gpt-oss-20b            (mixture of experts, 24 blocks)       units: gptoss20b_clean (T6)
  qwen38       Qwen/Qwen3.8-27B              (hybrid linear attention, 64 blocks)  units: built here
  museglimmer  meta-models/Muse-Glimmer-30B  (dense, sliding window, 52 blocks)    units: built here
Development and fresh steps come from DISJOINT scenarios: a scenario is in development if sha256(scenario_id) is even.
Target: the agent's hidden state at its decision point (the end-of-reasoning token just before the tool call) at 5
depths, standardized on development, top 128 PCs. Replicas: ridge maps from Qwen3-8B layer-32 features (READER) or
TF-IDF of context and action (TXT). E1: R^2 READER - TXT. E2: a probe on TRUE development states, applied to true vs
READER-reconstructed fresh states (within-template AUROC; non-inferiority -0.02).
Stages: units (CPU), hidden actor=A (GPU; Muse needs venv_tf520), reader actor=A (GPU), analyze (CPU).
python scripts/96_t17_replication_agents.py stage=units|hidden|reader|analyze [actor=A] [limit=N]
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "8")
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox import cape_bridge as cb  # noqa: E402
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

I = Path("/data/models/graybox_ws/outputs/interrogate")
OUT = Path("/data/models/graybox_ws/outputs/t17")
RES = ROOT / "results/t1"
D = OmegaConf.create({"stage": "analyze", "actor": None, "limit": 0, "n_boot": 1000})
AGENTS = ["gptoss20b", "qwen38", "museglimmer"]
MARK = {"gptoss20b": "<|end|>", "qwen38": "</think>", "museglimmer": "<|eom|>"}   # end of the agent's reasoning
# every core24b and core24b_reasoning set of the agent in CAPE (positives are scarce); negatives subsampled as in T1
SETS = {"qwen38": {"minus": ["core24b_minus", "core24b_minus2345", "core24b_reasoning_minus0123"], "plus": ["core24b_plus"],
                   "n_safe": 3000, "n_plus": 1500},
        "museglimmer": {"minus": [f"xm_museglimmer_{b}_minus{k}" for b, ks in (("core24b", ("", "23", "45", "67", "89", "1011", "1213", "1415")),
                                                                                 ("core24b_reasoning", ("", "23", "45", "67"))) for k in ks],
                        "plus": [], "n_safe": 4500, "n_plus": 0}}
NXT = 16                                              # tokens of the actual tool call used for the fidelity check
FRACS = (0.5, 0.62, 0.72, 0.82, 0.9)                 # the depth grid of T1/T15 (Gemma: 14, 18, 21, 24, 26)
ALPHAS = (1, 10, 100, 1000, 10000)
SEEDS = {"gptoss20b": 171, "qwen38": 181, "museglimmer": 191, "gemma4": 201}


def bench_of(set_name):
    """<prefix>_<actor>_<bench>_<minus|plus><seeds>, prefix optional (Qwen3.8 sets have none; scripts/15 assumed one)."""
    s = re.sub(r"^(?:xm|gb)_[a-z0-9]+_", "", set_name)
    b = re.match(r"^(.+?)_(?:minus|plus)", s).group(1)
    assert (cb.CAPE / f"outputs/capebench/{b}/scenarios.jsonl").exists(), (set_name, b)
    return b


def units_path(actor):
    return I / ("gptoss20b_clean_units.parquet" if actor == "gptoss20b" else f"{actor}_t17_units.parquet")


def is_dev(scenario_ids):
    return np.array([int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % 2 == 0 for s in scenario_ids])


def stratum(templates):
    return np.array([re.sub(r"<\|.*$", "", t) for t in templates])   # gpt-oss tool names carry harmony residue


def stage_units(cfg):
    s15 = importlib.import_module("15_clean_units")
    for actor, c in SETS.items():
        rows = [r for s in c["minus"] for r in s15.units_of(s, "minus", bench_of(s))] + \
               [r for s in c["plus"] for r in s15.units_of(s, "plus", bench_of(s))]
        U = pd.DataFrame(rows)
        rng = np.random.default_rng(0)
        keep = [U[U.cls == "XUNSAFE"]]
        for cls, n in (("XSAFE", c["n_safe"]), ("XPLUS", c["n_plus"])):
            g = U[U.cls == cls]
            if n:
                keep.append(g.iloc[rng.choice(len(g), min(n, len(g)), replace=False)])
        U = pd.concat(keep).sort_values(["set", "episode_id", "event_id"]).reset_index(drop=True)
        U["I"] = np.nan
        U.to_parquet(units_path(actor))
        print(actor, len(U), U.cls.value_counts().to_dict(), "scenarios", U.scenario_id.nunique(), flush=True)


def texts(actor):
    """Reader-side text of each unit (the transcript and the proposed action only): scripts/68 ctx_text / act_text."""
    p = OUT / f"{actor}_texts.parquet"
    if p.exists():
        return pd.read_parquet(p)
    t3 = importlib.import_module("68_t3_deployable")
    U = pd.read_parquet(units_path(actor))
    ctx, act = [], []
    for r in U.itertuples():
        ep = cb.rollouts(r.set).loc[r.episode_id]
        evs = list(ep.events)
        ev = {int(x["event_id"]): x for x in evs}[int(r.event_id)]
        ctx.append(t3.ctx_text(cb.specs(r.bench)[ep.instance_id], evs, int(r.event_id)))
        act.append(t3.act_text(ev["tool"], json.loads(ev["args_json"] or "{}")))
    T = pd.DataFrame({"ctx": ctx, "act": act})
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_parquet(p)
    return T


def decision_contexts(actor, tok, U):
    """The agent's own chat template on (turn-start context + this turn's reasoning + tool call), cut after the
    end-of-reasoning token. ok = reasoning present, the turn prompt is a prefix of the rendering, the marker follows."""
    a, mark = cb.ACT[actor], MARK[actor]
    out, nxt, why = [], [], {"no_reasoning": 0, "prefix_mismatch": 0, "no_marker": 0}
    for r in U.itertuples():
        ep = cb.rollouts(r.set).loc[r.episode_id]
        evs = list(ep.events)
        e = {int(x["event_id"]): x for x in evs}[int(r.event_id)]
        reason = cb.TAG.sub("", e["reasoning_text"] or "")
        if not reason.strip():
            out.append(None); nxt.append(None); why["no_reasoning"] += 1
            continue
        msgs = cb.turn_start_messages(cb.specs(r.bench)[ep.instance_id], evs, int(r.event_id), a["suffix"])
        tp = cb.render(tok, msgs, a["kwargs"], gen=True)
        full = cb.render(tok, msgs + [cb.assistant_msg(e)], a["kwargs"], gen=False)
        if not full.startswith(tp):
            out.append(None); nxt.append(None); why["prefix_mismatch"] += 1
            continue
        k = full.find(mark, len(tp))
        if k < 0:
            out.append(None); nxt.append(None); why["no_marker"] += 1
            continue
        out.append(full[: k + len(mark)])
        nxt.append(full[k + len(mark): k + len(mark) + 400])           # what the agent emitted next (its tool call)
    return out, nxt, why


def load_agent(path):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    multimodal = hasattr(AutoConfig.from_pretrained(path), "vision_config")
    order = (AutoModelForImageTextToText, AutoModelForCausalLM) if multimodal else (AutoModelForCausalLM, AutoModelForImageTextToText)
    err = None
    for cls in order:
        try:
            return cls.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        except Exception as ex:  # noqa: PERF203
            err = ex
    raise err


def stage_hidden(cfg):
    import torch
    from cape.models.load import discover_transformer_blocks, model_path, text_config
    from cape.utils.config import resolve_model
    from transformers import AutoTokenizer
    actor = cfg.actor
    torch.backends.cuda.enable_cudnn_sdp(False)
    U = pd.read_parquet(units_path(actor))
    if cfg.limit:
        U = U.head(int(cfg.limit))
    path = model_path(resolve_model(cb.ACT[actor]["model"]))
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    ctx, nxt, why = decision_contexts(actor, tok, U)
    print(f"{actor}: {len(U)} units; contexts in {time.time()-t0:.0f}s; excluded {why}", flush=True)
    mark_id = tok.convert_tokens_to_ids(MARK[actor])
    model = load_agent(path)
    blocks, _ = discover_transformer_blocks(model)
    n = len(blocks)
    layers = [round(n * f) - 1 for f in FRACS]
    dim = text_config(model).hidden_size
    cap, at = {}, [0]

    def hook_for(l):
        def fn(mod, inp, outp):
            h = outp[0] if isinstance(outp, tuple) else outp
            cap[l] = h[0, at[0]].float().cpu().numpy()      # the decision point, not the end of the appended call
        return fn
    hooks = [blocks[l].register_forward_hook(hook_for(l)) for l in layers]
    X = np.zeros((len(ctx), len(layers), dim), np.float16)
    ok = np.zeros(len(ctx), bool)
    ntok = np.zeros(len(ctx), np.int32)
    # fidelity: teacher-forced, does the agent's model predict the first NXT tokens it actually emitted (tool name, args)?
    acc_next = np.full(len(ctx), np.nan, np.float32)
    p_next = np.full(len(ctx), np.nan, np.float32)
    t0 = time.time()
    for i, c in enumerate(ctx):
        if c is None:
            continue
        ids = tok(c, add_special_tokens=False)["input_ids"]
        assert ids[-1] == mark_id, f"unit {i}: the decision point must be the {MARK[actor]} token"
        full = tok(c + nxt[i], add_special_tokens=False)["input_ids"]
        tail = full[len(ids): len(ids) + NXT] if full[: len(ids)] == ids else []
        at[0] = len(ids) - 1
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids + tail]).cuda(), logits_to_keep=len(tail) + 1, use_cache=False)
        X[i] = np.stack([cap[l] for l in layers]).astype(np.float16)
        ok[i], ntok[i] = True, len(ids)
        if tail:
            pr = torch.softmax(o.logits[0, :-1].float(), -1)
            t = torch.tensor(tail, device=pr.device)
            acc_next[i] = float((pr.argmax(-1) == t).float().mean())
            p_next[i] = float(pr[torch.arange(len(tail)), t].mean())
        if i % 500 == 0:
            print(f"{i}/{len(ctx)} {time.time()-t0:.0f}s tokens {len(ids)}", flush=True)
    for h in hooks:
        h.remove()
    assert np.isfinite(X[ok].astype(np.float32)).all(), "non-finite hidden states"
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_smoke{int(cfg.limit)}" if cfg.limit else ""
    np.savez_compressed(OUT / f"{actor}_decision_hidden{suffix}.npz", h=X, ok=ok, layers=np.array(layers), n_blocks=n, n_tokens=ntok,
                        p_next=p_next, acc_next=acc_next, excluded=json.dumps(why))
    print(f"saved {X.shape}; ok {ok.mean():.4f}; layers {layers} of {n}; next {NXT} tokens: mean accuracy {np.nanmean(acc_next):.4f}, mean p {np.nanmean(p_next):.4f}; {time.time()-t0:.0f}s", flush=True)


def stage_reader(cfg):
    """Qwen3-8B on the reader prompt (scripts/71 for gpt-oss: outputs/t6/gptoss_act.npz, reused)."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    actor = cfg.actor
    assert actor != "gptoss20b", "gpt-oss reader features exist: outputs/t6/gptoss_act.npz"
    t4 = importlib.import_module("69_t4_reader_action")
    T = texts(actor)
    if cfg.limit:
        T = T.head(int(cfg.limit))
    p = snapshot_download("Qwen/Qwen3-8B", cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda").eval()
    L = [18, 22, 26, 30, 32]                          # the layers of outputs/t4 and outputs/t6
    H = np.zeros((len(T), len(L), model.config.hidden_size), np.float16)
    t0 = time.time()
    for i, (c, a) in enumerate(zip(T.ctx, T.act)):
        s = tok.apply_chat_template([{"role": "user", "content": t4.Q.format(ctx=c, cot="", act=a)}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][:, -12000:].cuda()
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        H[i] = np.stack([hs[l][0, -1].float().cpu().numpy() for l in L]).astype(np.float16)
        if i % 1000 == 0:
            print(f"{i}/{len(T)} {time.time()-t0:.0f}s", flush=True)
    suffix = f"_smoke{int(cfg.limit)}" if cfg.limit else ""
    np.savez_compressed(OUT / f"{actor}_act{suffix}.npz", h=H, layers=np.array(L))
    print("saved", H.shape, f"{time.time()-t0:.0f}s", flush=True)


def reader_features(actor):
    f = Path("/data/models/graybox_ws/outputs/t6/gptoss_act.npz") if actor == "gptoss20b" else OUT / f"{actor}_act.npz"
    R = np.load(f)
    return R["h"][:, [int(x) for x in R["layers"]].index(32)].astype(np.float32)


def replicate(U, Hd, Xr, T, seed):
    """T15's E1/E2 on one agent. U units, Hd (h, ok) decision states, Xr reader layer-32 features, T texts."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t9 = importlib.import_module("77_t9_text_baseline")
    ok = Hd["ok"].astype(bool)
    dev = is_dev(U.scenario_id.values)
    d, f = ok & dev, ok & ~dev
    Yfull = Hd["h"].reshape(len(U), -1).astype(np.float32)
    sc = StandardScaler().fit(Yfull[d])
    pca = PCA(128, random_state=0).fit(sc.transform(Yfull[d]))
    Yd, Yf = pca.transform(sc.transform(Yfull[d])), pca.transform(sc.transform(Yfull[f]))
    y = U.y.values.astype(int)
    gd, gf, stf = U.scenario_id.values[d], U.scenario_id.values[f], stratum(U.template.values[f])
    tx = t9.TXT()
    tx.v = [t9.vec(), t9.vec()]
    src = {"READER": (Xr[d], Xr[f]), "TXT": (tx._X(T.ctx.values[d], T.act.values[d], True), tx._X(T.ctx.values[f], T.act.values[f], False))}
    nl, dim = Hd["h"].shape[1], Hd["h"].shape[2]
    res = {"n_dev": int(d.sum()), "n_fresh": int(f.sum()), "pos_dev": int(y[d].sum()), "pos_fresh": int(y[f].sum()),
           "scenarios_dev": int(len(set(gd))), "scenarios_fresh": int(len(set(gf))), "excluded_not_ok": int((~ok).sum()),
           "layers": [int(x) for x in Hd["layers"]], "state_dim": int(nl * dim), "pcs": 128,
           "variance_kept_by_128_pcs": round(float(pca.explained_variance_ratio_.sum()), 4), "alpha": {}, "r2_fresh": {},
           "r2_fresh_per_layer_full_space": {}}
    rec = {}
    for name, (Xd, Xf) in src.items():
        dense = name == "READER"
        xs = StandardScaler().fit(Xd) if dense else None
        Xd_, Xf_ = (xs.transform(Xd), xs.transform(Xf)) if dense else (Xd, Xf)
        cv = {}
        for a in ALPHAS:
            r = [r2_score(Yd[te], Ridge(alpha=a).fit(Xd_[tr], Yd[tr]).predict(Xd_[te]), multioutput="variance_weighted")
                 for tr, te in GroupKFold(5).split(Xd_, Yd, gd)]
            cv[a] = float(np.mean(r))
        a = max(cv, key=cv.get)
        rec[name] = Ridge(alpha=a).fit(Xd_, Yd).predict(Xf_)
        res["alpha"][name] = {"chosen": a, "cv_r2": {str(k): round(v, 4) for k, v in cv.items()}}
        res["r2_fresh"][name] = round(float(r2_score(Yf, rec[name], multioutput="variance_weighted")), 4)
        full_pred, full_true = pca.inverse_transform(rec[name]), sc.transform(Yfull[f])
        res["r2_fresh_per_layer_full_space"][name] = {int(L): round(float(r2_score(full_true[:, k * dim:(k + 1) * dim], full_pred[:, k * dim:(k + 1) * dim],
                                                                                   multioutput="variance_weighted")), 4) for k, L in enumerate(Hd["layers"])}
        print(name, "alpha", a, "R2", res["r2_fresh"][name], flush=True)
    res["E1_R2_READER_minus_TXT"] = [round(float(x), 4) for x in _ci(cluster_boot(
        lambda i: r2_score(Yf[i], rec["READER"][i], multioutput="variance_weighted") - r2_score(Yf[i], rec["TXT"][i], multioutput="variance_weighted"), gf, 1000, seed=seed))]
    res["E1_pass"] = res["E1_R2_READER_minus_TXT"][1] > 0
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000)).fit(Yd, y[d])
    yf = y[f]
    S = {"true": probe.decision_function(Yf), "READER_replica": probe.decision_function(rec["READER"]), "TXT_replica": probe.decision_function(rec["TXT"])}
    res["E2_auroc"] = {k: round(within_auroc(yf, v, stf), 4) for k, v in S.items()}
    res["E2_replica_minus_true"] = [round(float(x), 4) for x in _ci(cluster_boot(
        lambda i: within_auroc(yf[i], S["READER_replica"][i], stf[i]) - within_auroc(yf[i], S["true"][i], stf[i]), gf, 1000, seed=seed + 1))]
    res["E2_pass"] = res["E2_replica_minus_true"][1] > -0.02
    return res


def stage_analyze(cfg):
    out = {"split": "scenario-disjoint: development iff sha256(scenario_id) is even", "agents": {}}
    for actor in AGENTS:
        U = pd.read_parquet(units_path(actor))
        Hd = np.load(OUT / f"{actor}_decision_hidden.npz")
        Xr = reader_features(actor)
        T = texts(actor)
        assert len(U) == len(Hd["h"]) == len(Xr) == len(T), actor
        print("==", actor, flush=True)
        out["agents"][actor] = replicate(U, Hd, Xr, T, SEEDS[actor])
    a = out["agents"]
    out["E1_all_pass"] = all(a[x]["E1_pass"] for x in AGENTS)
    out["E2_all_pass"] = all(a[x]["E2_pass"] for x in AGENTS)
    out["n_agents_both_pass"] = int(sum(a[x]["E1_pass"] and a[x]["E2_pass"] for x in AGENTS))
    # reported (exploratory): Gemma-4's T15 development units under the same scenario-disjoint split
    t9 = importlib.import_module("77_t9_text_baseline")
    Ug = pd.read_parquet(I / "gemma4_clean_units.parquet")
    Hg = np.load(I / "gemma4_clean_decision_hidden.npz")
    Rg = np.load("/data/models/graybox_ws/outputs/t4/clean_act.npz")
    print("== gemma4 (exploratory)", flush=True)
    out["exploratory_gemma4_same_split"] = replicate(Ug, Hg, Rg["h"][:, [int(x) for x in Rg["layers"]].index(32)].astype(np.float32),
                                                     t9.gemma_texts("clean"), SEEDS["gemma4"])
    (RES / "t17_replication_agents.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"units": stage_units, "hidden": stage_hidden, "reader": stage_reader, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
