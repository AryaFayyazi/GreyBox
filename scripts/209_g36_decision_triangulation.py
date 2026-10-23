#!/usr/bin/env python
"""G36: triangulation at the agent's decision (docs/prereg_G36_decision_triangulation.md). How well do an agent's own
answers at a proposed call reconstruct its own decision-state geometry? For a closed agent the decision states are not
visible; G34's triangulation, moved from generic contexts to the calls the monitor scores.
Reference geometries: G1's 8 open models read each call's transcript and proposed action (scripts/69's reader prompt;
each model's chat template; last prompt token; G1's development-chosen layer). Nothing is trained.
Calls: G15's held-out calls of Gemma-4 (600), gpt-oss-20b (599), Qwen3.8 (373), Muse-Glimmer (600), and Closed-A's 495.
  stage=texts                (CPU)  -> outputs/g36/texts_<agent>.parquet
  stage=hidden reader=<m>    (GPU)  -> outputs/g36/<m>.npz   (all agents' calls)
  stage=analyze              (CPU)  -> results/t1/g36_decision_triangulation.json
"""
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
OUT = Path("/data/models/graybox_ws/outputs/g36")
OPEN = ["gemma4", "gptoss20b", "qwen38", "museglimmer"]
AGENTS = OPEN + ["closed"]
READERS = ["gemma3_4b", "gptoss20b", "llama31_8b", "mistral7b", "olmo3_7b", "phi4mini", "qwen25_7b", "qwen3_8b"]
LAYER = {"gemma3_4b": 34, "gptoss20b": 12, "llama31_8b": 24, "mistral7b": 8, "olmo3_7b": 29, "phi4mini": 29, "qwen25_7b": 25, "qwen3_8b": 27}
B_N, SEED, DLAYER = 500, 3600, 3


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def calls(agent):
    """the G15 call table of an agent: episode_id, event_id, y, template, scenario_id, counts"""
    M = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    if agent == "closed":
        C = pd.DataFrame([json.loads(x) for x in open(M.OUT / "claude_t8b.jsonl")])
        return C[~C.set_index(["episode_id", "event_id"]).index.isin(M.PILOT)].reset_index(drop=True)
    B = pd.read_parquet(M.OUT / f"{agent}_{'fresh' if agent == 'gemma4' else 'heldout'}.parquet")
    U, _ = M.states(agent)
    bad = set(zip(U.episode_id[~U.ok.values], U.event_id[~U.ok.values].astype(int)))
    return B[[(e, int(v)) not in bad for e, v in zip(B.episode_id, B.event_id)]].reset_index(drop=True)


def stage_texts():
    from graybox import cape_bridge as cb
    t3 = importlib.import_module("68_t3_deployable")
    M = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    OUT.mkdir(parents=True, exist_ok=True)
    for a in AGENTS:
        C = calls(a)
        if a == "closed":
            T = importlib.import_module("76_t8b_claude_all_actions").units()
            U = importlib.import_module("172_claude_decision_greybox").units()
            assert (T.episode_id.values == U.episode_id.values).all()
            pos = {(e, int(v)): i for i, (e, v) in enumerate(zip(U.episode_id, U.event_id))}
            ix = [pos[(e, int(v))] for e, v in zip(C.episode_id, C.event_id)]
            ctx, act = list(T.ctx.values[ix]), list(T.act.values[ix])
        else:
            U, _ = M.states(a)
            info = {(e, int(v)): (s, b) for e, v, s, b in zip(U.episode_id, U.event_id, U.set, U.bench)}
            ctx, act = [], []
            for e, v in zip(C.episode_id, C.event_id):
                s, b = info[(e, int(v))]
                ep = cb.rollouts(s).loc[e]
                evs = list(ep.events)
                ev = {int(x["event_id"]): x for x in evs}[int(v)]
                ctx.append(t3.ctx_text(cb.specs(b)[ep.instance_id], evs, int(v)))
                act.append(t3.act_text(ev["tool"], json.loads(ev["args_json"] or "{}")))
        pd.DataFrame({"episode_id": C.episode_id, "event_id": C.event_id.astype(int), "ctx": ctx, "act": act}).to_parquet(OUT / f"texts_{a}.parquet")
        print(a, len(C), flush=True)


def stage_hidden(reader):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    L2 = _mod("l2", "scripts/12_l2_run.py")
    t4 = importlib.import_module("69_t4_reader_action")
    cfg = L2.MODELS[reader]
    path = L2.path_of(cfg["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    except Exception:  # noqa: BLE001
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    out = {}
    for a in AGENTS:
        T = pd.read_parquet(OUT / f"texts_{a}.parquet")
        H = np.zeros((len(T), model.config.get_text_config().hidden_size), np.float32)
        t0 = time.time()
        for i, (c, x) in enumerate(zip(T.ctx, T.act)):
            s = tok.apply_chat_template([{"role": "user", "content": t4.Q.format(ctx=c, cot="", act=x)}], tokenize=False,
                                        add_generation_prompt=True, **cfg["kw"])
            ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][:, -12000:].cuda()
            with torch.no_grad():
                hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
            H[i] = hs[LAYER[reader]][0, -1].float().cpu().numpy()
        out[a] = H
        print(reader, a, H.shape, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT / f"{reader}.npz", **out)


# ------------------------------------------------------------------ analysis (CPU)
def cosd(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return 1 - X @ X.T


def answer_dist(C):
    """Theorem 5's debiased Fisher-Rao distances from the agent's own answers (k yes of m parsed, 4 asked)"""
    k, m = C[..., 0], C[..., 1]
    th = 2 * np.arcsin(np.sqrt((k + 3 / 8) / (np.maximum(m, 1) + 3 / 4)))
    var = 1 / (np.maximum(m, 1) + 0.5)
    G = th @ th.T
    nr = np.diag(G)
    D2 = nr[:, None] + nr[None, :] - 2 * G - (var.sum(1)[:, None] + var.sum(1)[None, :])
    D = np.sqrt(np.maximum(D2, 0))
    np.fill_diagonal(D, 0)
    return D


def upper(M):
    return M[np.triu_indices(len(M), 1)]


def srho(A, B):
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(upper(A)), rankdata(upper(B)))[0, 1])


def geometries():
    M = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    R = {r: np.load(OUT / f"{r}.npz") for r in READERS}
    G = {}
    for a in AGENTS:
        C = calls(a)
        Hr = {r: cosd(R[r][a]) for r in READERS}
        A = answer_dist(np.array([json.loads(c) if isinstance(c, str) else c for c in C.counts], float))
        Hown = None
        if a != "closed":
            U, H = M.states(a)
            key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
            Hown = cosd(H[[key[(e, int(v))] for e, v in zip(C.episode_id, C.event_id)], DLAYER].astype(np.float64))
        G[a] = {"Hr": Hr, "A": A, "Hown": Hown, "n": len(C)}
    return G


def loadings(S):
    n = len(S)
    lam = np.sqrt(np.clip([np.mean([S[i, j] for j in range(n) if j != i]) for i in range(n)], 1e-6, None))
    for _ in range(500):
        new = np.array([sum(S[i, j] * lam[j] for j in range(n) if j != i) / sum(lam[j] ** 2 for j in range(n) if j != i) for i in range(n)])
        if np.max(np.abs(new - lam)) < 1e-10:
            break
        lam = new
    return lam


def per_agent(g, s=None):
    sub = (lambda X: X) if s is None else (lambda X: X[np.ix_(s, s)])
    Hr = [sub(g["Hr"][r]) for r in READERS]
    S = np.array([[1.0 if i == j else srho(Hr[i], Hr[j]) for j in range(len(READERS))] for i in range(len(READERS))])
    lam = loadings(S)
    ra = np.array([srho(sub(g["A"]), h) for h in Hr])
    out = {"kappa": float(ra @ lam / (lam @ lam)), "rho_answers_vs_readers": ra}
    if g["Hown"] is not None:
        rh = np.array([srho(sub(g["Hown"]), h) for h in Hr])
        out["lambda_own"] = float(rh @ lam / (lam @ lam))
        out["observed"] = srho(sub(g["A"]), sub(g["Hown"]))
    return out


def triangulate(P):
    res = {}
    for a in OPEN:
        others = [P[b]["lambda_own"] for b in OPEN if b != a]
        res[a] = {"observed": P[a]["observed"], "predicted": P[a]["kappa"] * float(np.median(others)),
                  "interval": [P[a]["kappa"] * min(others), P[a]["kappa"] * max(others)],
                  "lambda_own": P[a]["lambda_own"], "kappa": P[a]["kappa"]}
    lams = [P[b]["lambda_own"] for b in OPEN]
    res["closed"] = {"predicted": P["closed"]["kappa"] * float(np.median(lams)),
                     "interval": [P["closed"]["kappa"] * min(lams), P["closed"]["kappa"] * max(lams)], "kappa": P["closed"]["kappa"]}
    return res


def stage_analyze():
    G = geometries()
    P = {a: per_agent(g) for a, g in G.items()}
    T = triangulate(P)
    mae = float(np.mean([abs(T[a]["observed"] - T[a]["predicted"]) for a in OPEN]))
    inside = int(sum(T[a]["interval"][0] <= T[a]["observed"] <= T[a]["interval"][1] for a in OPEN))
    rng = np.random.default_rng(SEED)
    bc, bm = [], []
    for _ in range(B_N):
        Pb = {a: per_agent(g, np.unique(rng.integers(0, g["n"], g["n"]))) for a, g in G.items()}
        Tb = triangulate(Pb)
        bc.append(Tb["closed"]["predicted"])
        bm.append(np.mean([abs(Tb[a]["observed"] - Tb[a]["predicted"]) for a in OPEN]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    r4 = lambda x: round(float(x), 4)  # noqa: E731
    res = {"calls": {a: G[a]["n"] for a in AGENTS},
           "leave_one_agent_out": {a: {k: (r4(v) if not isinstance(v, list) else [r4(x) for x in v]) for k, v in T[a].items()} for a in OPEN},
           "V1_mean_abs_error": [r4(mae)] + ci(bm), "V2_inside_interval": inside,
           "E_closed": {"predicted": [r4(T["closed"]["predicted"])] + ci(bc), "interval": [r4(x) for x in T["closed"]["interval"]],
                        "kappa": r4(T["closed"]["kappa"])},
           "rho_answers_vs_readers": {a: [r4(x) for x in P[a]["rho_answers_vs_readers"]] for a in AGENTS}}
    res["endpoints"] = {"V1_mae_lt_0.10": mae < 0.10, "V2_inside_ge_3_of_4": inside >= 3, "E1_closed_lower_gt_0": res["E_closed"]["predicted"][1] > 0}
    (ROOT / "results/t1/g36_decision_triangulation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    {"texts": lambda: stage_texts(), "hidden": lambda: stage_hidden(kw["reader"]), "analyze": stage_analyze}[kw.get("stage", "analyze")]()
