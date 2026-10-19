#!/usr/bin/env python
"""G7: what each output channel of an API reveals of a model's own state. The channel ladder, on the same contexts:
  answers  P(yes) on the 88 battery-v2 questions (side branches; G1's channel)
  top-k    the next-token distribution at the first content position, coarse-grained to its top k tokens + one
           "other" outcome (k = 1, 5, 20; OpenAI-style APIs return up to 20)
  full     the whole next-token distribution (every token), Fisher-Rao
  logprob  the centred log-probability vectors (cosine): the linear image of the final state that Theorem 6 uses
Distances: Fisher-Rao (= 2 arccos of the Bhattacharyya coefficient) on each channel's law; for answers, the arcsine
geometry of Theorem 1. By Cencov monotonicity, a coarser channel's distances are contractions of a finer one's.
Ground truth: the model's hidden states at the SAME position (L2's 6 layers; G1's dev-chosen layer, and the last layer).
  stage=contexts                      build the G7 contexts (fresh: not in L2 or G6)
  stage=gen  model=M set=dev|g7       (GPU) next-token distribution channels + hidden states at the same position
  stage=answers model=M set=g7        (GPU, vLLM) answers on the G7 contexts (scripts/132's code)
  stage=analyze set=dev|g7            (CPU)
The dev set = 600 L2 development contexts (100 per kind, seed 7) not in G1's layer-selection set.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = _mod("g", "129_infogeom.py")
OUT = Path("/data/models/graybox_ws/outputs/g7")
RES = ROOT / "results/t1"
KS = (1, 5, 20)
D = OmegaConf.create({"stage": "analyze", "model": "qwen3_8b", "set": "dev", "n_boot": 500, "batch": 8})


def contexts(which):
    if which == "dev":
        C = g.contexts().reset_index().rename(columns={"index": "row"})
        dev = np.where(C.split.values == "dev")[0]
        dev_sel = set(np.random.default_rng(0).choice(dev, 1000, replace=False).tolist())
        C = C[(C.split == "dev") & ~C.row.isin(dev_sel)]
        rng = np.random.default_rng(7)
        parts = [x.iloc[np.sort(rng.choice(len(x), 100, replace=False))] for _, x in sorted(C.groupby("source"))]
        C = pd.concat(parts).reset_index(drop=True).rename(columns={"source": "kind"})
        return C
    return pd.read_parquet(OUT / "contexts.parquet")


def stage_contexts(cfg):
    """1,000 fresh contexts of G6's ten kinds; texts in L2 or G6 are excluded (numpy seed 77; 100 per kind)."""
    from datasets import load_dataset
    used = set(pd.read_parquet(g.L2 / "contexts.parquet").text) | set(pd.read_parquet(OUT.parent / "g6/contexts.parquet").text)
    rng = np.random.default_rng(77)

    def take(items, n=100):
        items = [x for x in items if x["text"] not in used]
        idx = rng.choice(len(items), min(n, len(items)), replace=False)
        return [items[int(i)] for i in np.sort(idx)]
    out = []
    out += [dict(x, kind="sst2") for x in take([{"text": f"What do you think of this movie review?\n\n\"{r['sentence'].strip()}\""} for r in load_dataset("stanfordnlp/sst2", split="train")])]
    out += [dict(x, kind="harmful") for x in take([{"text": r["prompt"].strip()} for r in load_dataset("LLM-LAT/harmful-dataset", split="train")])]
    out += [dict(x, kind="benign") for x in take([{"text": r["prompt"].strip()} for r in load_dataset("LLM-LAT/benign-dataset", split="train").select(range(20000, 40000))])]
    out += [dict(x, kind="boolq") for x in take([{"text": f"{r['passage'].strip()}\n\nQuestion: {r['question'].strip()}?"} for r in load_dataset("google/boolq", split="validation")])]
    out += [dict(x, kind="gsm8k") for x in take([{"text": r["question"].strip()} for r in load_dataset("openai/gsm8k", "main", split="test")])]
    wk = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    out += [dict(x, kind="wiki") for x in take([{"text": f"Summarize this paragraph:\n\n{t.strip()}"} for t in wk["text"] if 300 < len(t.strip()) < 1200 and not t.strip().startswith("=")])]
    out += [dict(x, kind="news") for x in take([{"text": r["text"].strip()} for r in load_dataset("fancyzhx/ag_news", split="train")])]
    mb = load_dataset("google-research-datasets/mbpp")
    out += [dict(x, kind="code") for x in take([{"text": r["text"].strip()} for s in ("train", "validation", "prompt", "test") for r in mb[s]])]
    jb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    out += [dict(x, kind="jailbreak_bench") for x in take([{"text": r["Goal"].strip()} for s in ("harmful", "benign") for r in jb[s]])]
    pi = load_dataset("deepset/prompt-injections")
    out += [dict(x, kind="injection") for x in take([{"text": r["text"].strip()} for s in ("train", "test") for r in pi[s]])]
    df = pd.DataFrame(out)
    df["id"] = [f"g7_{k}_{i}" for i, k in enumerate(df.kind)]
    df["new_kind"] = df.kind.isin(["news", "code", "jailbreak_bench", "injection"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "contexts.parquet")
    print(df.groupby("kind").size().to_dict(), "overlap with L2/G6:", int(df.text.isin(used).sum()))


def stage_gen(cfg):
    """One forward pass per context: the next-token distribution at the first content position (the plain prompt plus
    the model's answer prefill, as L2 renders it) and the hidden states there. Stores per context the top-20 tokens and
    log-probabilities, and for the full channel the Bhattacharyya Gram BC[i, j] = sum_v sqrt(p_i(v) p_j(v)) (float64)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    L = _mod("l2run", "12_l2_run.py")
    m = L.MODELS[cfg.model]
    path = L.path_of(m["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = [int(x) for x in np.load(g.L2 / cfg.model / "hidden.npz")["layers"]]
    C = contexts(cfg.set)
    n = len(C)
    S = None                                                    # sqrt-probabilities on the GPU, float32 (n x V)
    Lc = None                                                   # centred log-probabilities, float32 (n x V)
    top_ids, top_lp, H = np.zeros((n, 20), np.int64), np.zeros((n, 20), np.float32), []
    for i, t in enumerate(C.text):
        s = L.render(tok, t, m) + (m["prefill"] or "")
        ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][:, -4096:].cuda()
        with torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
        lp = torch.log_softmax(o.logits[0, -1].float(), -1)
        if S is None:
            S = torch.zeros(n, lp.shape[0], dtype=torch.float32, device="cuda")
            Lc = torch.zeros(n, lp.shape[0], dtype=torch.float32, device="cuda")
        S[i] = torch.exp(0.5 * lp)
        Lc[i] = torch.clamp(lp, min=-60.0) - torch.clamp(lp, min=-60.0).mean()      # -inf guard (bf16 underflow)
        v, k = torch.topk(lp, 20)
        top_ids[i], top_lp[i] = k.cpu().numpy(), v.cpu().numpy()
        H.append(np.stack([o.hidden_states[l][0, -1].float().cpu().numpy() for l in layers]))
    BC = (S.double() @ S.double().T).cpu().numpy()
    LG = (Lc.double() @ Lc.double().T).cpu().numpy()
    out = OUT / cfg.set / cfg.model
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "channels.npz", bc_full=BC, lp_gram=LG, top_ids=top_ids, top_lp=top_lp, ids=C.id.values)
    np.savez_compressed(out / "hidden.npz", h=np.stack(H).astype(np.float32), layers=np.array(layers))
    print(cfg.model, cfg.set, "contexts", n, "vocab", S.shape[1], flush=True)


def stage_answers(cfg):
    g4 = _mod("g4", "132_g4_shift.py")
    g4.CONDITIONS = {"none": None}
    g4.contexts = lambda split, per_source: contexts(cfg.set)
    g4.OUT = OUT
    g4.stage_gen(OmegaConf.merge(g4.D, {"model": cfg.model, "split": cfg.set + "_answers", "replies": 1, "reply_tokens": 1}))


def topk_dist(ids, lp, k):
    """Fisher-Rao distance of the coarse-grained law: the top-k tokens + one 'other' outcome per context. Two contexts'
    laws live on the union of their top-k tokens and 'other'; tokens in one context's top-k but not the other's carry
    probability only on that side (the other side's mass for them is inside its 'other'), so the Bhattacharyya
    coefficient is the sum over shared top-k tokens plus sqrt(other_i * other_j)."""
    from scipy.sparse import csr_matrix
    ids, lp = ids[:, :k], lp[:, :k]
    n = len(ids)
    P = np.exp(lp.astype(np.float64))
    other = np.clip(1 - P.sum(1), 0, 1)
    Sp = csr_matrix((np.sqrt(P).ravel(), (np.repeat(np.arange(n), k), ids.ravel())), shape=(n, int(ids.max()) + 1))
    BC = (Sp @ Sp.T).toarray() + np.sqrt(np.outer(other, other))
    D = 2 * np.arccos(np.clip(BC, 0, 1))
    np.fill_diagonal(D, 0)
    return D


def stage_analyze(cfg):
    C = contexts(cfg.set)
    Call = g.contexts()
    dev = np.where(Call.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    M = g.MODELS
    ch = {f"top{k}": {} for k in KS}
    ch["full"], ch["answers"], ch["logprob"] = {}, {}, {}
    Hk, Hlast = {}, {}
    for m in M:
        d = OUT / cfg.set / m
        z = np.load(d / "channels.npz")
        for k in KS:
            ch[f"top{k}"][m] = topk_dist(z["top_ids"], z["top_lp"], k)
        Df = 2 * np.arccos(np.clip(z["bc_full"], 0, 1))
        np.fill_diagonal(Df, 0)
        ch["full"][m] = Df
        G = z["lp_gram"]
        nrm = np.sqrt(np.diag(G))
        ch["logprob"][m] = 1 - G / nrm[:, None] / nrm[None, :]
        if cfg.set == "dev":
            p, _ = g.p_yes(m)
            ch["answers"][m] = np.sqrt(g.sq_dist(g.theta(p[C.row.values])))
        else:
            ch["answers"][m] = np.sqrt(g.sq_dist(g.theta(_mod("g4", "132_g4_shift.py").p_of(np.load(OUT / f"{cfg.set}_answers" / m / "psi_none.npz")))))
        hz = np.load(d / "hidden.npz")
        kk, _, _ = g.select_layer(m, dev_sel)
        Hk[m], Hlast[m] = g.cos_dist(hz["h"][:, kk].astype(np.float64)), g.cos_dist(hz["h"][:, -1].astype(np.float64))
    n = len(C)
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731

    def excess(Dc, Dh, s):
        R = np.array([[g.rho(sub(Dc[a], s), sub(Dh[b], s)) for b in M] for a in M])
        res = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()
        return float(np.diag(res).mean()), R
    res = {"set": cfg.set, "n": n, "channels": {}}
    rng = np.random.default_rng(1390)
    boots = [np.unique(rng.integers(0, n, n)) for _ in range(cfg.n_boot)]
    order = ["answers", "top1", "top5", "top20", "full", "logprob"]
    allidx = np.arange(n)
    for hname, Dh in (("chosen_layer", Hk), ("last_layer", Hlast)):
        B = {}
        for c in order:
            e, R = excess(ch[c], Dh, allidx)
            B[c] = np.array([excess(ch[c], Dh, s)[0] for s in boots])
            res["channels"].setdefault(hname, {})[c] = {"excess": [round(e, 4), round(float(np.percentile(B[c], 2.5)), 4), round(float(np.percentile(B[c], 97.5)), 4)],
                                                        "own_mean": round(float(np.mean(np.diag(R))), 4),
                                                        "others_mean": round(float(R[~np.eye(len(M), dtype=bool)].mean()), 4),
                                                        "own_best": int(sum(R[i, i] == R[i].max() for i in range(len(M))))}
            print(hname, c, res["channels"][hname][c], flush=True)
        diffs = {}                                               # paired: the same resamples for every channel
        for a, bb in zip(order[:-1], order[1:]):
            dd = B[bb] - B[a]
            diffs[f"{bb}_minus_{a}"] = [round(float(np.mean(dd)), 4), round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]
        res["channels"][hname]["ladder"] = diffs
        if hname == "last_layer":
            B_last = B
        print(hname, "ladder", diffs, flush=True)
    L = res["channels"]
    res["K1_pass"] = L["chosen_layer"]["answers"]["excess"][1] > -0.02 and L["chosen_layer"]["answers"]["excess"][2] < 0.02
    res["K2_pass"] = L["last_layer"]["top20"]["excess"][1] > 0.02
    ld = L["last_layer"]["ladder"]
    t20a = (B_last["top20"] - B_last["answers"])
    res["K3_top20_minus_answers"] = [round(float(np.mean(t20a)), 4), round(float(np.percentile(t20a, 2.5)), 4), round(float(np.percentile(t20a, 97.5)), 4)]
    res["K3_pass"] = res["K3_top20_minus_answers"][1] > 0 and ld["top5_minus_top1"][1] > 0 and ld["top20_minus_top5"][1] > 0
    lt = (B_last["logprob"] - B_last["top20"])
    res["K4_logprob_minus_top20"] = [round(float(np.mean(lt)), 4), round(float(np.percentile(lt, 2.5)), 4), round(float(np.percentile(lt, 97.5)), 4)]
    res["K4_pass"] = res["K4_logprob_minus_top20"][1] > 0
    res["K5_pass"] = L["last_layer"]["logprob"]["own_best"] >= 7 and L["last_layer"]["answers"]["own_best"] <= 2
    print({k: res[k] for k in res if k.startswith("K")}, flush=True)
    name = f"g7_channels_DRYRUN_{cfg.set}.json" if cfg.set == "dev" else "g7_channels.json"
    (RES / name).write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"contexts": stage_contexts, "gen": stage_gen, "answers": stage_answers, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
