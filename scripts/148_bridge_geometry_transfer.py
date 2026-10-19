#!/usr/bin/env python
"""Bridge test (second review): does a label-free GEOMETRIC agreement between a reader and an agent predict how well a
monitor built on that reader transfers to that agent?
Readers: T19's 8 (minilm, mpnet, qemb06 encoders; Qwen3 0.6B, 1.7B, 4B, 8B (shipped), 14B decoders), same extraction.
Agents: T17's three open agents with stored decision states (gpt-oss-20b, Qwen3.8-27B, Muse-Glimmer-30B).
For each (reader r, agent a):
  transfer   within-template AUROC on agent a of a probe trained ONLY on Gemma-4's 17,623 steps in reader r's features
             (the shipped recipe: standardise, PCA 256, logistic C 0.05)
  geometry   Spearman between reader r's cosine geometry and agent a's own decision-state geometry (Euclidean on the
             top-128 PCs of the standardised state, PCA fit on the same split), over 1,000 random steps; no labels
Split: T17's scenario hash. dev = development scenarios (dry run); held = held-out scenarios (confirmatory).
  stage=features reader=R split=dev|held  (GPU)  the split's units only (held-out features are made after the lock)
  stage=analyze split=dev|held  (CPU)
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import within_auroc  # noqa: E402

t19 = importlib.import_module("100_t19_small_readers")
t17 = importlib.import_module("96_t17_replication_agents")
OUT = Path("/data/models/graybox_ws/outputs/bridge")
RES = ROOT / "results/t1"
AGENTS = ("gptoss20b", "qwen38", "museglimmer")
READERS = ("minilm", "mpnet", "qemb06", "q06", "q17", "q4", "q8", "q14")
D = OmegaConf.create({"stage": "analyze", "reader": "q06", "split": "dev", "n_geo": 1000, "n_boot": 500})


def stage_features(cfg):
    r = cfg.reader
    OUT.mkdir(parents=True, exist_ok=True)
    if r in t19.BROWSER:
        from graybox.embed import Encoder
        enc = Encoder(r, "cuda", t19.HF)
        f = lambda c, a: enc.encode(list(c), list(a))  # noqa: E731
    else:
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig
        from graybox.monitor import ReaderMonitor
        n = AutoConfig.from_pretrained(snapshot_download(t19.DECODER[r], cache_dir="/data/models", local_files_only=True)).num_hidden_layers
        mon = ReaderMonitor(probe=None, layer=round(n * 8 / 9), reader=t19.DECODER[r])
        f = lambda c, a: np.vstack([mon.features(t19.Q.format(ctx=x, cot="", act=y)) for x, y in zip(c, a)])  # noqa: E731
    import pandas as pd
    for a in AGENTS:
        T = t17.texts(a)
        U = pd.read_parquet(t17.units_path(a))
        dev = t17.is_dev(U.scenario_id.values)
        rows = np.where(dev if cfg.split == "dev" else ~dev)[0]              # held-out features only after the lock
        np.savez_compressed(OUT / f"{r}_{a}_{cfg.split}.npz", h=f(T.ctx.values[rows], T.act.values[rows]).astype(np.float16), rows=rows)
        print(r, a, cfg.split, len(rows), "done", flush=True)


def reader_feats(r, a, split, n_units):
    """Reader r's features for agent a's units, as a full-length array (rows outside the split are NaN)."""
    if r == "q8":
        return t17.reader_features(a).astype(np.float32)
    z = np.load(OUT / f"{r}_{a}_{split}.npz")
    X = np.full((n_units, z["h"].shape[1]), np.nan, np.float32)
    X[z["rows"]] = z["h"].astype(np.float32)
    return X


def probe():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=4000))


def rho(A, B):
    from scipy.stats import rankdata
    iu = np.triu_indices(len(A), 1)
    return float(np.corrcoef(rankdata(A[iu]), rankdata(B[iu]))[0, 1])


def cosd(X):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return 1 - Xn @ Xn.T


def eucd(X):
    G = X @ X.T
    n = np.diag(G)
    return np.sqrt(np.maximum(n[:, None] + n[None, :] - 2 * G, 0))


def stage_analyze(cfg):
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    tr = t19.data("train")
    ytr = tr.y.values.astype(int)
    rng = np.random.default_rng(1480)
    res = {"split": cfg.split, "pairs": {}}
    probes = {r: probe().fit(t19.features(r, "train"), ytr) for r in READERS}
    for a in AGENTS:
        U = pd.read_parquet(t17.units_path(a))
        Hd = np.load(t17.OUT / f"{a}_decision_hidden.npz")
        ok = Hd["ok"].astype(bool)
        dev = t17.is_dev(U.scenario_id.values)
        m = ok & (dev if cfg.split == "dev" else ~dev)
        Y = Hd["h"].reshape(len(U), -1).astype(np.float32)[m]
        Z = PCA(128, random_state=0).fit_transform(StandardScaler().fit_transform(Y))
        idx = np.sort(rng.choice(m.sum(), min(cfg.n_geo, m.sum()), replace=False))
        Dag = eucd(Z[idx].astype(np.float64))
        y, st = U.y.values[m].astype(int), t17.stratum(U.template.values[m])
        for r in READERS:
            X = reader_feats(r, a, cfg.split, len(U))[m]
            auc = within_auroc(y, probes[r].predict_proba(X)[:, 1], st)
            geo = rho(cosd(X[idx].astype(np.float64)), Dag)
            res["pairs"][f"{r}|{a}"] = {"auroc": round(float(auc), 4), "geometry": round(geo, 4)}
            print(a, r, res["pairs"][f"{r}|{a}"], flush=True)
    from scipy.stats import spearmanr
    keys = list(res["pairs"])
    au = np.array([res["pairs"][k]["auroc"] for k in keys])
    ge = np.array([res["pairs"][k]["geometry"] for k in keys])
    res["spearman_all_pairs"] = round(float(spearmanr(ge, au)[0]), 4)
    within = {a: round(float(spearmanr([res["pairs"][f"{r}|{a}"]["geometry"] for r in READERS],
                                       [res["pairs"][f"{r}|{a}"]["auroc"] for r in READERS])[0]), 4) for a in AGENTS}
    res["spearman_within_agent"] = within
    res["spearman_within_agent_mean"] = round(float(np.mean(list(within.values()))), 4)
    # permutation null for the within-agent mean: shuffle readers' geometry within each agent
    null = []
    for _ in range(10000):
        null.append(np.mean([spearmanr(rng.permutation([res["pairs"][f"{r}|{a}"]["geometry"] for r in READERS]),
                                       [res["pairs"][f"{r}|{a}"]["auroc"] for r in READERS])[0] for a in AGENTS]))
    res["within_mean_perm_p"] = round(float((1 + np.sum(np.array(null) >= res["spearman_within_agent_mean"])) / (1 + len(null))), 4)
    print(json.dumps({k: res[k] for k in res if k != "pairs"}, indent=1))
    name = "bridge_DRYRUN_dev.json" if cfg.split == "dev" else "bridge_geometry_transfer.json"
    (RES / name).write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"features": stage_features, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
