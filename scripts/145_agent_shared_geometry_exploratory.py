#!/usr/bin/env python
"""EXPLORATORY: the link between the geometry results and the monitor. On real agent steps, does the open reader's
hidden geometry match each AGENT's own decision-state geometry, more than the transcript text does? Nothing is fit:
  agent    Euclidean distances of the agent's decision state (its top-128 principal components, the released Y_true)
  reader   1 - cosine of the reader's (Qwen3-8B, layer 32) raw state on the same step's transcript + action
  text     1 - cosine of TF-IDF (T9's vectorizer settings, fit without labels on these steps) of the same input
Spearman over all pairs of 1,500 random held-out steps per agent (numpy seed 145); bootstrap over steps (200).
python scripts/145_agent_shared_geometry_exploratory.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
I = Path("/data/models/graybox_ws/outputs/interrogate")
T4 = Path("/data/models/graybox_ws/outputs/t4")
U = ROOT / "results/units"
RES = ROOT / "results/t1"


def cosd(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return 1 - Xn @ Xn.T


def eucd(X):
    G = X @ X.T
    n = np.diag(G)
    return np.sqrt(np.maximum(n[:, None] + n[None, :] - 2 * G, 0))


def rho(A, B):
    from scipy.stats import rankdata
    iu = np.triu_indices(len(A), 1)
    return float(np.corrcoef(rankdata(A[iu]), rankdata(B[iu]))[0, 1])


def data():
    t9 = importlib.import_module("77_t9_text_baseline")
    m = importlib.import_module("96_t17_replication_agents")
    out = {}
    Uf = pd.read_parquet(I / "gemma4_fresh_units.parquet")
    okf = np.load(I / "gemma4_fresh_decision_hidden.npz")["ok"].astype(bool)
    Hf = np.load(T4 / "fresh_act.npz")
    jj = [int(x) for x in Hf["layers"]].index(32)
    Tf = t9.gemma_texts("fresh")
    out["Gemma-4"] = (np.load(U / "t15_gemma_replication.npz")["Y_true"].astype(np.float64), Hf["h"][okf, jj].astype(np.float64),
                      (Tf.ctx.values[okf], Tf.act.values[okf]))
    for a, name in (("gptoss20b", "gpt-oss-20b"), ("qwen38", "Qwen3.8-27B"), ("museglimmer", "Muse-Glimmer-30B")):
        Ua = pd.read_parquet(m.units_path(a))
        ok = np.load(m.OUT / f"{a}_decision_hidden.npz")["ok"].astype(bool)
        f = ok & ~m.is_dev(Ua.scenario_id.values)
        T = m.texts(a)
        out[name] = (np.load(U / f"t17_{a}_replication.npz")["Y_true"].astype(np.float64), m.reader_features(a)[f].astype(np.float64),
                     (T.ctx.values[f], T.act.values[f]))
    return out


def main():
    t9 = importlib.import_module("77_t9_text_baseline")
    rng = np.random.default_rng(145)
    res = {}
    for name, (Y, R, (ctx, act)) in data().items():
        assert len(Y) == len(R) == len(ctx), (name, len(Y), len(R), len(ctx))
        s = np.sort(rng.choice(len(Y), min(1500, len(Y)), replace=False))
        v = t9.vec()
        X = v.fit_transform([c + "\n" + a for c, a in zip(ctx[s], act[s])]).toarray()
        Da, Dr, Dt = eucd(Y[s]), cosd(R[s]), cosd(X + 1e-12)
        b = []
        for _ in range(200):
            k = np.unique(rng.integers(0, len(s), len(s)))
            sub = lambda A: A[np.ix_(k, k)]  # noqa: E731
            b.append((rho(sub(Dr), sub(Da)), rho(sub(Dt), sub(Da))))
        b = np.array(b)
        d = b[:, 0] - b[:, 1]
        res[name] = {"n": int(len(s)), "rho_reader_vs_agent": round(rho(Dr, Da), 4), "rho_text_vs_agent": round(rho(Dt, Da), 4),
                     "reader_minus_text": [round(float(d.mean()), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
        print(name, json.dumps(res[name]), flush=True)
    (RES / "agent_shared_geometry_exploratory.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
