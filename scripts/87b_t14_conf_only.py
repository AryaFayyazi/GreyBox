#!/usr/bin/env python
"""Exploratory follow-up of T14 (reviewer finding M6): the open agents' labelled AgentDojo calls that T14 added to
training (T7) come from CAPE's development split of user tasks; T14's test (T13b) pooled development and confirmation
calls of the closed agent. Here the test is restricted to the CONFIRMATION split, whose user tasks are disjoint from the
training calls (checked below), and both cuts are reported. Same fits as scripts/87 (k = 0 and all 3,455 open-agent
calls). Output: results/t1/t14_conf_only_exploratory.json
python scripts/87b_t14_conf_only.py
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"                       # as scripts/87 (T14): BLAS results depend on the thread count
import importlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

I, T4 = Path("/data/models/graybox_ws/outputs/interrogate"), Path("/data/models/graybox_ws/outputs/t4")
AD = Path("/data/models/graybox_ws/outputs/agentdojo_gpt")


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t7, t13b = importlib.import_module("73_t7_agentdojo"), importlib.import_module("86_t13b_analysis")
    UA = pd.concat([pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")], ignore_index=True)
    Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz")
    j = [int(x) for x in Hd["layers"]].index(32)
    XC, yC = np.vstack([Hd["h"][:, j], Hf["h"][:, j]]).astype(np.float32), UA.y.values.astype(int)
    A7 = t7.units()
    X7, y7 = np.load(t7.OUT / "reader.npz")["h"].astype(np.float32), A7.y.values.astype(int)
    G = t13b.units()
    XG, yG = np.load(t13b.OUT / "reader.npz")["h"].astype(np.float32), G.y.values.astype(int)
    conf_eps = {json.loads(l)["key"] for f in ("conf_gpt6luna_none_attack.jsonl", "conf_gpt6luna_none_benign.jsonl") for l in open(AD / f)}
    conf = G.episode.isin(conf_eps).values
    ut = lambda ep: tuple(str(ep).split("|")[:2])  # noqa: E731   (suite, user task)
    train_tasks = {ut(e) for e in A7.episode}
    overlap = {ut(e) for e in G.episode[conf]} & train_tasks
    fits = {}
    for k, add in (("zero_open_labels", np.arange(0)), ("all_open_labels", np.arange(len(X7)))):
        m = make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=4000))
        fits[k] = m.fit(np.vstack([XC, X7[add]]), np.concatenate([yC, y7[add]])).predict_proba(XG)[:, 1]
    J = pd.read_parquet(t13b.OUT / "judge.parquet").p.values
    y, st, g, ben = yG[conf], G.suite.values[conf], G.episode.values[conf], (G.kind == "benign").values[conf]
    S = {"full": fits["all_open_labels"][conf], "zero": fits["zero_open_labels"][conf], "judge": J[conf]}
    res = {"conf_calls": int(conf.sum()), "conf_injected": int(y.sum()), "conf_user_tasks_also_in_T7_training": len(overlap),
           "auroc_within_suite": {k: round(within_auroc(y, v, st), 4) for k, v in S.items()},
           "caught_label_free_5pct": {k: round(float((v[y == 1] >= np.quantile(v, .95)).mean()), 4) for k, v in S.items()},
           "caught_benign_cut_5pct": {k: round(float((v[y == 1] >= np.quantile(v[ben], .95)).mean()), 4) for k, v in S.items()}}

    def boot(a, b, seed):
        d = np.asarray(cluster_boot(lambda i: within_auroc(y[i], S[a][i], st[i]) - within_auroc(y[i], S[b][i], st[i]), g, 1000, seed=seed), float)
        d = d[np.isfinite(d)]
        return [round(float(within_auroc(y, S[a], st) - within_auroc(y, S[b], st)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    res["full_minus_judge"], res["full_minus_zero"] = boot("full", "judge", 1411), boot("full", "zero", 1421)
    (ROOT / "results/t1/t14_conf_only_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
