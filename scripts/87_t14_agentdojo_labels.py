#!/usr/bin/env python
"""T14: do labelled AgentDojo calls from OPEN agents fix the external benchmark for a CLOSED agent?
(docs/prereg_T14_agentdojo_labels.md). CPU only; threads pinned to 1.
python scripts/87_t14_agentdojo_labels.py
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
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
KS, DRAWS = (0, 100, 400, 3455), 5


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
    assert len(X7) == len(A7) == 3455
    G = t13b.units()
    XG, yG = np.load(t13b.OUT / "reader.npz")["h"].astype(np.float32), G.y.values.astype(int)
    assert len(XG) == len(G)
    st, g, ben = G.suite.values, G.episode.values, (G.kind == "benign").values
    J = pd.read_parquet(t13b.OUT / "judge.parquet").p.values
    S = {}
    for k in KS:
        S[k] = []
        for d in range(DRAWS if 0 < k < len(X7) else 1):
            add = np.random.default_rng(d).choice(len(X7), k, replace=False) if 0 < k < len(X7) else np.arange(len(X7) if k else 0)
            X, y = np.vstack([XC, X7[add]]), np.concatenate([yC, y7[add]])
            m = make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=4000)).fit(X, y)
            S[k].append(m.predict_proba(XG)[:, 1])
        print("fitted k =", k, flush=True)
    res = {"test_calls": int(len(G)), "test_injected": int(yG.sum()), "judge_auroc": round(within_auroc(yG, J, st), 4), "curve": {}}
    for k in KS:
        a = [within_auroc(yG, s, st) for s in S[k]]
        c_b = [(s[yG == 1] >= np.quantile(s[ben], .95)).mean() for s in S[k]]
        c_l = [(s[yG == 1] >= np.quantile(s, .95)).mean() for s in S[k]]
        res["curve"][str(k)] = {"auroc": round(float(np.mean(a)), 4), "caught_benign5": round(float(np.mean(c_b)), 4),
                                "caught_labelfree5": round(float(np.mean(c_l)), 4), "draws": len(S[k])}
    full, zero = S[KS[-1]][0], S[0][0]

    def boot(f, seed):
        d = np.array(cluster_boot(f, g, 1000, seed=seed))
        d = d[np.isfinite(d)]
        return [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    res["L1_full_minus_judge"] = boot(lambda i: within_auroc(yG[i], full[i], st[i]) - within_auroc(yG[i], J[i], st[i]), 141)
    res["L2_full_minus_zero"] = boot(lambda i: within_auroc(yG[i], full[i], st[i]) - within_auroc(yG[i], zero[i], st[i]), 142)
    res["L1_pass"], res["L2_pass"] = res["L1_full_minus_judge"][1] > 0, res["L2_full_minus_zero"][1] > 0
    (ROOT / "results/t1/t14_agentdojo_labels.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
