"""Exploratory (post hoc, on the stored per-step data of the preregistered replication tests): the replica
Rao-Blackwellizes a white-box probe.

Proposition (paper, Method): if Y is independent of the agent's state phi given the reader's state psi, then for every
convex loss l, E l(E[s|psi], Y) <= E l(s, Y) for a probe score s = w'phi; the same holds for convex pairwise surrogates
of the AUROC. Checks on the four agents (Gemma-4 and the three other architectures):
  1. the premise: given the replica score h, does the residual e = s - h still carry label information
     (grouped 5-fold CV, AUROC of a logistic model on (h) vs (h, e))?
  2. the conclusion: pairwise logistic risk and log loss of the best affine calibration, true state vs replica;
     pooled and within-template AUROC.

python scripts/123_rao_blackwell_exploratory.py -> results/t1/rao_blackwell_exploratory.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.stats import within_auroc  # noqa: E402

FILES = {"Gemma-4": "t15_gemma_replication", "gpt-oss-20b": "t17_gptoss20b_replication",
         "Qwen3.8-27B": "t17_qwen38_replication", "Muse-Glimmer-30B": "t17_museglimmer_replication"}


def main():
    out = {}
    for name, f in FILES.items():
        z = np.load(ROOT / "results/units" / f"{f}.npz")
        y, g, tpl = z["y"].astype(int), z["scenario"], z["template"]
        s, h = z["probe_true"].astype(np.float64), z["probe_READER"].astype(np.float64)
        e = s - h
        cv = GroupKFold(5)
        p1 = cross_val_predict(LogisticRegression(C=1.0), h[:, None], y, groups=g, cv=cv, method="decision_function")
        p2 = cross_val_predict(LogisticRegression(C=1.0), np.c_[h, e], y, groups=g, cv=cv, method="decision_function")
        cal = lambda v: cross_val_predict(LogisticRegression(C=1.0), v[:, None], y, groups=g, cv=cv, method="predict_proba")[:, 1]  # noqa: E731
        rng = np.random.default_rng(0)
        P, N = np.where(y == 1)[0], np.where(y == 0)[0]
        i, j = rng.choice(P, 500_000), rng.choice(N, 500_000)
        pair = lambda v: float(np.mean(np.logaddexp(0.0, -(v[i] - v[j]))))  # noqa: E731
        out[name] = {
            "steps": int(len(y)), "violations": int(y.sum()),
            "premise_auroc_h": round(float(roc_auc_score(y, p1)), 4), "premise_auroc_h_plus_residual": round(float(roc_auc_score(y, p2)), 4),
            "residual_share_of_score_variance": round(float(np.var(e) / np.var(s)), 3),
            "pairwise_logistic_risk": {"true": round(pair(s), 4), "replica": round(pair(h), 4)},
            "calibrated_log_loss": {"true": round(float(log_loss(y, cal(s))), 4), "replica": round(float(log_loss(y, cal(h))), 4)},
            "auroc_pooled": {"true": round(float(roc_auc_score(y, s)), 4), "replica": round(float(roc_auc_score(y, h)), 4)},
            "auroc_within_template": {"true": round(float(within_auroc(y, s, tpl)), 4), "replica": round(float(within_auroc(y, h, tpl)), 4)},
        }
    (ROOT / "results/t1/rao_blackwell_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
