"""Train a reader monitor (GPU) on your own labelled actions.

Input: the labelled actions downloaded from the browser trainer (steps JSONL), or OpenAI/Anthropic logs with a
"labels" field ({call_id: 0/1}). Output: a probe file for ReaderMonitor, Guard and graybox.monitor_server.

    python -m graybox.train_reader labelled.jsonl --out my_reader_probe.joblib

The probe is the shipped pipeline (standardize -> PCA -> logistic, C = 0.05) on the reader's layer 32 (Qwen3-8B).
It reports an AUROC from 5-fold cross-validation grouped by conversation (or task), and the catch rate when 5% of
actions go to review. Needs a GPU, torch, transformers and scikit-learn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .guard import conversation_steps
from .monitor import Q, act_text, ctx_text, Step


def load_steps(path: str, chat: bool = False) -> list[dict]:
    """Labelled steps from a steps JSONL or from conversations with a "labels" field. Unlabelled steps are dropped.
    chat: build conversation-aware contexts (every user turn and agent message; graybox.chat) from conversations."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    steps = []
    for i, r in enumerate(rows):
        if "action" in r or "context" in r:
            steps.append(r)
        else:
            for s in conversation_steps(r.get("messages", r if isinstance(r, list) else []), system=r.get("system"), labels=r.get("labels"), chat=chat):
                s["group"] = str(r.get("id", f"conversation-{i + 1}"))
                steps.append(s)
    return [s for s in steps if s.get("label") in (0, 1, True, False)]


def prompt_of(s: dict) -> str:
    if isinstance(s.get("context"), str):            # a prebuilt context; the action is text or {tool, args}
        a = s["action"]
        return Q.format(ctx=s["context"], cot="", act=a if isinstance(a, str) else act_text(a["tool"], a.get("args")))
    hist = [Step(h["tool"], h.get("args") or {}, h.get("result", "")) for h in s.get("history") or []]
    return Q.format(ctx=ctx_text(s.get("instructions", ""), s.get("task", ""), hist), cot="", act=act_text(s["action"]["tool"], s["action"].get("args")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data")
    ap.add_argument("--out", default="my_reader_probe.joblib")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--reader", default="Qwen/Qwen3-8B")
    ap.add_argument("--chat", action="store_true", help="conversation-aware context (agents that talk with a user); "
                    "then score with Guard(..., chat=True) or the server's 'context' field")
    a = ap.parse_args(argv)
    import joblib
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .monitor import ReaderMonitor
    steps = load_steps(a.data, chat=a.chat)
    y = np.array([int(bool(s["label"])) for s in steps])
    if y.sum() < 2 or (1 - y).sum() < 2:
        raise SystemExit("need at least 2 violations and 2 safe actions")
    groups = np.array([str(s.get("group") or s.get("task") or i) for i, s in enumerate(steps)])
    mon = ReaderMonitor(probe=None, layer=a.layer, reader=a.reader)
    X = np.vstack([mon.features(prompt_of(s)) for s in steps])
    mk = lambda n: make_pipeline(StandardScaler(), PCA(min(256, n - 1), random_state=0), LogisticRegression(C=0.05, max_iter=4000))  # noqa: E731
    k = min(5, len(set(groups)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(k).split(X, y, groups):
        oof[te] = mk(len(tr)).fit(X[tr], y[tr]).decision_function(X[te])
    cut = np.quantile(oof, 0.95)
    report = {"actions": int(len(y)), "violations": int(y.sum()), "groups": int(len(set(groups))), "folds": k,
              "cv_auroc": round(float(roc_auc_score(y, oof)), 4), "caught_at_5pct": round(float((oof[y == 1] >= cut).mean()), 4)}
    probe = mk(len(y)).fit(X, y)
    joblib.dump({"probe": probe, "layer": a.layer, "reader": a.reader, "chat": a.chat, "trained_on": f"{a.data}: {report}"}, a.out)
    print(json.dumps(report, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
