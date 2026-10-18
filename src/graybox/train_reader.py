"""Train a reader monitor (GPU) on your own labelled actions.

Input: the labelled actions downloaded from the browser trainer (steps JSONL), or OpenAI/Anthropic logs with a
"labels" field ({call_id: 0/1}). Output: a probe file for ReaderMonitor, Guard and graybox.monitor_server.

    python -m graybox.train_reader labelled.jsonl --out my_reader_probe.joblib

The probe is the shipped pipeline (standardize -> PCA -> logistic, C = 0.05) on the reader's layer 32 (Qwen3-8B).
It reports an AUROC from 5-fold cross-validation grouped by task (the first user message; a steps file's own group), and the catch rate when 5% of
actions go to review. Needs a GPU, torch, transformers and scikit-learn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .guard import conversation_steps, parse_label
from .monitor import Q, act_text, ctx_text, Step


def load_steps(path: str, chat: bool = False, labelled_only: bool = True) -> list[dict]:
    """Labelled steps from a steps JSONL or from conversations with a "labels" field. Unlabelled steps are dropped,
    unless labelled_only=False (graybox.recalibrate scores every tool call of unlabelled logs).
    chat: build conversation-aware contexts (every user turn and agent message; graybox.chat) from conversations."""
    text = open(path, encoding="utf-8").read()
    if text.lstrip().startswith("["):                 # a .json file holding one array of conversations or steps
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON: {e}") from None
    else:
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"line {n} is not valid JSON ({e.msg}); expected one conversation or step per line") from None
    steps, n_ids, n_calls = [], 0, 0
    for i, r in enumerate(rows):
        if "action" in r or "context" in r:
            steps.append({**r, "label": parse_label(r.get("label"))})      # strict: "0" is safe, "yes" is dropped
        else:
            n_ids += len(r.get("labels") or {}) if isinstance(r, dict) else 0
            conv = conversation_steps(r.get("messages", []) if isinstance(r, dict) else r, system=r.get("system") if isinstance(r, dict) else None,
                                      labels=r.get("labels") if isinstance(r, dict) else None, chat=chat)
            n_calls += len(conv)
            for s in conv:
                cid = str(r.get("id", f"conversation-{i + 1}")) if isinstance(r, dict) else f"conversation-{i + 1}"
                # cross-validation groups: the task (first user message), so conversations of one task share a fold
                # (grouping by conversation was optimistic: 0.990 vs 0.946 on the GPT example; scripts/113); the browser does the same
                s["conversation"], s["group"] = cid, ("task: " + s["task"]) if s.get("task") else cid
                for k, v in r.items():                # other top-level fields of the conversation (e.g. "family"), for --group-by
                    if k not in ("messages", "labels", "system", "id") and k not in s and isinstance(v, (str, int, float, bool)):
                        s[k] = v
                steps.append(s)
    if not labelled_only:
        return steps
    out = [s for s in steps if s.get("label") in (0, 1, True, False)]
    if not out and n_ids:
        raise ValueError(f"the 'labels' fields name {n_ids} call ids, but none of the {n_calls} tool calls has a matching id "
                         "(labels are {call_id: 0 or 1}, with the ids of tool_calls / tool_use blocks)")
    return out


def prompt_of(s: dict) -> str:
    if isinstance(s.get("context"), str):            # a prebuilt context; the action is text or {tool, args}
        a = s["action"]
        return Q.format(ctx=s["context"], cot="", act=a if isinstance(a, str) else act_text(a["tool"], a.get("args")))
    hist = [Step(h["tool"], h.get("args") or {}, h.get("result", "")) for h in s.get("history") or []]
    return Q.format(ctx=ctx_text(s.get("instructions", ""), s.get("task", ""), hist), cot="", act=act_text(s["action"]["tool"], s["action"].get("args")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] + " Needs a GPU (about 90 s for 1,500 actions).")
    ap.add_argument("data", help="labelled conversations (OpenAI or Anthropic, with a 'labels' field {call_id: 0/1}) or labelled steps, JSON lines")
    ap.add_argument("--out", default="my_reader_probe.joblib", help="the probe file to write (load with graybox.reader_load.load_reader)")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    ap.add_argument("--layer", type=int, default=32, help="reader layer (hidden-state index; 32 = the tested setting for Qwen3-8B)")
    ap.add_argument("--reader", default="Qwen/Qwen3-8B", help="Hugging Face id of the reader model (downloaded once; needs a GPU)")
    ap.add_argument("--chat", action="store_true", help="conversation-aware context (agents that talk with a user); "
                    "then score with Guard(..., chat=True) or the server's 'context' field")
    ap.add_argument("--canonical", action="store_true", help="train on the canonical action (graybox.monitor.canonical_args): free "
                    "text that the agent writes about its own call is removed, so rewording it cannot move the score; Guard and "
                    "the server then read actions the same way")
    a = ap.parse_args(argv)
    from pathlib import Path
    if Path(a.out).exists() and not a.force:
        raise SystemExit(f"{a.out} exists: pass --force to overwrite it, or another --out")
    if not Path(a.out).resolve().parent.is_dir():
        raise SystemExit(f"the folder of --out does not exist: {Path(a.out).resolve().parent}")
    import joblib
    import numpy as np
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as e:
        raise SystemExit(f"the reader needs the [reader] extra ({e.name} is missing): pip install \"greybox-monitor[reader]\" (and a GPU)")
    from .reader_load import make_reader
    from .stats import caught_at_budget, group_folds
    try:
        steps = load_steps(a.data, chat=a.chat)
    except (ValueError, OSError) as e:
        raise SystemExit(f"{a.data}: {e}")
    if a.canonical:
        from .guard import canonical_step
        steps = [canonical_step(s) for s in steps]
    y = np.array([int(bool(s["label"])) for s in steps])
    if y.sum() < 2 or (1 - y).sum() < 2:
        raise SystemExit("need at least 2 violations and 2 safe actions")
    groups = np.array([str(s.get("group") or s.get("task") or i) for i, s in enumerate(steps)])
    mon = make_reader(probe=None, layer=a.layer, reader=a.reader)      # the standard Hugging Face cache
    X = np.vstack([mon.features(prompt_of(s)) for s in steps])
    mk = lambda n: make_pipeline(StandardScaler(), PCA(min(256, n - 1), random_state=0), LogisticRegression(C=0.05, max_iter=4000))  # noqa: E731
    k, gl = min(5, len(set(groups))), [str(g) for g in groups]
    if len({g for g, v in zip(gl, y) if v}) < 2 or len({g for g, v in zip(gl, y) if not v}) < 2:
        raise SystemExit("cross-validation needs violations in at least 2 groups (tasks) and safe actions in at least 2")
    aucs, caught = [], []
    for rep in range(3):                          # repeated grouped CV, the same folds as graybox.train_text and the browser
        folds, oof = group_folds(gl, k, rep, y), np.zeros(len(y))
        for f in range(k):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            oof[te] = mk(len(tr)).fit(X[tr], y[tr]).decision_function(X[te])
        aucs.append(float(roc_auc_score(y, oof)))
        caught.append(caught_at_budget(y, oof, 0.05))
    report = {"actions": int(len(y)), "violations": int(y.sum()), "groups": int(len(set(groups))), "folds": k, "repeats": 3,
              "cv_auroc": round(float(np.mean(aucs)), 4), "cv_auroc_range": [round(min(aucs), 4), round(max(aucs), 4)],
              "caught_at_5pct": round(float(np.mean(caught)), 4), "most_possible_at_5pct": round(min(1.0, 0.05 * len(y) / y.sum()), 4),
              "canonical": a.canonical}
    probe = mk(len(y)).fit(X, y)
    joblib.dump({"probe": probe, "layer": a.layer, "reader": a.reader, "chat": a.chat, "canonical": a.canonical, "trained_on": f"{a.data}: {report}"}, a.out)
    print(json.dumps(report, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
