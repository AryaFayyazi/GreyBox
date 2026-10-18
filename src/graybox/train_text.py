"""Train the text monitor (CPU, seconds) on your own labelled actions, without a browser.

Input: OpenAI or Anthropic conversations with a "labels" field ({call_id: 0/1}), or labelled steps (JSONL), the same
files as the browser trainer and graybox.train_reader. Output: a monitor file (graybox-text-monitor/1) for
TextMonitor.load_json, Guard and graybox.monitor_server (GB_TEXT_MONITOR=...).

    python -m graybox.train_text labelled.jsonl --out greybox-text-monitor.json [--chat] [--budget 0.05]

Same model as the browser trainer: TF-IDF of context and action (word 1-2-grams), logistic regression with C from
{0.1, 1, 10} by 5-fold cross-validation grouped by conversation. The escalation cut is the top `budget` of the scores
of your actions, set without labels. It reports the cross-validated AUROC (pooled and within tool) and the share of
violations caught at the budget. Reliable on tools and policies like the ones you labelled; for new kinds of tasks use the reader monitor.
"""
from __future__ import annotations

import argparse
import json
import re

from .train_reader import load_steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", help="labelled conversations (with a 'labels' field) or labelled steps, JSON lines")
    ap.add_argument("--out", default="greybox-text-monitor.json", help="the monitor file to write")
    ap.add_argument("--chat", action="store_true", help="conversation-aware context: keep every user turn (agents that talk with a user)")
    ap.add_argument("--budget", type=float, default=0.05, help="share of actions to escalate to review (default 0.05)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the linear algebra (default 4)")
    ap.add_argument("--group-by", default=None, help="step field that groups the cross-validation folds (default: 'group', "
                    "then 'task', else one group per step). Group by task to estimate the score on tasks the monitor has not seen")
    a = ap.parse_args(argv)
    import numpy as np
    from .monitor import TextMonitor
    steps = load_steps(a.data, chat=a.chat)
    y = np.array([int(bool(s["label"])) for s in steps])
    if y.sum() < 2 or (1 - y).sum() < 2:
        raise SystemExit("need at least 2 violations and 2 safe actions")
    if a.group_by:
        groups = [str(s.get(a.group_by)) if s.get(a.group_by) is not None else f"_{i}" for i, s in enumerate(steps)]
    else:
        groups = [str(s.get("group") or s.get("task") or i) for i, s in enumerate(steps)]
    try:                                     # keep a shared machine responsive: BLAS threads capped DURING the fit
        from threadpoolctl import threadpool_limits
    except ImportError:
        import contextlib
        threadpool_limits = lambda n: contextlib.nullcontext()  # noqa: E731
    with threadpool_limits(a.threads):
        tm = TextMonitor().fit(steps, y, groups, budget=a.budget, chat=a.chat)
    report = {"actions": int(len(y)), "violations": int(y.sum()), "groups": len(set(groups)), "C": tm.C}
    if tm.cv_auroc:
        from .stats import within_auroc
        cut = float(np.quantile(tm.oof, 1 - a.budget))
        tools = [_tool(s) for s in steps]
        wt = within_auroc(y, tm.oof, tools)
        report.update(cv_auroc=round(tm.cv_auroc[tm.C], 4), cv_auroc_by_C={str(k): round(v, 4) for k, v in tm.cv_auroc.items()},
                      cv_auroc_within_tool=None if np.isnan(wt) else round(float(wt), 4),
                      cv_caught_at_budget=round(float((tm.oof[y == 1] >= cut).mean()), 4), budget=a.budget,
                      estimate=f"cross-validated over {len(set(groups))} groups: each score comes from a model that did not see its group",
                      cut=tm._json["cut_source"] + ". On a NEW kind of task any stored cut escalates far less; set the cut on live "
                          "traffic there (QuantileGate)")
        if not np.isnan(wt) and tm.cv_auroc[tm.C] - wt > 0.1:
            report["warning"] = ("most of the AUROC comes from WHICH tool is called, not from the call itself (within-tool "
                                 f"{wt:.3f} vs pooled {tm.cv_auroc[tm.C]:.3f}): on unseen groups it does not separate safe from unsafe calls of the "
                                 "same tool. Do not rely on it for that; check such rules in code or review the calls")
        report["note"] = ("word counts memorize entities (ids, names, amounts) of the labelled tasks and do not check numeric, date or "
                          "recipient rules on unseen values. Use --group-by task to estimate the score on unseen tasks")
    tm._json["trained_on"] = report
    tm.save(a.out)
    print(json.dumps(report, indent=1))
    print("saved", a.out)


def _tool(s) -> str:
    a = s.get("action") if isinstance(s, dict) else s[3]
    if isinstance(a, dict):
        return str(a.get("tool"))
    if isinstance(a, (tuple, list)):
        return str(a[0])
    m = re.match(r"\s*([\w.\-]+)", str(a))           # rendered "tool {args}" or "tool(args)"
    return m.group(1) if m else ""


def cli():
    """Command-line entry: BLAS thread pools are sized when numpy loads, so --threads is applied by restarting the process
    once with OMP/OPENBLAS/MKL_NUM_THREADS set (a no-op when they already match)."""
    import os
    import sys
    n = "4"
    for i, x in enumerate(sys.argv):
        if x == "--threads" and i + 1 < len(sys.argv):
            n = sys.argv[i + 1]
        elif x.startswith("--threads="):
            n = x.split("=", 1)[1]
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    if os.environ.get("GB_THREADS_SET") != n:
        os.environ.update({k: n for k in keys}, GB_THREADS_SET=n)
        os.execv(sys.executable, [sys.executable, "-m", "graybox.train_text", *sys.argv[1:]])
    main()


if __name__ == "__main__":
    cli()
