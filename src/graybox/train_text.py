"""Train the text monitor (CPU, seconds) on your own labelled actions, without a browser.

Input: OpenAI or Anthropic conversations with a "labels" field ({call_id: 0/1}), or labelled steps (JSONL), the same
files as the browser trainer and graybox.train_reader. Output: a monitor file (graybox-text-monitor/1) for
TextMonitor.load_json, Guard and graybox.monitor_server (GB_TEXT_MONITOR=...).

    python -m graybox.train_text labelled.jsonl --out greybox-text-monitor.json [--chat] [--budget 0.05]

The same model type, privacy filter and file format as the browser trainer (the two solvers give weights within about 0.1, and can pick a
different C; a saved file scores identically in both): TF-IDF of context and action (word 1-2-grams), logistic regression with C from
{0.1, 1, 10} by 5-fold cross-validation grouped by task (the first user message; a steps file's own group). The escalation cut is the top `budget` of the scores
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
    ap.add_argument("--group-by", default=None, help="step field that groups the cross-validation folds (default: 'group', which is the task for conversations "
                    "and a steps file's own field, then 'task'). Every step must have it")
    ap.add_argument("--eval-only", action="store_true", help="print the cross-validated report and write no file")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    ap.add_argument("--no-privacy", action="store_true", help="turn the privacy filter off. By default emails and URLs are masked and "
                    "terms with digits, long codes or rare words are dropped; names that recur in your logs stay in the file")
    a = ap.parse_args(argv)
    import sys
    from pathlib import Path
    if not 0 < a.budget < 1:
        raise SystemExit(f"--budget is a share of actions between 0 and 1 (0.05 = 5%), not {a.budget}")
    if not a.eval_only and Path(a.out).exists() and not a.force:
        raise SystemExit(f"{a.out} exists: pass --force to overwrite it, another --out, or --eval-only")
    if not a.eval_only and not Path(a.out).resolve().parent.is_dir():
        raise SystemExit(f"the folder of --out does not exist: {Path(a.out).resolve().parent}")
    try:
        import numpy as np
        from .monitor import TextMonitor
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(f"training needs the [text] extra ({e.name} is missing): pip install \"greybox-monitor[text]\"")
    try:
        steps = load_steps(a.data, chat=a.chat)
    except (ValueError, OSError) as e:
        raise SystemExit(f"{a.data}: {e}")
    y = np.array([int(bool(s["label"])) for s in steps])
    if y.sum() < 2 or (1 - y).sum() < 2:
        raise SystemExit(f"need at least 2 violations and 2 safe actions; found {int(y.sum())} and {int((1 - y).sum())} labelled actions"
                         + " (labels are 0/1 per call id in a 'labels' field, or a 'label' per step)")
    if a.group_by:
        miss = [i for i, s in enumerate(steps) if s.get(a.group_by) in (None, "")]
        if miss:
            raise SystemExit(f"--group-by {a.group_by}: {len(miss)} of {len(steps)} steps have no field {a.group_by!r} (first: step {miss[0] + 1}); "
                             f"fields of step 1: {sorted(steps[0])}")
        groups = [str(s.get(a.group_by)) for s in steps]
    else:
        groups = [str(s.get("group") or s.get("task") or i) for i, s in enumerate(steps)]
    try:                                     # keep a shared machine responsive: BLAS threads capped DURING the fit
        from threadpoolctl import threadpool_limits
    except ImportError:
        import contextlib
        threadpool_limits = lambda n: contextlib.nullcontext()  # noqa: E731
    with threadpool_limits(a.threads):
        try:
            tm = TextMonitor().fit(steps, y, groups, budget=a.budget, chat=a.chat, privacy=not a.no_privacy)
        except ValueError as e:
            raise SystemExit(str(e))
    report = {"actions": int(len(y)), "violations": int(y.sum()), "groups": len(set(groups)),
              "grouped_by": a.group_by or "task (first user message) for conversations; 'group' for labelled steps", "C": tm.C,
              "privacy_filter": not a.no_privacy}
    if tm.cv_auroc:
        from .stats import caught_at_budget, within_auroc
        tools = [_tool(s) for s in steps]
        wt = float(np.mean([within_auroc(y, o, tools) for o in tm.oof_reps]))          # mean over the 3 fold assignments
        lo, hi = tm.cv_auroc_range[tm.C]
        report.update(cv_auroc=round(tm.cv_auroc[tm.C], 4), cv_auroc_range=[round(lo, 4), round(hi, 4)],
                      cv_auroc_by_C={str(k): round(v, 4) for k, v in tm.cv_auroc.items()},
                      cv_auroc_within_tool=None if np.isnan(wt) else round(float(wt), 4),
                      cv_caught_at_budget=round(float(np.mean([caught_at_budget(y, o, a.budget) for o in tm.oof_reps])), 4), budget=a.budget,
                      most_possible_at_budget=round(min(1.0, a.budget * len(y) / y.sum()), 4),
                      estimate=f"cross-validated over {len(set(groups))} groups, mean of 3 fold assignments (range above): each score comes "
                               "from a model that did not see its group (C is the best of 3 on the same folds: slightly optimistic)",
                      cut=tm._json["cut_source"] + ". On a NEW kind of task any stored cut escalates far less; set the cut on live "
                          "traffic there (QuantileGate)")
        if not np.isnan(wt) and tm.cv_auroc[tm.C] - wt > 0.1:
            report["warning"] = ("most of the AUROC comes from WHICH tool is called, not from the call itself (within-tool "
                                 f"{wt:.3f} vs pooled {tm.cv_auroc[tm.C]:.3f}): on unseen groups it does not separate safe from unsafe calls of the "
                                 "same tool. Do not rely on it for that; check such rules in code or review the calls")
        report["note"] = ("word counts memorize entities (ids, names, amounts) of the labelled tasks and do not check numeric, date or "
                          "recipient rules on unseen values. A new KIND of task can score far lower than this estimate: to measure it, "
                          "hold out a task family with --group-by <field>")
    if a.eval_only:
        print(json.dumps(report, indent=1))
        print("not saved (--eval-only)", file=sys.stderr)
        return
    tm._json["trained_on"] = report
    tm.save(a.out)
    print(json.dumps(report, indent=1))
    print("saved", a.out, file=sys.stderr)


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
