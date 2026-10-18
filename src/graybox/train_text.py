"""Train the text monitor (CPU, seconds) on your own labelled actions, without a browser.

Input: OpenAI or Anthropic conversations with a "labels" field ({call_id: 0/1}), or labelled steps (JSONL), the same
files as the browser trainer and graybox.train_reader. Output: a monitor file (graybox-text-monitor/1) for
TextMonitor.load_json, Guard and graybox.monitor_server (GB_TEXT_MONITOR=...).

    python -m graybox.train_text labelled.jsonl --out greybox-text-monitor.json [--chat] [--budget 0.05]

Same model as the browser trainer: TF-IDF of context and action (word 1-2-grams), logistic regression with C from
{0.1, 1, 10} by 5-fold cross-validation grouped by conversation. The escalation cut is the top `budget` of the scores
of your actions, set without labels. It reports the cross-validated AUROC and the share of violations escalated at the
budget. Reliable on tools and policies like the ones you labelled; for new kinds of tasks use the reader monitor.
"""
from __future__ import annotations

import argparse
import json

from .train_reader import load_steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", help="labelled conversations (with a 'labels' field) or labelled steps, JSON lines")
    ap.add_argument("--out", default="greybox-text-monitor.json", help="the monitor file to write")
    ap.add_argument("--chat", action="store_true", help="conversation-aware context: keep every user turn (agents that talk with a user)")
    ap.add_argument("--budget", type=float, default=0.05, help="share of actions to escalate to review (default 0.05)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the linear algebra (default 4)")
    a = ap.parse_args(argv)
    import numpy as np
    from .monitor import TextMonitor
    steps = load_steps(a.data, chat=a.chat)
    y = np.array([int(bool(s["label"])) for s in steps])
    if y.sum() < 2 or (1 - y).sum() < 2:
        raise SystemExit("need at least 2 violations and 2 safe actions")
    groups = [str(s.get("group") or s.get("task") or i) for i, s in enumerate(steps)]
    try:                                     # keep a shared machine responsive: BLAS threads capped
        from threadpoolctl import threadpool_limits
        threadpool_limits(a.threads)
    except ImportError:
        pass
    tm = TextMonitor().fit(steps, y, groups, budget=a.budget, chat=a.chat)
    report = {"actions": int(len(y)), "violations": int(y.sum()), "groups": len(set(groups)), "C": tm.C}
    if tm.cv_auroc:
        cut = float(np.quantile(tm.oof, 1 - a.budget))
        report.update(cv_auroc=round(tm.cv_auroc[tm.C], 4), cv_auroc_by_C={str(k): round(v, 4) for k, v in tm.cv_auroc.items()},
                      escalated_at_budget=round(float((tm.oof[y == 1] >= cut).mean()), 4), budget=a.budget,
                      cut="the saved cut: the top budget of the out-of-fold scores (as this estimate). On a NEW kind of task the text "
                          "monitor escalates far less; set the cut on live traffic there (QuantileGate)")
    tm._json["trained_on"] = report
    tm.save(a.out)
    print(json.dumps(report, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
