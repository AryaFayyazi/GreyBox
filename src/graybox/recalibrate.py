"""Set the review cut again on new, unlabelled traffic: after a change of agent, prompt, tools or tasks.

    python -m graybox.recalibrate new_logs.jsonl --monitor greybox-text-monitor.json --out recalibrated.json [--budget 0.05]

The logs are OpenAI or Anthropic conversations (or steps), as for training, but need no labels. Every tool call is
scored with the monitor, and the new cut is the top `budget` of these scores (the cut that QuantileGate freezes).
The report says how far the old cut has drifted on this traffic: the share of the new actions that it would escalate.
Why: a cut escalates its budget only on traffic like the one it was set on (the paper's frozen-cut and shift
propositions). In a preregistered live test, a notice that the agent is monitored moved Claude Haiku 5.5's scores so that the old
cut escalated 16.6% of its actions; a cut set again on its new, unlabelled traffic escalated 4.7% (budget 5%) and still
blocked every violation.

For the reader monitor (a probe file, GPU), use the server's /v1/monitor/calibrate with past scores, or QuantileGate.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

from .guard import canonical_step
from .monitor import Step, TextMonitor, act_text
from .stats import budget_cut
from .train_reader import load_steps


def load_monitor(path):
    d = json.loads(Path(path).read_text())
    if str(d.get("format", "")).startswith("graybox-embed-monitor"):
        from .embed_monitor import EmbedMonitor
        return EmbedMonitor.load_json(path)
    return TextMonitor.load_json(path)


def score_steps(mon, steps) -> np.ndarray:
    """The monitor's logit for each step (a prebuilt "context" is used as is, as Guard does)."""
    out = []
    canonical = bool((getattr(mon, "_json", None) or {}).get("canonical"))
    for s in steps:
        if canonical:                                   # as Guard: the monitor was trained on canonical actions
            s = canonical_step(s)
        a = s["action"]
        tool, args = (a["tool"], a.get("args") or {}) if isinstance(a, dict) else (None, None)
        if isinstance(s.get("context"), str):
            act = a if isinstance(a, str) else act_text(tool, args)
            out.append(mon.decision(None, None, None, None, rendered=(s["context"], act)))
        else:
            hist = [Step(h["tool"], h.get("args") or {}, h.get("result", "")) for h in s.get("history") or []]
            out.append(mon.decision(s.get("instructions", ""), s.get("task", ""), hist, (tool, args)))
    return np.asarray(out, float)


def frozen_cut_sd(n: int, budget: float) -> float:
    """Standard deviation of the long-run escalation rate of a cut frozen on n i.i.d. scores (Beta(n+1-k, k))."""
    k = math.ceil((1 - budget) * (n + 1))
    return math.sqrt(k * (n + 1 - k) / ((n + 1) ** 2 * (n + 2)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", help="new traffic: conversations or steps, JSON lines (labels are not needed and are ignored)")
    ap.add_argument("--monitor", required=True, help="a monitor file (graybox.train_text, the browser trainer)")
    ap.add_argument("--out", help="the recalibrated monitor file to write (omit for a report only)")
    ap.add_argument("--budget", type=float, help="share of actions to escalate (default: the file's budget, or 0.05)")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    a = ap.parse_args(argv)
    if a.out and Path(a.out).exists() and not a.force:
        raise SystemExit(f"{a.out} exists: pass --force to overwrite it, or another --out")
    mon = load_monitor(a.monitor)
    d = mon._json
    budget = a.budget if a.budget is not None else float(d.get("budget") or 0.05)
    if not 0 < budget < 1:
        raise SystemExit("--budget must be between 0 and 1")
    try:
        steps = load_steps(a.logs, chat=bool(d.get("chat")), labelled_only=False)
    except (ValueError, OSError) as e:
        raise SystemExit(f"{a.logs}: {e}")
    if len(steps) < 100:
        raise SystemExit(f"need at least 100 tool calls to set a cut; the logs have {len(steps)}")
    z = score_steps(mon, steps)
    old = float(d["threshold_logit"])
    new, new_share = budget_cut(z, budget)
    n, sd = len(z), frozen_cut_sd(len(z), budget)
    old_share = float((z >= old).mean())
    shifted = abs(old_share - budget) > max(3 * math.sqrt(budget * (1 - budget) / n), 0.01)
    report = {"actions": n, "budget": budget, "old_cut": round(old, 4), "old_cut_escalates": round(old_share, 4),
              "new_cut": round(new, 4), "new_cut_escalates": round(new_share, 4),
              "expected_long_run_rate": [round(budget, 4), "sd", round(sd, 4)], "shifted": shifted}
    print(json.dumps(report, indent=1))
    if new_share > budget + 2.0 / n:
        print(f"one score is shared by {new_share:.1%} of the actions (identical actions): the cut escalates them all",
              file=sys.stderr)
    if shifted:
        print(f"the old cut would escalate {old_share:.1%} of this traffic instead of {budget:.0%}: the traffic has shifted",
              file=sys.stderr)
    if a.out:
        out = dict(d, threshold_logit=new, budget=budget,
                   calibration={"actions": n, "date": datetime.date.today().isoformat(), "source": Path(a.logs).name,
                                "old_cut": old, "old_cut_escalates": old_share})
        Path(a.out).write_text(json.dumps(out))
        print("saved", a.out, file=sys.stderr)
    return report


def cli():
    main()


if __name__ == "__main__":
    main()
