#!/usr/bin/env python
"""Tool check: how the cross-validation grouping and the privacy filter change the text monitor's held-out AUROC.
On the website's GPT example (site/dist/data/example_gpt_openai.jsonl, built by site/build.py; also on the public site),
folds hold out conversations, tasks (the first user request; the trainers' default) or task families (the id's first
two parts, 24 templates), with and without the privacy filter (the default).
python scripts/113_grouping_privacy.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.monitor import TextMonitor  # noqa: E402
from graybox.train_reader import load_steps  # noqa: E402


def _best(tm):
    return tm.cv_auroc[tm.C]


def main():
    st = load_steps(str(ROOT / "site/dist/data/example_gpt_openai.jsonl"))
    y = np.array([int(s["label"]) for s in st])
    fam = [".".join(s["conversation"].split(".")[:2]) for s in st]
    G = {"conversation": [s["conversation"] for s in st], "task": [s["group"] for s in st], "family": fam}
    res = {"actions": int(len(y)), "violations": int(y.sum()), "groups": {k: len(set(v)) for k, v in G.items()}, "cv_auroc": {}}
    for k, g in G.items():
        res["cv_auroc"][k] = {("privacy" if p else "no_privacy"): round(_best(TextMonitor().fit(st, y, g, privacy=p)), 4) for p in (True, False)}
    (ROOT / "results/tool/grouping_privacy.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
