#!/usr/bin/env python
"""EXPLORATORY (after G38's locked analysis): G38's read-outs against the call's position in the episode. Position = the
call's ordinal among the episode's T11 units. Strata template|tool|position; cluster bootstrap over scenarios (1000, seed 1).
python scripts/211c_g38_position_exploratory.py  -> results/t1/g38_position_exploratory.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

s = importlib.util.spec_from_file_location("g38", ROOT / "scripts/211_g38_gpt_readout.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def main():
    S = pd.read_json(m.GA / "g38.jsonl", orient="records", lines=True)
    U = m.units()
    U["t"] = U.groupby("episode_id").cumcount()
    S = S.merge(U[["episode_id", "event_id", "t"]], on=["episode_id", "event_id"])
    th = lambda k, n: 2 * np.arcsin(np.sqrt((k + 3 / 8) / (n + 3 / 4)))  # noqa: E731
    sc = {}
    for arm, (_, qs) in m.batteries().items():
        sg = np.array([g for _, g in qs])
        C = np.array([[[sum(x for x in a if x is not None), sum(x is not None for x in a)] for a in o[arm]] for o in S.answers], float)
        T = th(C[..., 0], C[..., 1])
        sc[arm] = (T * sg).mean(1) if arm == "AT" else T @ sg
    y, tm, cl = S.y.values, S.template.values, S.scenario_id.values
    st3 = np.array([f"{a}|{b}" for a, b in zip(tm, S.t)])
    res = {"position_alone_within_template_tool": round(float(within_auroc(y, S.t.values.astype(float), tm)), 4)}
    for arm in sc:
        b = cluster_boot(lambda i: within_auroc(y[i], sc[arm][i], st3[i]), cl, 1000, seed=1)
        b = b[np.isfinite(b)]
        res[f"{arm}_within_template_tool_position"] = [round(float(within_auroc(y, sc[arm], st3)), 4),
                                                       round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
    both = [k for k in np.unique(st3) if 0 < y[st3 == k].sum() < (st3 == k).sum()]
    res["strata_with_both_classes"], res["violations_in_them"] = len(both), int(sum(y[st3 == k].sum() for k in both))
    (ROOT / "results/t1/g38_position_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
