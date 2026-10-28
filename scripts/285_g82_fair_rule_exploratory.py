#!/usr/bin/env python
"""Exploratory (review round 22): G82's task-agnostic rule (share of draws that are not CAPE read tools) is below chance on
three agents, so geometry - rule partly beats a rule pointing the wrong way, while the geometry has no sign. Two fairer
rules on G82's units and draws (unchanged): (a) sign-free, like the geometry: |share - the agent's median share| (no label);
(b) the rule in its better direction per agent, chosen after seeing the labels (generous to the rule). Mean over the four
agents of geometry - rule, cluster bootstrap over scenarios (2,000, seed 8201, as G82).
-> results/t1/g82_fair_rule_exploratory.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("g82f", ROOT / "scripts/281_g82_open_geometry_vs_rules.py")
G82 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(G82)


def main():
    from graybox.stats import within_auroc
    G70 = G82._mod("g70_f", "scripts/262_g70_open_geometry_vs_whitebox.py")
    reads = G82.G68.G46.G45.T11.READS
    tool = lambda d: json.loads(d)[0] if d else None  # noqa: E731
    data, per = {}, {}
    for a in G82.AGENTS:
        U = G82.units(a)
        rec = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])): json.loads(x) for x in open(G82.out_file(a))}
        U = U[np.array([(e, int(v)) in rec for e, v in zip(U.episode_id, U.event_id)])].reset_index(drop=True)
        R = [rec[(e, int(v))] for e, v in zip(U.episode_id, U.event_id)]
        rule = np.array([np.mean([bool(d) and tool(d) not in reads for d in r["draws"]]) for r in R])
        D = {"y": U.y.values, "st": U.stratum.values, "cl": U.scenario_id.values, "geo": G70.geometric(R), "rule": rule,
             "rule_signfree": np.abs(rule - np.median(rule))}
        W = lambda s, D=D: within_auroc(D["y"], D[s], D["st"])  # noqa: E731
        r = W("rule")
        D["sign"] = 1 if r >= 0.5 else -1
        data[a] = D
        per[a] = {"geo": round(W("geo"), 4), "rule_registered": round(r, 4), "rule_signfree": round(W("rule_signfree"), 4),
                  "rule_best_direction": round(max(r, 1 - r), 4)}
        print(a, per[a], flush=True)
    rng = np.random.default_rng(8201)
    B = []
    for _ in range(2000):
        x, y_ = [], []
        for D in data.values():
            us = np.unique(D["cl"])
            i = np.concatenate([np.flatnonzero(D["cl"] == u) for u in rng.choice(us, len(us))])
            g = within_auroc(D["y"][i], D["geo"][i], D["st"][i])
            x.append(g - within_auroc(D["y"][i], D["rule_signfree"][i], D["st"][i]))
            y_.append(g - within_auroc(D["y"][i], D["sign"] * D["rule"][i], D["st"][i]))
        B.append((np.nanmean(x), np.nanmean(y_)))
    B = np.array(B)
    ci = lambda c: [round(float(np.nanpercentile(B[:, c], 2.5)), 4), round(float(np.nanpercentile(B[:, c], 97.5)), 4)]  # noqa: E731
    m = lambda k: float(np.mean([per[a][k] for a in per]))  # noqa: E731
    out = {"per_agent": per, "mean": {k: round(m(k), 4) for k in ("geo", "rule_registered", "rule_signfree", "rule_best_direction")},
           "geo_minus_signfree_rule": [round(m("geo") - m("rule_signfree"), 4)] + ci(0),
           "geo_minus_best_direction_rule": [round(m("geo") - m("rule_best_direction"), 4)] + ci(1)}
    (ROOT / "results/t1/g82_fair_rule_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
