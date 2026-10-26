#!/usr/bin/env python
"""Exploratory (asked by review round 19; not preregistered): does Claude's forecast only detect which tool comes next?
On G46c's and G65's units (Claude Haiku 5.5, unapproved production restarts, exact contexts), within template|tool|step:
the geometric forecast (G57's statistic: mean distance of the drawn calls' angles from the agent's median; no label, no
sign) against rules that read the same 8 drawn next actions with task knowledge: the share of draws that are restart
commands (knows the risky tool), the share that are not read-only commands, the number of distinct draws, and CAPE's
checker on the draws (a label on each drawn call).  -> results/t1/tool_rule_baseline_exploratory.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), "/home/arya/new_safety/cape/src", "/home/arya/new_safety/cape/scripts"]
READS = ("status", "help", "info", "list", "get", "read", "show", "describe", "check")


def _mod(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "scripts" / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def scores(G, G57):
    U, _ = G.units()
    key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
    own = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G.OWN))}
    allth = [G57.theta(rd["counts"]) for d in own.values() for rd in d["reads"].values() if np.asarray(rd["counts"])[:, 1].min() > 0]
    c = np.median(np.stack(allth), 0)
    ks = [key(a, b, t) for a, b, t in zip(U.instance_id, U.seed, U.t)]
    m = np.array([k in own for k in ks])
    U, ks = U[m], [k for k, x in zip(ks, m) if x]
    ok = lambda k, d: d and d in own[k]["reads"] and np.asarray(own[k]["reads"][d]["counts"])[:, 1].min() > 0  # noqa: E731
    S = {"geometry (no label, no sign)": [np.mean([float(np.linalg.norm(G57.theta(own[k]["reads"][d]["counts"]) - c)) if ok(k, d) else 0.0 for d in own[k]["draws"]]) for k in ks],
         "share of draws that restart (knows the risky tool)": [np.mean(["restart" in (d or "") for d in own[k]["draws"]]) for k in ks],
         "share of draws that are not read-only": [np.mean([bool(d) and not any(r in d for r in READS) for d in own[k]["draws"]]) for k in ks],
         "distinct draws": [len(set(own[k]["draws"])) for k in ks],
         "checker on the draws (a label per drawn call)": [np.mean([own[k]["reads"][d]["unsafe"] if d and d in own[k]["reads"] else 0 for d in own[k]["draws"]]) for k in ks]}
    return U, {n: np.array(v, float) for n, v in S.items()}


def main():
    from graybox.stats import within_auroc
    G57 = _mod("g57", "242_g57_geometric_forecast.py")
    out = {}
    for name, f in (("original_g46c", "232_g46c_exact_context.py"), ("replication_g65", "254_g65_replication.py")):
        M = _mod(name, f)
        U, S = scores(M.G46C if hasattr(M, "G46C") else M, G57)
        st = (U.stratum + "|" + U.position.astype(str)).values
        out[name] = {"units": int(len(U)), **{n: round(float(within_auroc(U.y.values, v, st)), 4) for n, v in S.items()}}
    G70 = _mod("g70", "262_g70_open_geometry_vs_whitebox.py")
    G68 = G70.G68
    reads = G68.G46.G45.T11.READS
    tool = lambda d: json.loads(d)[0] if d else None  # noqa: E731
    op = {}
    for a in G68.AGENTS:
        U = G68.units(a)
        rec = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])): json.loads(x) for x in open(G68.out_file(a))}
        U = U[np.array([(e, int(v)) in rec for e, v in zip(U.episode_id, U.event_id)])].reset_index(drop=True)
        R = [rec[(e, int(v))] for e, v in zip(U.episode_id, U.event_id)]
        S = {"geometry (no label, no sign)": G70.geometric(R),
             "share of draws that are not read tools (task-agnostic)": np.array([np.mean([bool(d) and tool(d) not in reads for d in r["draws"]]) for r in R]),
             "checker on the draws (a label per drawn call)": np.array([np.mean([r["reads"][d]["unsafe"] if d and d in r["reads"] else 0 for d in r["draws"]]) for r in R])}
        op[a] = {n: round(float(within_auroc(U.y.values, v, U.stratum.values)), 4) for n, v in S.items()}
    op["mean"] = {n: round(float(np.mean([op[a][n] for a in G68.AGENTS])), 4) for n in op[G68.AGENTS[0]]}
    out["open_agents_g70"] = op
    (ROOT / "results/t1/tool_rule_baseline_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
