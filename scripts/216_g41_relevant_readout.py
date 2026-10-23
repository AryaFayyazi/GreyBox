#!/usr/bin/env python
"""G41: read the state along the directions the call can violate (docs/prereg_G41_relevant_readout.md). At a proposed call
the tool is known, and the airline policy says which rules can apply to it. The relevant read-out averages, for a call
with tool T, the signed angles of the general questions (0: complies; 7: forbidden) and of the rule questions that the
policy ties to T (MAP below, written from the policy text before any per-tool result was computed). Nothing is fit; no
label is used. Data (existing): Claude Haiku 5.5's 360 tau2-bench writes; arms G18_R4 (G18's 4 answers) and D16 (G33).
CPU only.  python scripts/216_g41_relevant_readout.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from graybox.stats import within_auroc  # noqa: E402

# G18's questions: 0 complies (-1); 1 cancellation conditions; 2 basic-economy change or origin/destination/trip type;
# 3 cancelled reservation or flown flight; 4 payment methods; 5 certificate/refund/compensation; 6 bags, insurance after
# booking, passenger count; 7 forbidden.  Policy: cancellations (1, 3, 5: refunds); flight changes (2, 3, 4: payment of the
# fare difference); bookings (4: payment; 5: certificates used to pay); baggage updates (3, 4, 6); passenger updates (3, 6).
MAP = {"cancel_reservation": [1, 3, 5], "update_reservation_flights": [2, 3, 4], "book_reservation": [4, 5],
       "update_reservation_baggages": [3, 4, 6], "update_reservation_passengers": [3, 6]}
GENERAL = [0, 7]
SG = np.array([-1, 1, 1, 1, 1, 1, 1, 1])
B_N, SEED = 2000, 4100


def data():
    s = importlib.util.spec_from_file_location("g33", ROOT / "scripts/205_g33_closed_dither.py")
    G33 = importlib.util.module_from_spec(s)
    s.loader.exec_module(G33)
    S = pd.read_json(G33.OUT / "g33_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    G18 = pd.read_json(G33.OUT / "g18_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    G18 = G18.assign(k=G18.groupby(["sim_id", "msg_idx"]).cumcount())
    J = S.merge(G18[["sim_id", "msg_idx", "k", "counts"]], on=["sim_id", "msg_idx", "k"], how="left")
    arms = {"G18_R4": np.array([np.array(c, float) for c in J.counts]), "D16": G33.counts(list(J.answers), "D16")}
    return J, arms


def readouts(J, C):
    T = 2 * np.arcsin(np.sqrt((C[..., 0] + 3 / 8) / (C[..., 1] + 3 / 4))) * SG
    full = T.mean(1)
    rel = np.array([T[i, GENERAL + MAP.get(t, [])].mean() for i, t in enumerate(J.tool)])
    return full, rel


def main():
    J, arms = data()
    y, st, cl = J.y.values.astype(int), J.tool.values, J.task_id.values
    sc = {}
    for a, C in arms.items():
        sc[f"{a}_full"], sc[f"{a}_relevant"] = readouts(J, C)
    W = lambda k, i: within_auroc(y[i], sc[k][i], st[i])  # noqa: E731
    rng = np.random.default_rng(SEED)
    us = np.unique(cl)
    keys = {"R1_D16_relevant_minus_full": ("D16_relevant", "D16_full"), "R2_G18_relevant_minus_full": ("G18_R4_relevant", "G18_R4_full")}
    B = {k: [] for k in keys}
    Bv = {k: [] for k in sc}
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        for k, (a, b) in keys.items():
            B[k].append(W(a, i) - W(b, i))
        for k in sc:
            Bv[k].append(W(k, i))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    i0 = np.arange(len(y))
    res = {"writes": int(len(y)), "violations": int(y.sum()), "map": MAP, "general": GENERAL,
           "within_tool_auroc": {k: [round(float(W(k, i0)), 4)] + ci(Bv[k]) for k in sc}}
    for k, (a, b) in keys.items():
        res[k] = [round(float(W(a, i0) - W(b, i0)), 4)] + ci(B[k])
    res["endpoints"] = {"R1_lower_gt_0": res["R1_D16_relevant_minus_full"][1] > 0, "R2_lower_gt_0": res["R2_G18_relevant_minus_full"][1] > 0}
    (ROOT / "results/t1/g41_relevant_readout.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
