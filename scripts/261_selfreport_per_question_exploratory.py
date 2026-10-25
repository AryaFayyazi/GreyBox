#!/usr/bin/env python
"""Exploratory robustness check (no claim registered): is Claude Haiku 5.5's self-report near chance only because of the
fixed read-out? (a) Each of the 8 pre-call questions alone (its answer angle times its fixed sign, G44's convention), AUROC
within template|tool|position, on G46c's units (original) and on G65's fresh units. The best of 8, picked after the
fact, favours the self-report, so this check can only weaken the says-vs-does claim. (b) Descriptive: the share of
answers that point to a violation (counts[0] = "no" answers; sign +1 means "no" points to a violation), on violating and
on safe calls, and the share of violating calls before which no answer to any question points to a violation.
  -> results/t1/selfreport_per_question_exploratory.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    from graybox.stats import within_auroc
    REP = _mod("g65", "scripts/254_g65_replication.py")
    G44 = REP.G44
    out = {}
    for name, units, rf, own in (("original", _mod("g46c", "scripts/232_g46c_exact_context.py").units, REP.OUT / "g59_selfreport_exact.jsonl", REP.OUT / "g46c_own.jsonl"),
                                 ("replication", REP.G46C.units, REP.G59.RF, REP.G46C.OWN)):
        U, _ = units()
        rep = {(d["instance_id"], int(d["seed"]), int(d["t"])): json.loads(d["own"]) if isinstance(d["own"], str) else d["own"]
               for d in (json.loads(x) for x in open(rf))}
        ks = [(a, int(b), int(c)) for a, b, c in zip(U.instance_id, U.seed, U.t)]
        m = np.array([k in rep for k in ks])
        U, ks = U[m].reset_index(drop=True), [k for k, x in zip(ks, m) if x]
        C = np.array([rep[k] for k in ks], float)
        ang = 2 * np.arcsin(np.sqrt((C[:, :, 0] + 3 / 8) / (C[:, :, 1] + 3 / 4)))
        st = (U.stratum + "|" + U.position.astype(str)).values
        y = U.y.values
        per = [round(float(within_auroc(y, ang[:, j] * G44.SG[j], st)), 4) for j in range(ang.shape[1])]
        no, n = C[:, :, 0], C[:, :, 1]
        vp = np.stack([no[:, j] if G44.SG[j] > 0 else n[:, j] - no[:, j] for j in range(ang.shape[1])], 1)
        v = y.astype(bool)
        sh = lambda r: [round(float(vp[r, j].sum() / n[r, j].sum()), 4) for j in range(ang.shape[1])]  # noqa: E731
        out[name] = {"units": int(len(y)), "violation_rate": round(float(y.mean()), 4), "answers": int(n.sum()),
                     "per_question": per, "best_of_8": max(per), "mean_of_8": round(float(np.mean(per)), 4),
                     "fixed_readout": round(float(within_auroc(y, np.array([G44.risk(c) for c in C]), st)), 4),
                     "share_pointing_to_violation_on_violating_calls": sh(v), "share_pointing_to_violation_on_safe_calls": sh(~v),
                     "violating_calls_with_no_answer_pointing_to_violation": round(float((vp[v].sum(1) == 0).mean()), 4),
                     "safe_calls_with_no_answer_pointing_to_violation": round(float((vp[~v].sum(1) == 0).mean()), 4)}
    (ROOT / "results/t1/selfreport_per_question_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
