#!/usr/bin/env python
"""G67 (docs/prereg_G67_matched_dose.md): G47's reviewer-free prevention on FRESH seeds 320-331 with the random arm's dose
matched. In G47 the random arm held each non-read call with probability 0.22, but identical re-proposals of a held call
count as calls and cannot be held again, so it held 0.177 calls per call against the self arm's 0.227. From G47's random
arm, holds per call = r / (1 + r*rho) with rho = 1.094; matching 0.227 gives r = 0.30. scripts/225's episode, run and
analysis code are unchanged; only RATE, the seeds and the output file differ.
  stage=run cap=.. ; stage=analyze -> results/t1/g67_matched_dose.json
"""
import importlib.util
import json
import sys
from math import erf, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS, RATE, DOSE_BAND = "320-331", 0.30, 0.03


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


P = _mod("p225", "scripts/225_selfgate_pilot_exploratory.py")
FN = P.OUT / "g67.jsonl"
P.cut_orig = P.cut
P.cut = lambda: (P.cut_orig()[0], RATE)        # CUT from development answers (unchanged); RATE fixed above


def stage_analyze():
    out = P.stage_analyze(FN)
    hp = out["holds_per_call"]
    Pn = lambda z: 0.5 * (1 - erf(z / sqrt(2)))  # noqa: E731
    rows = {"Q1_self_minus_random": (out["self_minus_random"], 0.0), "Q3_self_reduction": (out["self_reduction"], 0.0),
            "Q4_success_self_minus_none": (out["success_self_minus_none"], -0.05)}
    ps = {}
    for k, (v, t) in rows.items():
        se = (v[2] - v[1]) / 3.92
        ps[k] = Pn((v[0] - t) / se) if se > 0 else 0.0
    order, go, holm = sorted(ps, key=ps.get), True, {}
    for j, k in enumerate(order):
        go = go and ps[k] <= 0.05 / (len(order) - j)
        holm[k] = bool(go and rows[k][0][1] > rows[k][1])
    out["endpoints"] = {"Q1_self_minus_random_lower_gt_0": rows["Q1_self_minus_random"][0][1] > 0,
                        "Q2_dose_matched_within_0.03": abs(hp["random"] - hp["self"]) <= DOSE_BAND,
                        "Q3_self_reduction_lower_gt_0": rows["Q3_self_reduction"][0][1] > 0,
                        "Q4_success_lower_gt_-0.05": rows["Q4_success_self_minus_none"][0][1] > -0.05}
    out["p_one_sided"], out["holm_pass"] = {k: round(v, 6) for k, v in ps.items()}, holm
    (ROOT / "results/t1/g67_matched_dose.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"endpoints": out["endpoints"], "holm": holm, "holds_per_call": hp}, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    if kw.get("stage") == "run":
        P.stage_run(kw["cap"], kw.get("seeds", SEEDS), "cons_service_restart", P.OUT / kw["out"] if "out" in kw else FN)
    else:
        stage_analyze()
