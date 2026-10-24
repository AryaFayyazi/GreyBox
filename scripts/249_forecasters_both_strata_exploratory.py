#!/usr/bin/env python
"""Exploratory (not preregistered): every forecaster before the call on the SAME units, under both strata used in the
paper: within template|tool (across positions) and within template|tool|position. Claude Haiku 5.5: G46c's units (exact
contexts); GPT-6-luna: G48's units. Position alone (the unit's ordinal) is the baseline that every forecast must beat
across positions. -> results/t1/forecasters_both_strata_exploratory.json"""
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


G46C, G48 = _mod("g46c", "scripts/232_g46c_exact_context.py"), _mod("g48", "scripts/229_g48_gpt_rollout.py")
G46, G44 = G46C.G46, _mod("g44", "scripts/220_g44_self_vs_other_forecast.py")
G52, G54 = _mod("g52", "scripts/237_g52_text_judge_vs_policy.py"), _mod("g54", "scripts/239_g54_gpt_text_judge.py")
G55, G57 = _mod("g55", "scripts/240_g55_gpt_judges_claude.py"), _mod("g57", "scripts/242_g57_geometric_forecast.py")
G58 = _mod("g58", "scripts/243_g58_other_reads_draws.py")
key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
load = lambda f: {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(f))}  # noqa: E731


def geo(own):
    allth = [G57.theta(rd["counts"]) for d in own.values() for rd in d["reads"].values() if np.asarray(rd["counts"])[:, 1].min() > 0]
    centre = np.median(np.stack(allth), 0)

    def fc(d):
        v = [float(np.linalg.norm(G57.theta(d["reads"][dr]["counts"]) - centre))
             if dr and dr in d["reads"] and np.asarray(d["reads"][dr]["counts"])[:, 1].min() > 0 else 0.0 for dr in d["draws"]]
        return float(np.mean(v)) if v else 0.0
    return fc


def table(U, own, scorers):
    from graybox.stats import within_auroc
    ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    out = {}
    st2, st3, y = U.stratum.values, (U.stratum + "|" + U.position.astype(str)).values, U.y.values
    for name, f in scorers.items():
        v = np.array([f(k) for k in ks], float)
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        out[name] = {"within_template_tool": round(float(within_auroc(y, v, st2)), 4),
                     "within_position": round(float(within_auroc(y, v, st3)), 4) if name != "position alone" else None}
    return {"units": int(len(y)), "violations": int(y.sum()), "scores": out}


def jp(d):
    return lambda k: (d[k]["p"] if k in d and d[k].get("p") is not None else np.nan)


def main():
    Uc, _ = G46C.units()
    oc, pos_c = load(G46C.OWN), dict(zip([key(a, b, c) for a, b, c in zip(Uc.instance_id, Uc.seed, Uc.t)], Uc.position))
    rd = load(G58.JF)
    sc = {"position alone": lambda k: pos_c[k]}
    rf = G46C.OUT / "g59_selfreport_exact.jsonl"
    if rf.exists():
        rep = load(rf)
        sc["self-report"] = lambda k: G44.risk(rep[k]["own"]) if k in rep else np.nan
    sc.update({"Claude text judge": jp(load(G52.JF)), "GPT-6-luna text judge": jp(load(G55.JF)),
               "own draws, own read": lambda k: G46.forecast(oc[k]) if k in oc else np.nan,
               "own draws, GPT-6-luna read": lambda k: (np.mean([(rd[k]["p"].get(x) if rd[k]["p"].get(x) is not None else 0.5) if x else 0.0
                                                                  for x in rd[k]["draws"]]) if k in rd else np.nan),
               "own draws, geometry": (lambda f: lambda k: f(oc[k]) if k in oc else np.nan)(geo(oc))})
    Ug, _ = G48.units()
    og, pos_g = load(G48.OWN), dict(zip([key(a, b, c) for a, b, c in zip(Ug.instance_id, Ug.seed, Ug.t)], Ug.position))
    rep_g = dict(zip([key(a, b, c) for a, b, c in zip(Ug.instance_id, Ug.seed, Ug.t)], Ug.report))
    sg = {"position alone": lambda k: pos_g[k], "self-report": lambda k: rep_g[k], "GPT-6-luna text judge": jp(load(G54.JF)),
          "own draws, own read": lambda k: G46.forecast(og[k]) if k in og else np.nan,
          "own draws, geometry": (lambda f: lambda k: f(og[k]) if k in og else np.nan)(geo(og))}
    res = {"claude": table(Uc, oc, sc), "gpt": table(Ug, og, sg)}
    (ROOT / "results/t1/forecasters_both_strata_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
