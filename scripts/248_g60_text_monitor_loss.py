#!/usr/bin/env python
"""G60: does the theory predict, in size and without labels, how much a text monitor loses against the agent's policy?
(docs/prereg_G60_text_monitor_loss.md). Proposition 3: a forecaster q that sees only the context loses, in Brier score,
E[(pi - q)^2] against the agent's policy pi. With the agent's own k draws (share f of violating non-read draws) this gap
is estimated with no label: gap = mean (f - q)^2 - mean f(1-f)/(k-1). With labels Y, the observed excess of q over the
draws is mean (Y - q)^2 - mean (Y - f)^2, whose expectation is E[(pi-q)^2] - E[pi(1-pi)/k]; the prediction from the draws
alone is mean (f - q)^2 - 2 mean f(1-f)/(k-1) (scripts/230's identity, applied to a text monitor instead of a reader).
Pairs (existing data, not analysed this way before this lock):
  claude_self  Claude Haiku 5.5's units with exact contexts (G46c own draws) x Claude reasoning over the transcript (G52)
  claude_gpt   same units x GPT-6-luna reasoning over Claude's transcript (G55)
  gpt_self     GPT-6-luna's units, faithful replay (G48 own draws) x GPT-6-luna reasoning over its transcript (G54)
Units with at least 2 non-read draws. -> results/t1/g60_text_monitor_loss.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED, BAND = 2000, 6000, 0.02


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def load(units, own_file, judge_file, shares):
    U = units()
    U = U[0] if isinstance(U, tuple) else U
    key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
    own = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(own_file))}
    jd = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(judge_file))}
    rows = []
    for a, b, c, y, sid in zip(U.instance_id, U.seed, U.t, U.y, U.scenario_id):
        k = key(a, b, c)
        if k not in own or k not in jd or jd[k].get("p") is None:
            continue
        f, n = shares(own[k])
        if n >= 2:
            rows.append((float(y), f, n, float(jd[k]["p"]), sid))
    y, f, n, q, cl = (np.array(v) for v in zip(*rows))
    return {"y": y, "f": f, "k": n.astype(float), "q": q, "cl": cl}


def quantities(d, i):
    y, f, k, q = d["y"][i], d["f"][i], d["k"][i], d["q"][i]
    noise = np.mean(f * (1 - f) / (k - 1))
    obs = np.mean((y - q) ** 2) - np.mean((y - f) ** 2)
    pred = np.mean((f - q) ** 2) - 2 * noise
    gap = np.mean((f - q) ** 2) - noise
    return obs, pred, obs - pred, gap, np.mean(y) - np.mean(f)


def main():
    G46B = _mod("g46b", "scripts/230_g46b_policy_gap.py")
    G46C = _mod("g46c", "scripts/232_g46c_exact_context.py")
    G48 = _mod("g48", "scripts/229_g48_gpt_rollout.py")
    G52 = _mod("g52", "scripts/237_g52_text_judge_vs_policy.py")
    G54 = _mod("g54", "scripts/239_g54_gpt_text_judge.py")
    G55 = _mod("g55", "scripts/240_g55_gpt_judges_claude.py")
    D = {"claude_self": load(G46C.units, G46C.OWN, G52.JF, G46B.shares),
         "claude_gpt": load(G46C.units, G46C.OWN, G55.JF, G46B.shares),
         "gpt_self": load(G48.units, G48.OWN, G54.JF, G46B.shares)}
    rng = np.random.default_rng(BOOT_SEED)
    boots = {p: [] for p in D}
    for _ in range(B_N):
        for p, d in D.items():
            us = np.unique(d["cl"])
            i = np.concatenate([np.flatnonzero(d["cl"] == u) for u in rng.choice(us, len(us))])
            boots[p].append(quantities(d, i))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    names = ("observed_excess", "predicted_excess", "observed_minus_predicted", "label_free_policy_gap", "calibration_y_minus_f")
    res = {"pairs": {}}
    for p, d in D.items():
        full = quantities(d, np.arange(len(d["y"])))
        B = np.array(boots[p])
        res["pairs"][p] = {"units": int(len(d["y"])), "violations": int(d["y"].sum())}
        res["pairs"][p].update({nm: [round(float(full[j]), 4)] + ci(B[:, j]) for j, nm in enumerate(names)})
    e1 = {p: (lambda v: v[1] >= -BAND and v[2] <= BAND)(res["pairs"][p]["observed_minus_predicted"]) for p in D}
    diff = np.array(boots["claude_self"])[:, 3] - np.array(boots["gpt_self"])[:, 3]
    full_diff = res["pairs"]["claude_self"]["label_free_policy_gap"][0] - res["pairs"]["gpt_self"]["label_free_policy_gap"][0]
    res["E2_claude_self_gap_minus_gpt_self_gap"] = [round(float(full_diff), 4)] + ci(diff)
    res["endpoints"] = {"E1_each_pair_inside_pm0.02": all(e1.values()), "E1_by_pair": e1,
                        "E2_lower_gt_0": res["E2_claude_self_gap_minus_gpt_self_gap"][1] > 0}
    (ROOT / "results/t1/g60_text_monitor_loss.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
