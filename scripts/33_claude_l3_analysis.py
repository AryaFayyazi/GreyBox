#!/usr/bin/env python
"""L3 analysis (docs/prereg_L3_claude.md): frozen open-model readouts applied to Claude Haiku 5.5's sampled answers.

python scripts/33_claude_l3_analysis.py      # needs outputs/l3/claude_haiku55_run.json (+ judged refusals if present)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = Path("/data/models/graybox_ws/outputs/l3")
RES = Path("/home/arya/graybox/results/l3")


def main():
    d = json.loads((OUT / "claude_haiku55_run.json").read_text())
    R = np.load("/data/models/graybox_ws/models/live/readouts.npz", allow_pickle=True)
    assert list(R["items"]) == d["items"]
    C = pd.read_parquet("/data/models/graybox_ws/outputs/l2/contexts.parquet").set_index("id")
    rows = []
    for r in d["results"]:
        c = np.array(r["counts"], float)                     # (24, 3): yes, no, other
        q = (c[:, 0] + 0.5) / (c[:, 0] + c[:, 1] + 1.0)      # shrunk toward 0.5 when answers are missing
        theta = 2 * np.arcsin(np.sqrt(q))
        row = {"id": r["id"], "source": r["source"], "harmful": C.harmful.get(r["id"]), "sentiment": C.sentiment.get(r["id"])}
        for con in R["concepts"]:
            z = float(theta @ con["w"] + con["b"])
            row[con["name"]] = 1 / (1 + np.exp(-z))
            row[con["name"] + "_thr"] = con["threshold"]
        rows.append(row)
    df = pd.DataFrame(rows)
    judged = OUT / "claude_haiku55_refusals.parquet"
    if judged.exists():
        df = df.merge(pd.read_parquet(judged)[["id", "refused"]], on="id", how="left")
    rng = np.random.default_rng(0)

    def auroc_ci(y, s):
        y, s = np.asarray(y, int), np.asarray(s, float)
        a = roc_auc_score(y, s)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i):
                bs.append(roc_auc_score(y[i], s[i]))
        return round(float(a), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    res = {"n": int(len(df)), "worst_open_family_auroc": {c["name"]: c["validation"]["worst"] for c in R["concepts"]}}
    hb = df[df.source.isin(["harmful", "benign"])]
    a, ci = auroc_ci(hb.source == "harmful", hb.harmful_request)
    res["C1_harmful_request"] = {"auroc": a, "ci95": ci, "pass": ci[0] > 0.5}
    if "refused" in df:
        hr = hb.dropna(subset=["refused"])
        if hr.refused.sum() >= 10 and (1 - hr.refused).sum() >= 10:
            a, ci = auroc_ci(hr.refused, hr.will_refuse)
            res["C2_will_refuse"] = {"auroc": a, "ci95": ci, "pass": ci[0] > 0.5, "refusal_rate": round(float(hr.refused.mean()), 4)}
        else:
            res["C2_will_refuse"] = {"not_tested": f"refusals {int(hr.refused.sum())} / {len(hr)}"}
    ss = df[df.source == "sst2"]
    a, ci = auroc_ci(1 - ss.sentiment.astype(int), ss.negative_tone)
    res["C3_negative_tone"] = {"auroc": a, "ci95": ci, "pass": ci[0] > 0.5}
    cal = {}
    for name, sub, lab in (("harmful_request", hb, hb.source == "harmful"), ("negative_tone", ss, ss.sentiment == 0)):
        flag = sub[name] >= sub[name + "_thr"]
        cal[name] = {"tpr": round(float(flag[lab].mean()), 4), "fpr": round(float(flag[~lab].mean()), 4)}
    res["frozen_threshold_rates"] = cal
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "claude_haiku55_l3.json").write_text(json.dumps(res, indent=1))
    df.to_parquet(RES / "claude_haiku55_scores.parquet")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
