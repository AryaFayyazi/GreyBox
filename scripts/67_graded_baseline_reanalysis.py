#!/usr/bin/env python
"""Re-analysis with GRADED moderation baselines (after L3e showed the 3-level verdict is a weak, tie-heavy baseline).

Not preregistered: a sensitivity re-analysis of locked results, reported next to them. Baselines: graded P(unsafe)
of the latest message from Qwen3Guard-Gen-8B and Llama-Guard-3-8B (scripts/62 stage=moderation,
outputs/livebench/moderation_graded.parquet). Paired bootstrap over conversations (2,000).
  L3b  (per-question, Claude): harmful eval conversations, AUROC of will_refuse vs graded moderation
  L3d  (transcript, Claude):   the same on its 400 harmful single-turn conversations
  open models, both formats (leave-one-family-out readouts as scripts/38): harmful-only refusal AUROC vs graded
       moderation, pooled within model; and the compliance-risk AP of graded moderation x (1 - P(refuse)) vs graded
       moderation alone
python scripts/67_graded_baseline_reanalysis.py
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
LB = Path("/data/models/graybox_ws/outputs/livebench")
RES = Path("/home/arya/graybox/results/l3")
B = 2000


def boot(y, s, b, rng):
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(roc_auc_score(y, s) - roc_auc_score(y, b)), 4), [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]]


def claude(rng, G):
    out = {}
    a42 = importlib.import_module("42_claude_livebench_analysis")
    R = np.load(a42.BUNDLE, allow_pickle=True)
    ev = pd.read_parquet(a42.CL / "eval.parquet")
    ref = pd.read_parquet(a42.CL / "eval_refusals.parquet").set_index("id").refused.reindex(ev.id).values
    sc = a42.scores(ev, R)
    m = (ev.harmful.values == 1) & (ref >= 0)
    y, s = ref[m], sc["will_refuse"][m]
    for k in ("qg_p", "lg_p"):
        b = G[k].reindex(ev.id).values[m]
        out[f"L3b_{k}"] = {"greybox": round(float(roc_auc_score(y, s)), 4), "baseline": round(float(roc_auc_score(y, b)), 4), "diff_ci": boot(y, s, b, rng)}
    a57 = importlib.import_module("57_claude_transcript_analysis")
    R = np.load(a57.BUNDLE, allow_pickle=True)
    ev = pd.read_parquet(a57.CT / "eval.parquet")
    ref = pd.concat([pd.read_parquet(a57.CB / f"{x}_refusals.parquet") for x in ("evalA", "evalB")]).set_index("id").refused.reindex(ev.id).values
    sc, _ = a57.scores(ev, R)
    m = (ev.harmful.values == 1) & (ref >= 0)
    y, s = ref[m], sc["will_refuse"][m]
    for k in ("qg_p", "lg_p"):
        b = G[k].reindex(ev.id).values[m]
        out[f"L3d_{k}"] = {"greybox": round(float(roc_auc_score(y, s)), 4), "baseline": round(float(roc_auc_score(y, b)), 4), "diff_ci": boot(y, s, b, rng)}
    return out


def open_models(rng, G, lb_dir, feat):
    os.environ["GB_LB"], os.environ["GB_FEAT"], os.environ["GB_NSAMP"] = str(lb_dir), feat, "0"
    a38 = importlib.reload(importlib.import_module("38_livebench_analysis"))
    items = pd.read_parquet(lb_dir / "items.parquet")
    g = {k: G[k].reindex(items.id).values for k in ("qg_p", "lg_p")}
    models = [d.name for d in sorted(lb_dir.iterdir()) if (d / "psi.npz").exists() and (d / "refusals.parquet").exists()]
    X = {m: a38.feats(np.load(lb_dir / m / "psi.npz")) for m in models}
    REF = {m: pd.read_parquet(lb_dir / m / "refusals.parquet").set_index("id").refused.reindex(items.id).values for m in models}
    tr, ev = (items.split == "train").values, (items.split == "eval").values
    h = items.harmful.values == 1
    per = {k: [] for k in ("gb", "qg_p", "lg_p", "ap_qg", "ap_qg_x_gb")}
    for m in models:
        others = [o for o in models if a38.run12.FAMILY[o] != a38.run12.FAMILY[m]]
        ok_o = [(o, REF[o] >= 0) for o in others]
        r = a38.clf().fit(np.vstack([X[o][tr & ok] for o, ok in ok_o]), np.concatenate([REF[o][tr & ok] for o, ok in ok_o])).predict_proba(X[m])[:, 1]
        sel = ev & h & (REF[m] >= 0)
        y = REF[m][sel]
        per["gb"].append(roc_auc_score(y, r[sel]))
        for k in ("qg_p", "lg_p"):
            per[k].append(roc_auc_score(y, g[k][sel]))
        okm = ev & (REF[m] >= 0)
        event = (h & (REF[m] == 0))[okm].astype(int)
        per["ap_qg"].append(average_precision_score(event, g["qg_p"][okm]))
        per["ap_qg_x_gb"].append(average_precision_score(event, g["qg_p"][okm] * (1 - r[okm])))
    return {"models": models, **{k: round(float(np.mean(v)), 4) for k, v in per.items()}, "per_model_gb_minus_qg": [round(a - b, 4) for a, b in zip(per["gb"], per["qg_p"])]}


def main():
    rng = np.random.default_rng(0)
    G = pd.read_parquet(LB / "moderation_graded.parquet").set_index("id")
    res = {"claude": claude(rng, G),
           "open_per_question": open_models(rng, G, LB, "theta_lo"),
           "open_transcript": open_models(rng, G, Path("/data/models/graybox_ws/outputs/livebench_transcript"), "theta")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "graded_baseline_reanalysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
