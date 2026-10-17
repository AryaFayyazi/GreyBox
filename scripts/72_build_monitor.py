#!/usr/bin/env python
"""Build the shippable reader-monitor probe, and check end to end that the SDK reproduces the preregistered numbers.

1. probe (dev only): the T4 pipeline (standardize -> PCA 256 -> logistic C=0.05) at layer 32 on Gemma development
   units -> models/monitor/reader_probe_dev.joblib (for verification: never trained on fresh units)
2. probe (all): the same on development + fresh units -> models/monitor/reader_probe.joblib (the shipped probe; its
   accuracy is quoted from T4/T5, never measured on its own training data)
3. end-to-end check (GPU): graybox.monitor.ReaderMonitor with the dev-only probe on 600 fresh units (all-unsafe
   oversampled: 200 unsafe + 400 safe, seed 3), built from the CAPE traces through the public SDK functions (Step,
   ctx_text, act_text). The AUROC must match T4's fresh AUROC on the same units from the stored features within 0.01;
   prompt-text identity with scripts/69 is asserted.
python scripts/72_build_monitor.py stage=build|check
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox import cape_bridge as cb  # noqa: E402

I = Path("/data/models/graybox_ws/outputs/interrogate")
T4 = Path("/data/models/graybox_ws/outputs/t4")
MOD = Path("/data/models/graybox_ws/models/monitor")
D = OmegaConf.create({"stage": "build", "layer": 32})


def fit(X, y):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=4000)).fit(X, y)


def stage_build(cfg):
    import joblib
    Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz")
    j = [int(x) for x in Hd["layers"]].index(cfg.layer)
    yd = pd.read_parquet(I / "gemma4_clean_units.parquet").y.values.astype(int)
    yf = pd.read_parquet(I / "gemma4_fresh_units.parquet").y.values.astype(int)
    MOD.mkdir(parents=True, exist_ok=True)
    joblib.dump({"probe": fit(Hd["h"][:, j].astype(np.float32), yd), "layer": cfg.layer, "reader": "Qwen/Qwen3-8B", "trained_on": "gemma4 dev units"}, MOD / "reader_probe_dev.joblib")
    X = np.vstack([Hd["h"][:, j], Hf["h"][:, j]]).astype(np.float32)
    joblib.dump({"probe": fit(X, np.concatenate([yd, yf])), "layer": cfg.layer, "reader": "Qwen/Qwen3-8B",
                 "trained_on": "gemma4 dev + fresh units (17,623)", "validated": "T4 0.994 same scenarios; T5 0.969 unseen families"}, MOD / "reader_probe.joblib")
    print("saved", sorted(p.name for p in MOD.iterdir()), flush=True)


def stage_check(cfg):
    from sklearn.metrics import roc_auc_score
    from graybox.monitor import ReaderMonitor, Step, Q, act_text, ctx_text
    t3, t4 = importlib.import_module("68_t3_deployable"), importlib.import_module("69_t4_reader_action")
    assert Q == t4.Q, "SDK prompt must be identical to the validated one"
    U = pd.read_parquet(I / "gemma4_fresh_units.parquet")
    rng = np.random.default_rng(3)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(U.y.values == 1), 200, replace=False), rng.choice(np.flatnonzero(U.y.values == 0), 400, replace=False)]))
    mon = ReaderMonitor.load(MOD / "reader_probe_dev.joblib")
    p_sdk, same, lat = [], 0, []
    for i in idx:
        r = U.iloc[i]
        ep = cb.rollouts(r.set).loc[r.episode_id]
        spec, evs = cb.specs(r.bench)[ep.instance_id], list(ep.events)
        hist = [Step(e["tool"], json.loads(e["args_json"] or "{}"), e["tool_result"] or "") for e in evs
                if int(e["event_id"]) < int(r.event_id) and e["is_tool_call"] and e["tool"] is not None]
        ev = {int(x["event_id"]): x for x in evs}[int(r.event_id)]
        action = (ev["tool"], json.loads(ev["args_json"] or "{}"))
        same += ctx_text(spec["system_prompt"], spec["task_prompt"], hist) == t3.ctx_text(spec, evs, int(r.event_id)) and act_text(*action) == t3.act_text(*action)
        t0 = time.time()
        p_sdk.append(mon.score(spec["system_prompt"], spec["task_prompt"], hist, action))
        lat.append(time.time() - t0)
    y = U.y.values[idx]
    import joblib
    d = joblib.load(MOD / "reader_probe_dev.joblib")
    Hf = np.load(T4 / "fresh_act.npz")
    j = [int(x) for x in Hf["layers"]].index(cfg.layer)
    p_ref = d["probe"].predict_proba(Hf["h"][idx, j].astype(np.float32))[:, 1]
    res = {"n": int(len(idx)), "text_identical": f"{same}/{len(idx)}", "auroc_sdk": round(float(roc_auc_score(y, p_sdk)), 4),
           "auroc_stored_features": round(float(roc_auc_score(y, p_ref)), 4), "max_abs_prob_diff": round(float(np.max(np.abs(np.array(p_sdk) - p_ref))), 4),
           "latency_s_median": round(float(np.median(lat)), 4)}
    res["pass"] = abs(res["auroc_sdk"] - res["auroc_stored_features"]) <= 0.01 and same == len(idx)
    (ROOT / "results/t1/monitor_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"build": stage_build, "check": stage_check}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
