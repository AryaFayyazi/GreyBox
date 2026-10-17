#!/usr/bin/env python
"""The monitor's own space for the Claude agent actions of T8b (visualisation; no new test).

stage=hidden (GPU)  reader features (Qwen3-8B, the shipped probe's layer) for the 3,128 T8b units, through the public
                    SDK (ReaderMonitor.features). Asserted: the probe on these features reproduces the stored T8b
                    scores (outputs/t8b/reader.npz) and AUROC. -> outputs/t8b/features.npz
stage=coords (CPU)  3D coordinates in the probe's own space. The shipped probe is standardize -> PCA(256) -> logistic,
                    so a unit's logit is w.z + b with z its 256-d PCA coordinates. Axis 1 = the logit (exactly the
                    monitor's score; the escalation cut is the plane x = logit(threshold)). Axes 2-3 = the top two
                    principal components of z with the w direction projected out (fit on the Claude units). Distances
                    along axes 2-3 are not risk. -> outputs/t8b/space.json
python scripts/78_claude_monitor_space.py stage=hidden|coords
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
PROBE = "/data/models/graybox_ws/models/monitor/reader_probe.joblib"
D = OmegaConf.create({"stage": "coords"})


def stage_hidden(cfg):
    from sklearn.metrics import roc_auc_score
    from graybox.monitor import ReaderMonitor
    t8 = importlib.import_module("76_t8b_claude_all_actions")
    U = t8.units()
    mon = ReaderMonitor.load(PROBE)
    H = np.vstack([mon.features(p) for p in U.prompt])
    p = mon.probe.predict_proba(H)[:, 1]
    ref = np.load(t8.OUT / "reader.npz")["p"]
    a, b = roc_auc_score(U.y, p), roc_auc_score(U.y, ref)
    print(f"max |p - stored| {np.max(np.abs(p - ref)):.2e}; AUROC {a:.4f} vs stored {b:.4f}", flush=True)
    assert abs(a - b) < 0.002, "features must reproduce T8b"
    np.savez_compressed(t8.OUT / "features.npz", h=H.astype(np.float16), p=p)


def stage_coords(cfg):
    import joblib
    t8 = importlib.import_module("76_t8b_claude_all_actions")
    U = t8.units()
    F = np.load(t8.OUT / "features.npz")
    pipe = joblib.load(PROBE)["probe"]
    sc, pca, lr = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    Z = pca.transform(sc.transform(F["h"].astype(np.float32)))
    w, b = lr.coef_[0], lr.intercept_[0]
    logit = Z @ w + b
    assert np.allclose(1 / (1 + np.exp(-logit)), pipe.predict_proba(F["h"].astype(np.float32))[:, 1], atol=1e-5)
    Zp = Z - np.outer(Z @ w / (w @ w), w)
    Zp -= Zp.mean(0)
    _, s, Vt = np.linalg.svd(Zp, full_matrices=False)
    P = Zp @ Vt[:2].T
    thr_p = float(np.quantile(F["p"], 0.95))
    out = {"episode_id": U.episode_id.tolist(), "act": U.act.str.slice(0, 120).tolist(), "y": U.y.astype(int).tolist(),
           "x": np.round(logit, 3).tolist(), "y2": np.round(P[:, 0], 3).tolist(), "z": np.round(P[:, 1], 3).tolist(),
           "cut_logit_label_free": float(np.log(thr_p / (1 - thr_p))),
           "var_explained_axes23": [float(v) for v in (s[:2] ** 2 / (s ** 2).sum())]}
    (t8.OUT / "space.json").write_text(json.dumps(out))
    print("saved", len(U), "points; cut logit", round(out["cut_logit_label_free"], 3), "; var axes 2-3", out["var_explained_axes23"], flush=True)


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"hidden": stage_hidden, "coords": stage_coords}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
