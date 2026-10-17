#!/usr/bin/env python
"""Data for the agent 3D example (Gemma-4 on CAPE-Bench, development clean units; real data, no new queries).

- Shared-space frame from Gemma's own battery answers (θ + log-odds, standardized): axis 1 = the violation-forecast
  direction (logistic weights, normalized); axes 2-3 = top principal components of the orthogonal complement.
- Every forecast shown is out of fold (GroupKFold by scenario): a point is scored by a model that never saw its scenario.
- Hidden view: Gemma's TRUE layer-18 state (top-3 PCs) vs the state estimated from its answers (ridge, out of fold).
- Trajectories: episodes with >= 3 clean turns that end in a violation, plus safe and authorized episodes.
Output: /tmp scratch JSON consumed by the artifact page.

python scripts/26_agent_demo_data.py out=<path.json>
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.battery import BATTERY_V1  # noqa: E402
from graybox.cape_bridge import rollouts  # noqa: E402
from graybox.stats import fisher_rao, yes_prob  # noqa: E402

t1 = importlib.import_module("17_clean_t1")
I = Path("/data/models/graybox_ws/outputs/interrogate/gemma4_clean")
D = OmegaConf.create({"out": "agent_demo.json", "layer": 18, "n_viol": 8, "n_safe": 3, "n_plus": 2})


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    U = pd.read_parquet(f"{I}_units.parquet")
    Z = np.load(f"{I}.npz", allow_pickle=True)
    y, g = U.y.values.astype(int), U.scenario_id.values
    q = yes_prob(Z["p_yes"], Z["p_no"])
    X = np.hstack([fisher_rao(q), np.clip(np.log(Z["p_yes"] + 1e-12) - np.log(Z["p_no"] + 1e-12), -25, 25)])
    Xs = StandardScaler().fit_transform(X)
    lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))  # noqa: E731
    p_psi = t1.oof_prob(X, y, g, lambda: lr(0.1), 5)
    Hz = np.load(f"{I}_hidden.npz")
    H = Hz["h"][:, list(Hz["layers"]).index(cfg.layer)].astype(np.float32)
    Hp = PCA(256, random_state=0).fit_transform(StandardScaler().fit_transform(H))
    p_wb = t1.oof_prob(Hp, y, g, lambda: lr(0.05), 5)
    Rr = pd.read_parquet(f"{I}_rollouts_K16.parquet")
    assert (Rr.episode_id.values == U.episode_id.values).all()
    p_roll = np.array([np.mean(eval(s_)) if eval(s_) else np.nan for s_ in Rr.R_unsafe])
    # frame: risk direction + 2 PCs of the complement (display only; fit on all units)
    w = LogisticRegression(C=0.1, max_iter=4000).fit(Xs, y).coef_[0]
    w /= np.linalg.norm(w)
    R = Xs - np.outer(Xs @ w, w)
    _, _, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    F = np.stack([w, Vt[0], Vt[1]], 1)
    P3 = (Xs - Xs.mean(0)) @ F
    P3 /= np.percentile(np.abs(P3), 98, axis=0)
    # hidden view: true top-3 PCs vs out-of-fold ridge estimate from the answers
    hpc = PCA(3, random_state=0).fit_transform(StandardScaler().fit_transform(H))
    hest = np.zeros_like(hpc)
    for tr, te in GroupKFold(5).split(X, y, g):
        sx = StandardScaler().fit(X[tr])
        hest[te] = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(sx.transform(X[tr]), hpc[tr]).predict(sx.transform(X[te]))
    hs = np.percentile(np.abs(hpc), 98, axis=0)
    hpc, hest = hpc / hs, hest / hs
    r2 = [round(float(1 - ((hpc[:, k] - hest[:, k]) ** 2).sum() / ((hpc[:, k] - hpc[:, k].mean()) ** 2).sum()), 3) for k in range(3)]
    # trajectories
    cnt = U.groupby("episode_id").size()
    viol_eps = [e for e in U[U.cls == "XUNSAFE"].episode_id.unique() if cnt[e] >= 3]
    safe_eps = [e for e in U[U.cls == "XSAFE"].episode_id.unique() if cnt[e] >= 4 and e not in set(U[U.cls == "XUNSAFE"].episode_id)]
    plus_eps = [e for e in U[U.cls == "XPLUS"].episode_id.unique() if cnt[e] >= 3]
    rng = np.random.default_rng(4)
    pick = [(e, "violation") for e in rng.choice(viol_eps, min(cfg.n_viol, len(viol_eps)), replace=False)] + \
           [(e, "safe") for e in rng.choice(safe_eps, min(cfg.n_safe, len(safe_eps)), replace=False)] + \
           [(e, "authorized") for e in rng.choice(plus_eps, min(cfg.n_plus, len(plus_eps)), replace=False)]
    trajs = []
    for e, kind in pick:
        rows = U[U.episode_id == e].sort_values("event_id")
        ep = rollouts(rows.set.iloc[0]).loc[e]
        evs = {int(x["event_id"]): x for x in ep.events}
        turns = []
        for i, r in rows.iterrows():
            ev = evs[int(r.event_id)]
            try:
                args = json.loads(ev["args_json"] or "{}")
            except Exception:
                args = {}
            arg_txt = ", ".join(f"{k}={str(v)[:60]}" for k, v in list(args.items())[:3])
            turns.append({"t": int(r.event_id), "point": P3[i].round(4).tolist(), "hid_true": hpc[i].round(4).tolist(),
                          "hid_est": hest[i].round(4).tolist(), "p_psi": round(float(p_psi[i]), 4), "p_wb": round(float(p_wb[i]), 4),
                          "p_roll": round(float(p_roll[i]), 4),
                          "unsafe": int(r.y), "action": f"{ev['tool']}({arg_txt})",
                          "answers": {BATTERY_V1[j][1]: round(float(q[i, j]), 3) for j in (18, 26, 13, 14, 3, 10)}})
        trajs.append({"episode": e, "kind": kind, "template": ep.template, "turns": turns})
    pick_cloud = rng.choice(len(U), min(3500, len(U)), replace=False)
    out = {"cloud": P3[pick_cloud].round(4).tolist(), "cloud_cls": U.cls.values[pick_cloud].tolist(),
           "cloud_p": np.round(p_psi[pick_cloud], 3).tolist(), "trajs": trajs,
           "meta": {"n_units": int(len(U)), "layer": cfg.layer, "hidden_r2_axes": r2,
                    "auroc_note": "forecasts are out of fold by scenario"}}
    Path(cfg.out).write_text(json.dumps(out, separators=(",", ":")))
    print(len(trajs), "trajectories;", "hidden R2 per axis", r2, "; bytes", len(json.dumps(out)))


if __name__ == "__main__":
    main()
