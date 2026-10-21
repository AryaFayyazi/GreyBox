#!/usr/bin/env python
"""Data for the two 3D views on the Self-policing page (site/dist/data/selfpolice3d.json).

A. Inside the agent at the moment it decides (Gemma-4, 5,492 development steps, CAPE-Bench). Its decision state at layer
   24 (2,816 numbers). x = the cross-fitted linear readout of "this call violates the policy" (5-fold, grouped by
   scenario; the shipped probe recipe); y, z = the two largest principal directions of the state after removing the
   readout direction. Colour = CAPE's checker; ring = the agent's own self-audit flagged the call (all 4 answers "no").
B. What a model's answers reveal (G1 confirm contexts, Qwen3-8B): 3D classical scaling of its hidden geometry (chosen
   layer, cosine) and of its observable geometry (Fisher-Rao distances between yes/no answer laws), the second
   aligned to the first by an orthogonal Procrustes rotation and one scale (no other fitting). Colour = kind of context.
   The Spearman agreement of the full distance matrices is printed (G1, preregistered).
python web/build_selfpolice3d.py <out.json>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graybox.stats import within_auroc  # noqa: E402
I = Path("/data/models/graybox_ws/outputs/interrogate")


def agent_view(layer_index=3, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    z = np.load(I / "gemma4_clean_decision_hidden.npz")
    U = pd.read_parquet(I / "gemma4_clean_units.parquet")
    A = pd.read_parquet("/data/models/graybox_ws/outputs/self_audit/gemma4_clean_sample.parquet").set_index(["episode_id", "event_id"]).s
    s = A.reindex(list(zip(U.episode_id, U.event_id))).values
    ok = z["ok"].astype(bool) & np.isfinite(s)
    H, y, g, st = z["h"][ok, layer_index].astype(np.float32), U.y.values[ok].astype(int), U.scenario_id.values[ok], U.template.values[ok]
    Xs = StandardScaler().fit_transform(H)
    logit, w = np.zeros(len(y)), np.zeros(H.shape[1])
    for tr, te in GroupKFold(5).split(Xs, y, g):
        m = make_pipeline(PCA(256, random_state=seed), LogisticRegression(C=0.05, max_iter=3000)).fit(Xs[tr], y[tr])
        logit[te] = m.decision_function(Xs[te])
        w += m[0].components_.T @ m[1].coef_[0]
    w /= np.linalg.norm(w)
    R = Xs - np.outer(Xs @ w, w)                                        # remove the readout direction
    pc = PCA(2, random_state=seed).fit(R)
    yz = pc.transform(R)
    tot = (Xs ** 2).sum()
    return {"x": np.round(logit, 3).tolist(), "y": np.round(yz[:, 0], 3).tolist(), "z": np.round(yz[:, 1], 3).tolist(),
            "viol": y.tolist(), "flag": (s[ok] >= 1 - 1e-9).astype(int).tolist(), "template": [t.split("|")[0] for t in st],
            "stats": {"n": int(len(y)), "violations": int(y.sum()), "layer": int(z["layers"][layer_index]),
                      "readout_auroc_within": round(float(within_auroc(y, logit, st)), 3),
                      "selfaudit_auroc_within": round(float(within_auroc(y, s[ok], st)), 3),
                      "selfaudit_flag_violation": round(float((s[ok][y == 1] >= 1 - 1e-9).mean()), 3),
                      "selfaudit_flag_safe": round(float((s[ok][y == 0] >= 1 - 1e-9).mean()), 3),
                      "variance_yz": round(float(pc.explained_variance_.sum() * (len(y) - 1) / tot), 3)}}


def classical_mds(D2, k=3):
    n = len(D2)
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ D2 @ J
    ev, V = np.linalg.eigh(B)
    idx = np.argsort(ev)[::-1][:k]
    return V[:, idx] * np.sqrt(np.maximum(ev[idx], 0)), ev[idx], np.maximum(ev, 0).sum()


def geometry_view(model="qwen3_8b", n_show=720, seed=0):
    spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/129_infogeom.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    C = g.contexts()
    dev = np.where(C.split.values == "dev")[0]
    k, layer_no, _ = g.select_layer(model, np.random.default_rng(0).choice(dev, 1000, replace=False))
    conf = np.where(C.split.values == "confirm")[0]
    H, _ = g.hidden(model)
    p, _ = g.p_yes(model)
    Dh = g.cos_dist(H[conf, k])
    Da2 = g.sq_dist(g.theta(p[conf]))
    rho = g.rho(np.sqrt(Da2), Dh)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(conf), min(n_show, len(conf)), replace=False))
    Xh, evh, toth = classical_mds(Dh[np.ix_(pick, pick)] ** 2)
    Xa, eva, tota = classical_mds(Da2[np.ix_(pick, pick)])
    Xh, Xa = Xh - Xh.mean(0), Xa - Xa.mean(0)
    Uo, S, Vt = np.linalg.svd(Xa.T @ Xh)                               # orthogonal Procrustes + one scale
    Q = Uo @ Vt
    Xa = Xa @ Q * (S.sum() / (Xa ** 2).sum())
    sc = 1 / np.abs(Xh).max()
    names = {"sst2": "movie review", "harmful": "harmful request", "benign": "benign request", "boolq": "yes/no question",
             "gsm8k": "maths problem", "wiki": "encyclopedia text"}
    kinds = np.array([names[x] for x in C.source.values[conf][pick]])
    return {"hidden": np.round(Xh * sc, 3).tolist(), "answers": np.round(Xa * sc, 3).tolist(), "kind": kinds.tolist(),
            "text": [t[:90] for t in C.text.values[conf][pick]],
            "stats": {"model": "Qwen3-8B", "contexts": int(len(conf)), "shown": int(len(pick)), "layer": int(layer_no), "spearman_all_pairs": round(float(rho), 3),
                      "mds_variance_hidden": round(float(evh.sum() / toth), 3), "mds_variance_answers": round(float(eva.sum() / tota), 3)}}


def main():
    out = Path(sys.argv[1])
    D = {"agent": agent_view(), "geometry": geometry_view()}
    out.write_text(json.dumps(D))
    print(json.dumps({k: v["stats"] for k, v in D.items()}, indent=1))


if __name__ == "__main__":
    main()
