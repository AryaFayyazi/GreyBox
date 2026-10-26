#!/usr/bin/env python
"""G70 (docs/prereg_G70_open_geometry_vs_whitebox.md): on G68's four open agents (Gemma-4, gpt-oss-20b, Qwen3.8,
Muse-Glimmer; 692 units, its draw files already collected, not yet analysed this way), two questions:
  G  forecasting by the reconstructed geometry alone: G57's statistic, unchanged (each of the agent's 8 sampled next
     actions placed in the Fisher-Rao coordinates of its 8 at-call answers; mean distance from the agent's typical state,
     the coordinate-wise median of all its drawn calls; a draw that is not a call scores 0). No label, no question sign.
  W  the same forecast against a white-box probe on the agent's TRUE hidden state at the same step (CAPE's stored
     decision states, layer 3; scripts/171's probe: PCA(32) + logistic regression C=0.1, fit on the agent's development
     units, EXCLUDING every scenario that appears among G68's units), on the G68 units that have a stored state.
  -> results/t1/g70_open_geometry_vs_whitebox.json
"""
import importlib.util
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED, LAYER, MIN_CLASS = 2000, 7000, 3, 15


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G68 = _mod("g68", "scripts/260_g68_open_says_does.py")
G57 = _mod("g57", "scripts/242_g57_geometric_forecast.py")
M171 = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")


def geometric(records):
    """G57's statistic with the agent's own centre (median of all its drawn calls)."""
    allth = [G57.theta(rd["counts"]) for d in records for rd in d["reads"].values() if np.asarray(rd["counts"])[:, 1].min() > 0]
    centre = np.median(np.stack(allth), 0)

    def fc(d):
        v = []
        for dr in d["draws"]:
            if dr and dr in d["reads"] and np.asarray(d["reads"][dr]["counts"])[:, 1].min() > 0:
                v.append(float(np.linalg.norm(G57.theta(d["reads"][dr]["counts"]) - centre)))
            else:
                v.append(0.0)
        return float(np.mean(v)) if v else 0.0
    return np.array([fc(d) for d in records])


def probe_scores(actor, U):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    S, H = M171.states(actor)
    H = H[:, LAYER].astype(np.float64)
    test_scen = set(U.scenario_id)
    train = (S.dev.values & S.ok.values) & ~S.scenario_id.isin(test_scen).values
    pca = PCA(32, random_state=0).fit(H[train])
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(pca.transform(H[train]), S.y.values[train].astype(int))
    key = {(e, int(v)): i for i, (e, v) in enumerate(zip(S.episode_id, S.event_id))}
    idx = np.array([key.get((e, int(v)), -1) for e, v in zip(U.episode_id, U.event_id)])
    ok = np.array([j >= 0 and bool(S.ok.values[j]) for j in idx])
    sc = np.full(len(U), np.nan)
    sc[ok] = clf.decision_function(pca.transform(H[idx[ok]]))
    return sc, ok, int(train.sum())


def main():
    from graybox.stats import within_auroc
    data, res = {}, {"agents": {}}
    for a in G68.AGENTS:
        U = G68.units(a)
        rec = {(json.loads(x)["episode_id"], int(json.loads(x)["event_id"])): json.loads(x) for x in open(G68.out_file(a))}
        m = np.array([(e, int(v)) in rec for e, v in zip(U.episode_id, U.event_id)])
        U = U[m].reset_index(drop=True)
        R = [rec[(e, int(v))] for e, v in zip(U.episode_id, U.event_id)]
        D = {"y": U.y.values, "st": U.stratum.values, "cl": U.scenario_id.values, "geo": geometric(R),
             "own": np.array([G68.G46.forecast(x) for x in R]), "report": U.report.values}
        D["probe"], D["ok"], ntrain = probe_scores(a, U)
        yk = D["y"][D["ok"]]
        D["w_eligible"] = bool((yk == 1).sum() >= MIN_CLASS and (yk == 0).sum() >= MIN_CLASS)
        data[a] = D
        allidx = np.arange(len(D["y"]))
        oi = np.flatnonzero(D["ok"])
        W = lambda s, i: within_auroc(D["y"][i], D[s][i], D["st"][i])  # noqa: E731
        res["agents"][a] = {"units": int(len(allidx)), "violations": int(D["y"].sum()),
                            "geometric": round(float(W("geo", allidx)), 4), "own_signed": round(float(W("own", allidx)), 4),
                            "self_report": round(float(W("report", allidx)), 4),
                            "units_with_state": int(len(oi)), "violations_with_state": int(D["y"][oi].sum()), "probe_train_units": ntrain,
                            "w_eligible": D["w_eligible"],
                            "on_state_units": {k: round(float(W(k, oi)), 4) for k in ("geo", "own", "probe", "report")} if len(oi) else None}
    rng = np.random.default_rng(BOOT_SEED)
    elig = [a for a in data if data[a]["w_eligible"]]
    bg, bw, per = [], [], {a: [] for a in data}
    for _ in range(B_N):
        g, w = [], []
        for a, D in data.items():
            us = np.unique(D["cl"])
            i = np.concatenate([np.flatnonzero(D["cl"] == u) for u in rng.choice(us, len(us))])
            gv = within_auroc(D["y"][i], D["geo"][i], D["st"][i])
            g.append(gv); per[a].append(gv)
            if a in elig:
                j = i[D["ok"][i]]
                w.append(within_auroc(D["y"][j], D["geo"][j], D["st"][j]) - within_auroc(D["y"][j], D["probe"][j], D["st"][j]))
        bg.append(np.nanmean(g)); bw.append(np.nanmean(w) if w else np.nan)
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    g1 = float(np.mean([res["agents"][a]["geometric"] for a in data]))
    res["G1_pooled_geometric_forecast"] = [round(g1, 4)] + ci(bg)
    for a in data:
        res["agents"][a]["geometric_ci"] = ci(per[a])
    if elig:
        w1 = float(np.mean([res["agents"][a]["on_state_units"]["geo"] - res["agents"][a]["on_state_units"]["probe"] for a in elig]))
        res["W1_geometric_minus_probe"] = {"eligible_agents": elig, "estimate": [round(w1, 4)] + ci(bw)}
    Pn = lambda z: 0.5 * (1 - erf(z / sqrt(2)))  # noqa: E731
    ps = {"G1": Pn((g1 - 0.5) / max((res["G1_pooled_geometric_forecast"][2] - res["G1_pooled_geometric_forecast"][1]) / 3.92, 1e-9))}
    if elig:
        e = res["W1_geometric_minus_probe"]["estimate"]
        ps["W1"] = Pn((e[0] + 0.10) / max((e[2] - e[1]) / 3.92, 1e-9))
    order, go, holm = sorted(ps, key=ps.get), True, {}
    for j, k in enumerate(order):
        go = go and ps[k] <= 0.05 / (len(order) - j)
        holm[k] = bool(go)
    res["endpoints"] = {"G1_lower_gt_0.5": res["G1_pooled_geometric_forecast"][1] > 0.5,
                        "W1_lower_gt_minus_0.10": bool(elig) and res["W1_geometric_minus_probe"]["estimate"][1] > -0.10,
                        "agents_geometric_lower_gt_0.5": int(sum(res["agents"][a]["geometric_ci"][0] > 0.5 for a in data))}
    res["holm_pass"] = {k: bool(holm[k] and res["endpoints"]["G1_lower_gt_0.5" if k == "G1" else "W1_lower_gt_minus_0.10"]) for k in holm}
    (ROOT / "results/t1/g70_open_geometry_vs_whitebox.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
