#!/usr/bin/env python
"""G35b: with a better measurement, does the read-out follow the hidden state's violation coordinate beyond the label?
(docs/prereg_G35b_dither_reads_state.md). G35's L1 and L3 on G31's re-measured steps (scripts/203), arms R4 (G27's
design) and D16 (4 wordings x 4 answers). P: G15's probe on the stored decision state (layer index 3, fit on
development calls; an instrument, never part of the method), applied to the steps that have a valid stored state
and are not development calls. Cells: template x tool x step x label with >= 3 steps.  CPU only.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from graybox.stats import within_auroc  # noqa: E402

B_N, SEED, MIN = 2000, 3550, 3


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def probe(M171, actor):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    U, H = M171.states(actor)
    H = H[:, 3].astype(np.float64)
    dm = (U.dev & U.ok).values
    pca = PCA(32, random_state=0).fit(H[dm])
    lr = LogisticRegression(C=0.1, max_iter=5000).fit(pca.transform(H[dm]), U.y.values[dm])
    return U, (lambda idx: lr.decision_function(pca.transform(H[idx])))


def load():
    G31 = _mod("g31", "scripts/203_g31_dither.py")
    M171 = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    import pandas as pd
    out = {}
    for a in G31.AGENTS:
        f = G31.G27.OUT / f"g31_{a}.parquet"
        if not f.exists():
            continue
        S = pd.read_parquet(f)
        A = [json.loads(x) for x in S.answers]
        arms = G31.arms(A)
        s = {k: G31.G27.angles(arms[k])[0] @ G31.SG for k in ("R4", "D16")}
        U, P = probe(M171, a)
        key = {k: i for i, k in enumerate(zip(U.episode_id, U.event_id.astype(int)))}
        idx = np.array([key.get((e, int(v)), -1) for e, v in zip(S.episode_id, S.event_id)])
        ok = (idx >= 0) & np.array([bool(U.ok.values[j]) and not bool(U.dev.values[j]) if j >= 0 else False for j in idx])
        out[a] = {"s": {k: v[ok] for k, v in s.items()}, "P": P(idx[ok]), "y": S.y.values[ok].astype(int),
                  "cell": np.array([f"{t}|{o}|{st}|{y}" for t, o, st, y in zip(S.template[ok], S.tool[ok], S.t[ok], S.y[ok])]),
                  "strata": np.array([f"{t}|{o}|{st}" for t, o, st in zip(S.template[ok], S.tool[ok], S.t[ok])]),
                  "sc": S.scenario_id.values[ok], "n_all": int(len(S)), "n_ok": int(ok.sum())}
    return out


def l1(D, arm, idx):
    a, b = [], []
    cells = {}
    for i in idx:
        cells.setdefault(D["cell"][i], []).append(i)
    for g in cells.values():
        g = np.array(g)
        if len(g) < MIN or np.ptp(D["s"][arm][g]) == 0 or np.ptp(D["P"][g]) == 0:
            continue
        a.append((rankdata(D["s"][arm][g]) - 0.5) / len(g) - 0.5)
        b.append((rankdata(D["P"][g]) - 0.5) / len(g) - 0.5)
    return float(np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1]) if a else np.nan


def l3(D, arm, idx):
    s, P, y, st = D["s"][arm][idx], D["P"][idx], D["y"][idx], D["strata"][idx]
    r = np.zeros(len(idx))
    for u in np.unique(st):
        i = st == u
        rs, rp = rankdata(s[i]), rankdata(P[i])
        r[i] = rs - np.polyval(np.polyfit(rp, rs, 1), rp) if i.sum() >= 3 and np.ptp(rp) > 0 else rs
    a0, a1 = within_auroc(y, s, st), within_auroc(y, r, st)
    return float((a0 - a1) / (a0 - 0.5)) if a0 > 0.5 else np.nan


def main():
    data = load()
    res = {a: {"steps": D["n_all"], "steps_with_state": D["n_ok"],
               **{f"L1_{k}": round(l1(D, k, np.arange(D["n_ok"])), 4) for k in ("R4", "D16")},
               **{f"L3_{k}": round(l3(D, k, np.arange(D["n_ok"])), 4) for k in ("R4", "D16")}} for a, D in data.items()}
    rng = np.random.default_rng(SEED)
    keys = {"K1_L1_D16": lambda D, i: l1(D, "D16", i), "K2_L1_D16_minus_R4": lambda D, i: l1(D, "D16", i) - l1(D, "R4", i),
            "L3_D16_minus_R4": lambda D, i: l3(D, "D16", i) - l3(D, "R4", i)}
    B = {k: [] for k in keys}
    for _ in range(B_N):
        v = {k: [] for k in keys}
        for D in data.values():
            us = np.unique(D["sc"])
            idx = np.concatenate([np.flatnonzero(D["sc"] == u) for u in rng.choice(us, len(us))])
            for k, f in keys.items():
                v[k].append(f(D, idx))
        for k in keys:
            B[k].append(np.nanmean(v[k]))
    for k, f in keys.items():
        res[k] = [round(float(np.nanmean([f(D, np.arange(D["n_ok"])) for D in data.values()])), 4),
                  round(float(np.nanpercentile(B[k], 2.5)), 4), round(float(np.nanpercentile(B[k], 97.5)), 4)]
    res["pooled_agents"] = list(data)
    res["endpoints"] = {"K1_lower_gt_0": res["K1_L1_D16"][1] > 0, "K2_lower_gt_0": res["K2_L1_D16_minus_R4"][1] > 0}
    (ROOT / "results/t1/g35b_dither_reads_state.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
