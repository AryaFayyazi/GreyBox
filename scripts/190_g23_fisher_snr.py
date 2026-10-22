#!/usr/bin/env python
"""G23: does the theory predict, without labels, which questions monitor? (docs/prereg_G23_fisher_snr.md)
In arcsine (Fisher-Rao) coordinates the sampling noise of an answer angle is known and constant: for m parsed answers,
Var(2 arcsin sqrt((k+3/8)/(m+3/4))) ~= 1/(m + 1/2) (Anscombe), whatever the answer law. So, with no label, each
question's signal-to-noise ratio across unlabelled calls is
    SNR_j = (pooled within-stratum variance of theta_j) / mean_i 1/(m_ij + 1/2)  -  1,
the share of its spread that is not sampling noise. (On raw P(yes) the noise p(1-p)/m depends on p, so no such
correction exists.) Predictions: questions with higher SNR rank violations better; weighting questions by SNR, with no
label, gives a better read-out than weighting them equally.
Pool: Gemma-4's answers about the same calls to 16 questions, the 8 tailored ones (G16 on CAPE-Bench, G21 on
tau2-bench) and the 8 universal ones (G22). Targets: CAPE-Bench held-out calls of gpt-oss-20b, Qwen3.8, Muse-Glimmer
and Claude Haiku 5.5's actions (strata: template; clusters: scenario), and Claude's tau2-bench writes (strata: tool;
clusters: task). Nothing is fit to labels. CPU only.  python scripts/190_g23_fisher_snr.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from graybox.stats import within_auroc  # noqa: E402

DG = Path("/data/models/graybox_ws/outputs/decision_greybox")
UN = Path("/data/models/graybox_ws/outputs/universal")
B_N, SEED = 2000, 2300


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def counts(col):
    return np.array([json.loads(c) if isinstance(c, str) else c for c in col], float)


def targets():
    """[(name, C (units x 16 x 2), signs (16,), y, strata, clusters)]"""
    M = _mod("m171", "scripts/171_decision_greybox_LOCKED.py")
    OR = _mod("obsref", "web/build_observable_ref.py")
    U22 = _mod("g22", "scripts/189_g22_universal_questions.py")
    g21 = _mod("g21", "scripts/188_g21_tau2_open_answerer.py")
    su = np.array([s for _, s in U22.UNIVERSAL])
    k = ["episode_id", "event_id"]
    out = []
    for t in ["gptoss20b", "qwen38", "museglimmer", "claude"]:
        tag = "claude_t8b" if t == "claude" else f"{t}_heldout"
        Tl = pd.read_parquet(DG / f"{tag}_by_gemma4.parquet")
        J = Tl.merge(pd.read_parquet(UN / f"{tag}_universal_gemma4.parquet")[k + ["counts"]].rename(columns={"counts": "cu"}), on=k)
        if t == "claude":
            J = J[~J.set_index(k).index.isin(M.PILOT)]
        else:
            Us, _ = M.states(t)
            bad = set(zip(Us.episode_id[~Us.ok.values], Us.event_id[~Us.ok.values].astype(int)))
            J = J[[(e, int(v)) not in bad for e, v in zip(J.episode_id, J.event_id)]]
        J = J.reset_index(drop=True)
        C = np.concatenate([counts(J.counts), counts(J.cu)], axis=1)
        st = np.array([-1] + [1] * 7)
        out.append((t, C, np.concatenate([st, su]), J.y.values.astype(int), J.template.values, J.scenario_id.values))
    Tl = pd.read_parquet(g21.OUT / "g21_gemma4.parquet")
    Un = pd.read_parquet(UN / "tau2_g18_universal_gemma4.parquet")
    assert (Tl.sim_id.values == Un.sim_id.values).all() and (Tl.msg_idx.values == Un.msg_idx.values).all()
    C = np.concatenate([counts(Tl.counts), counts(Un.counts)], axis=1)
    out.append(("tau2_claude", C, np.concatenate([np.array([s for _, s in g21.G18.QUESTIONS]), su]), Tl.y.values.astype(int),
                Tl.tool.values, Tl.task_id.values))
    return out


def theta(C):
    return 2 * np.arcsin(np.sqrt((C[..., 0] + 3 / 8) / (C[..., 1] + 3 / 4)))


def within_var(X, strata):
    """pooled within-stratum variance of each column (label-free: strata are known without labels)."""
    num, den = np.zeros(X.shape[1]), 0
    for s in np.unique(strata):
        m = strata == s
        if m.sum() > 1:
            num += ((X[m] - X[m].mean(0)) ** 2).sum(0)
            den += m.sum() - 1
    return num / max(den, 1)


def snr(C, strata):
    T = theta(C)
    noise = (1.0 / (C[..., 1] + 0.5)).mean(0)
    return within_var(T, strata) / noise - 1.0


def raw_var(C, strata):
    P = (C[..., 0] + 3 / 8) / (C[..., 1] + 3 / 4)
    return within_var(P, strata)


def stats(C, sg, y, st):
    T = theta(C) * sg
    au = np.array([within_auroc(y, T[:, j], st) for j in range(T.shape[1])])
    r = snr(C, st)
    w = np.clip(r, 0, None)
    rv = raw_var(C, st)
    s_eq = T.mean(1)
    s_w = (T * w).sum(1) / max(w.sum(), 1e-9)
    s_rv = (T * rv).sum(1) / max(rv.sum(), 1e-9)
    return {"rho_snr": spearmanr(r, au).correlation, "rho_rawvar": spearmanr(rv, au).correlation,
            "auc_eq16": within_auroc(y, s_eq, st), "auc_snr16": within_auroc(y, s_w, st), "auc_rawvar16": within_auroc(y, s_rv, st),
            "auc_tailored8": within_auroc(y, T[:, :8].mean(1), st), "auc_universal8": within_auroc(y, T[:, 8:].mean(1), st),
            "snr": r, "auroc": au}


def main():
    T = targets()
    rng = np.random.default_rng(SEED)
    res, pts = {}, {}
    for name, C, sg, y, st, cl in T:
        r = stats(C, sg, y, st)
        pts[name] = r
        res[name] = {"units": int(len(y)), "violations": int(y.sum()),
                     **{k: round(float(v), 4) for k, v in r.items() if k not in ("snr", "auroc")},
                     "snr": [round(float(v), 3) for v in r["snr"]], "auroc": [round(float(v), 4) for v in r["auroc"]]}
    keys = ["rho_snr", "rho_rawvar", "auc_eq16", "auc_snr16", "auc_rawvar16"]
    point = {k: float(np.nanmean([pts[n][k] for n in pts])) for k in keys}
    point["snr_minus_eq"] = float(np.nanmean([pts[n]["auc_snr16"] - pts[n]["auc_eq16"] for n in pts]))
    point["snr_minus_rawvar"] = float(np.nanmean([pts[n]["auc_snr16"] - pts[n]["auc_rawvar16"] for n in pts]))
    point["rho_snr_minus_rawvar"] = float(np.nanmean([pts[n]["rho_snr"] - pts[n]["rho_rawvar"] for n in pts]))
    boot = {k: [] for k in point}
    for _ in range(B_N):
        vals = {k: [] for k in point}
        for name, C, sg, y, st, cl in T:
            uc = np.unique(cl)
            idx = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(uc, len(uc))])
            r = stats(C[idx], sg, y[idx], st[idx])
            for k in keys:
                vals[k].append(r[k])
            vals["snr_minus_eq"].append(r["auc_snr16"] - r["auc_eq16"])
            vals["snr_minus_rawvar"].append(r["auc_snr16"] - r["auc_rawvar16"])
            vals["rho_snr_minus_rawvar"].append(r["rho_snr"] - r["rho_rawvar"])
        for k in point:
            boot[k].append(np.nanmean(vals[k]))
    ci = {k: [round(point[k], 4), round(float(np.nanpercentile(boot[k], 2.5)), 4), round(float(np.nanpercentile(boot[k], 97.5)), 4)] for k in point}
    res["F1_mean_spearman_snr_vs_auroc"] = ci["rho_snr"]
    res["F2_snr_weighted_minus_equal_16"] = ci["snr_minus_eq"]
    res["F3_snr_weighted_minus_rawvar_weighted_16"] = ci["snr_minus_rawvar"]
    res["reported"] = {k: ci[k] for k in ("rho_rawvar", "rho_snr_minus_rawvar", "auc_eq16", "auc_snr16", "auc_rawvar16")}
    res["endpoints"] = {"F1_lower_gt_0": ci["rho_snr"][1] > 0}
    (ROOT / "results/t1/g23_fisher_snr.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or k in ("endpoints", "reported")}, indent=1))


if __name__ == "__main__":
    main()
