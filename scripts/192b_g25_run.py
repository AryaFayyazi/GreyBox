#!/usr/bin/env python
"""Runs the locked scripts/192 (G25) with one fix (docs/deviation_G25_2026-10-22.md): for 8-question targets, G23's
stats() is replaced by the same computation without its two 16-question split read-outs (tailored 8 = all 8 here;
universal 8 does not exist). SNR, weights, endpoints and bootstrap: G23's code, unchanged."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("g25", ROOT / "scripts/192_g25_fisher_snr_own.py")
g25 = importlib.util.module_from_spec(s)
s.loader.exec_module(g25)
G23 = g25.G23
_stats = G23.stats


def stats(C, sg, y, st):
    if C.shape[1] == 16:
        return _stats(C, sg, y, st)
    T = G23.theta(C) * sg                                       # the lines of G23.stats(), for 8 questions
    au = np.array([G23.within_auroc(y, T[:, j], st) for j in range(T.shape[1])])
    r = G23.snr(C, st)
    w = np.clip(r, 0, None)
    rv = G23.raw_var(C, st)
    return {"rho_snr": spearmanr(r, au).correlation, "rho_rawvar": spearmanr(rv, au).correlation,
            "auc_eq16": G23.within_auroc(y, T.mean(1), st), "auc_snr16": G23.within_auroc(y, (T * w).sum(1) / max(w.sum(), 1e-9), st),
            "auc_rawvar16": G23.within_auroc(y, (T * rv).sum(1) / max(rv.sum(), 1e-9), st),
            "auc_tailored8": G23.within_auroc(y, T.mean(1), st), "auc_universal8": float("nan"), "snr": r, "auroc": au}


G23.stats = stats
g25.main()
