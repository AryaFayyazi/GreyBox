#!/usr/bin/env python
"""T9 sensitivity (exploratory, not preregistered): the locked C grid {0.1, 1, 10} chose C = 10, its upper edge, in
every fit. Refit TF-IDF (TXT, the locked features) with C fixed at 100 and 1000 on the same training sets, and report
the same AUROCs as T9 (no bootstrap). If a larger C closed the gap on unseen families or AgentDojo, T9's reading would
change.
python scripts/77b_t9_c_sensitivity.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import within_auroc  # noqa: E402

t9 = importlib.import_module("77_t9_text_baseline")
t6, t7, t8 = (importlib.import_module(m) for m in ("71_t6_cross_agent", "73_t7_agentdojo", "76_t8b_claude_all_actions"))


def fit_fixed(ctx, act, y, C):
    from sklearn.linear_model import LogisticRegression
    m = t9.TXT()
    m.v = [t9.vec(), t9.vec()]
    m.C, m.lr = C, LogisticRegression(C=C, max_iter=4000).fit(m._X(np.asarray(ctx, object), np.asarray(act, object), True), y)
    return m


def main():
    I = t9.I
    Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
    Td, Tf = t9.gemma_texts("clean"), t9.gemma_texts("fresh")
    yd, yf = Ud.y.values.astype(int), Uf.y.values.astype(int)
    Uo, oc, oa, _, _ = t6.gptoss_prompts()
    A, Cl = t7.units(), t8.units()
    fd, ff = Ud.scenario_id.str.split(".").str[1].values, Uf.scenario_id.str.split(".").str[1].values
    fams = sorted(set(fd) | set(ff))
    fold = {f: k % 5 for k, f in enumerate(fams)}
    ok = np.isfinite(np.load(I / "gemma4_fresh_decision_hidden.npz")["I_dec"])
    ca, aa, ya = pd.concat([Td.ctx, Tf.ctx]).values, pd.concat([Td.act, Tf.act]).values, np.concatenate([yd, yf])
    res = {"note": "exploratory; TXT features as locked in T9; C fixed (no CV)", "locked_C10": {k: json.loads((t9.RES / "t9_text_baseline.json").read_text())[k]["auroc"]["TXT"]
           for k in ("T4_same_scenarios", "T5_unseen_families", "T6_gptoss", "T7_agentdojo", "T8b_claude")}}
    for C in (100.0, 1000.0):
        r = {}
        m = fit_fixed(Td.ctx, Td.act, yd, C)
        r["T4_same_scenarios"] = within_auroc(yf, m.score(Tf.ctx, Tf.act), Uf.template.values)
        r["T6_gptoss"] = within_auroc(Uo.y.values.astype(int), m.score(oc, oa), Uo.template.values)
        s = np.full(len(yf), np.nan)
        for k in range(5):
            tr, te = np.array([fold[f] != k for f in fd]), np.array([fold[f] == k for f in ff])
            s[te] = fit_fixed(Td.ctx[tr], Td.act[tr], yd[tr], C).score(Tf.ctx[te], Tf.act[te])
        r["T5_unseen_families"] = within_auroc(yf[ok], s[ok], Uf.template.values[ok])
        M = fit_fixed(ca, aa, ya, C)
        r["T7_agentdojo"] = within_auroc(A.y.values, M.score(A.ctx, A.act), A.suite.values)
        sc = M.score(Cl.ctx, Cl.act)
        yc = Cl.y.values.astype(int)
        r["T8b_claude"] = within_auroc(yc, sc, Cl.template.values)
        r["T8b_caught_label_free_cut"] = float((sc[yc == 1] >= np.quantile(sc, .95)).mean())
        res[f"C{int(C)}"] = {k: round(float(v), 4) for k, v in r.items()}
        print(C, res[f"C{int(C)}"], flush=True)
    (t9.RES / "t9_c_sensitivity_exploratory.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
