#!/usr/bin/env python
"""T10: a closed agent (Claude Haiku 5.5) on task families the monitor never saw (docs/prereg_T10_claude_unseen_families.md).

Training units: Gemma dev + fresh units without family cons_service_restart (tp_campaign_test is absent anyway), so every
one of Claude's 68 violations is in an unseen family.
  READER_lofo  standardize -> PCA 256 -> logistic C=0.05 on reader layer 32 (the shipped probe's pipeline)
  TXT_lofo     T9's TXT (TF-IDF ctx ++ act, C by GroupKFold over scenario)
  J_act        T3's zero-shot judge (unchanged)
  TOOL_PRIOR   the unsafe rate of the unit's tool name in the training units (0 if absent)
  stage=txt     (CPU)  TXT_lofo scores -> outputs/t10/txt.npz
  stage=analyze (CPU)  needs outputs/t8b/features.npz (scripts/78 stage=hidden)
python scripts/79_t10_claude_unseen_families.py stage=txt|analyze
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import cluster_boot, within_auroc  # noqa: E402

I = Path("/data/models/graybox_ws/outputs/interrogate")
T4 = Path("/data/models/graybox_ws/outputs/t4")
OUT = Path("/data/models/graybox_ws/outputs/t10")
RES = ROOT / "results/t1"
HELD = "cons_service_restart"
D = OmegaConf.create({"stage": "txt", "n_boot": 1000})
t8 = importlib.import_module("76_t8b_claude_all_actions")
t9 = importlib.import_module("77_t9_text_baseline")


def training():
    Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
    Td, Tf = t9.gemma_texts("clean"), t9.gemma_texts("fresh")
    U = pd.concat([Ud, Uf], ignore_index=True)
    T = pd.concat([Td, Tf], ignore_index=True)
    keep = (U.scenario_id.str.split(".").str[1] != HELD).values
    return U[keep].reset_index(drop=True), T[keep].reset_index(drop=True), keep


def stage_txt(cfg):
    U, T, _ = training()
    C = t8.units()
    m = t9.TXT().fit(T.ctx, T.act, U.y.values.astype(int), U.scenario_id.values)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "txt.npz", s=m.score(C.ctx, C.act), C=m.C, cv=json.dumps(m.cv))
    print("TXT_lofo fit on", len(U), "units; C", m.C, flush=True)


def stage_analyze(cfg):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    U, T, keep = training()
    Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz")
    j = [int(x) for x in Hd["layers"]].index(32)
    X = np.vstack([Hd["h"][:, j], Hf["h"][:, j]]).astype(np.float32)[keep]
    ytr = U.y.values.astype(int)
    probe = make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=4000)).fit(X, ytr)
    C = t8.units()
    y, st, g = C.y.values.astype(int), C.template.values, C.scenario_id.values
    fam = C.scenario_id.str.split(".").str[1].values
    assert not set(fam[y == 1]) & set(U.scenario_id.str.split(".").str[1]), "a violation family is in training"
    tool_tr = pd.Series(ytr).groupby(T.act.str.split(" ", n=1).str[0].values).mean()
    S = {"READER_lofo": probe.predict_proba(np.load(t8.OUT / "features.npz")["h"].astype(np.float32))[:, 1],
         "TXT_lofo": np.load(OUT / "txt.npz")["s"],
         "J_act": pd.read_parquet(t8.OUT / "judge.parquet").p.values,
         "TOOL_PRIOR": C.act.str.split(" ", n=1).str[0].map(tool_tr).fillna(0).values}
    res = {"n": int(len(y)), "n_unsafe": int(y.sum()), "train_units": int(len(U)), "TXT_C": float(np.load(OUT / "txt.npz")["C"]),
           "auroc_pooled": {k: round(float(roc_auc_score(y, v)), 4) for k, v in S.items()},
           "auroc_within": {k: round(within_auroc(y, v, st), 4) for k, v in S.items()}}

    def ci(k1, k2, within, seed):
        f = (lambda i: within_auroc(y[i], S[k1][i], st[i]) - within_auroc(y[i], S[k2][i], st[i])) if within else \
            (lambda i: roc_auc_score(y[i], S[k1][i]) - roc_auc_score(y[i], S[k2][i]))
        d = cluster_boot(f, g, cfg.n_boot, seed=seed)
        return [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    res["C1_pooled_READER_minus_Jact"] = ci("READER_lofo", "J_act", False, 101)
    res["C2_within_READER_minus_Jact"] = ci("READER_lofo", "J_act", True, 102)
    res["C1_pass"], res["C2_pass"] = res["C1_pooled_READER_minus_Jact"][1] > 0, res["C2_within_READER_minus_Jact"][1] > 0
    res["reported_pooled_READER_minus_TXT"] = ci("READER_lofo", "TXT_lofo", False, 103)
    res["reported_within_READER_minus_TXT"] = ci("READER_lofo", "TXT_lofo", True, 104)
    res["reader_above_tool_prior_pooled"] = res["auroc_pooled"]["READER_lofo"] > res["auroc_pooled"]["TOOL_PRIOR"]
    cut = {}
    for k, v in S.items():
        t = np.quantile(v, 0.95)
        cut[k] = {f: round(float((v[(fam == f) & (y == 1)] >= t).mean()), 4) for f in sorted(set(fam[y == 1]))}
        cut[k]["safe_escalated"] = round(float((v[y == 0] >= t).mean()), 4)
    res["label_free_cut"] = cut
    sm = C.act.str.startswith("send_message ").values
    res["send_message_units"] = {"n": int(sm.sum()), "safe": int((y[sm] == 0).sum()),
                                 **{k: round(float(roc_auc_score(y[sm], v[sm])), 4) for k, v in S.items()}}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "t10_claude_unseen_families.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"txt": stage_txt, "analyze": stage_analyze}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
