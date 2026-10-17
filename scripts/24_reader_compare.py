#!/usr/bin/env python
"""Who should you ask about an agent: the agent itself (black box), or an open reader model (white box on the reader)?

Same CAPE turn-start units (Gemma-4 agent, development pilot). Monitors of Gemma's violation at this turn, all
out of fold (GroupKFold by scenario):
  PSI_self        Gemma's own answers to the battery (black-box access to the agent only)
  PSI_reader_<r>  reader r answering the same battery in the agent's place (black box on the reader)
  RP_<r>_L<l>     logistic probe on reader r's hidden state at the same turn start (white box on the reader:
                  the "Model Hermeneutics" setting)
  WB_self_L<l>    probe on Gemma's own hidden state (needs the agent's weights)
  TXT             TF-IDF text monitor
Readers: Qwen3-8B, Qwen3.8-27B. Metric: within-(template x tool) AUROC; bootstrap of PSI_self − best reader probe.

python scripts/24_reader_compare.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.stats import cluster_boot, fisher_rao, within_auroc, yes_prob  # noqa: E402

pil = importlib.import_module("03_pilot_analysis")
I = Path("/data/models/graybox_ws/outputs/interrogate")
RES = Path("/home/arya/graybox/results/pilot")


def psi_feats(z):
    q = yes_prob(z["p_yes"], z["p_no"])
    lo = np.clip(np.log(z["p_yes"] + 1e-12) - np.log(z["p_no"] + 1e-12), -25, 25)
    return np.hstack([fisher_rao(q), lo])


def main():
    tag = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("tag=")), "pilot")
    U = pd.read_parquet(I / f"gemma4_{tag}_units.parquet")
    y = (U.cls == "VIOL").astype(int).values if tag == "pilot" else U.y.values.astype(int)
    g, st = U.scenario_id.values, U.template.values
    lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))  # noqa: E731
    S = {"PSI_self": pil.oof(psi_feats(np.load(I / f"gemma4_{tag}.npz", allow_pickle=True)), y, g, lambda: lr(0.1), 5)}
    if tag == "pilot":
        S["TXT"] = pd.read_parquet(RES / "gemma4_pilot_scores.parquet")["TXT"].values

    def probes(path, prefix):
        if not path.exists():
            return
        Hz = np.load(path)
        for j, l in enumerate(Hz["layers"]):
            X = PCA(256, random_state=0).fit_transform(StandardScaler().fit_transform(Hz["h"][:, j].astype(np.float32)))
            S[f"{prefix}_L{l}"] = pil.oof(X, y, g, lambda: lr(0.05), 5)

    probes(I / f"gemma4_{tag}_hidden.npz", "WB_self")
    for r in ("qwen3_8b", "qwen38"):
        stem = I / f"gemma4_{tag}_reader-{r}"
        if Path(f"{stem}.npz").exists():
            Ur = pd.read_parquet(f"{stem}_units.parquet")
            assert (Ur.episode_id.values == U.episode_id.values).all()
            S[f"PSI_reader_{r}"] = pil.oof(psi_feats(np.load(f"{stem}.npz", allow_pickle=True)), y, g, lambda: lr(0.1), 5)
        probes(Path(f"{stem}_hidden.npz"), f"RP_{r}")
    pos, neg = ("VIOL", "XOTHER") if tag == "pilot" else ("XUNSAFE", "XSAFE")
    contrasts = {"all": np.ones(len(y), bool), "vs_XOTHER": U.cls.isin([pos, neg]).values,
                 "vs_XPLUS": U.cls.isin([pos, "XPLUS"]).values}
    res = {"auroc": {cn: {k: round(within_auroc(y[m], v[m], st[m]), 4) for k, v in S.items()} for cn, m in contrasts.items()}}
    for fam in ("RP_qwen3_8b", "RP_qwen38", "WB_self"):
        keys = [k for k in S if k.startswith(fam)]
        if not keys:
            continue
        best = max(keys, key=lambda k: res["auroc"]["all"][k])          # best layer chosen on the same data: favours the probe
        d = cluster_boot(lambda i: within_auroc(y[i], S["PSI_self"][i], st[i]) - within_auroc(y[i], S[best][i], st[i]), g, 1000)
        res[f"PSI_self_minus_{best}"] = [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4),
                                         round(float(np.percentile(d, 97.5)), 4)]
    (RES / f"reader_compare_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
