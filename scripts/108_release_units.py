#!/usr/bin/env python
"""Release the per-unit data behind the headline confirmatory tests, so that outsiders can recompute them
(reviewer finding: "no confirmatory number can be recomputed from public data"). No transcripts are released: only
ids, strata, clusters, labels and each monitor's score per unit, the 128-component true and replicated states for the
replication tests (float16), and the per-episode outcomes of the live tests. Every array is rebuilt with the test's own
deterministic fit and checked against the saved result before it is written.
Output: results/units/*.csv.gz and *.npz; recompute with scripts/109_recompute_public.py (no private files needed).
python scripts/108_release_units.py
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"                       # as the published analyses: BLAS results depend on the thread count
import importlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.stats import within_auroc  # noqa: E402

W, I, T4 = Path("/data/models/graybox_ws/outputs"), Path("/data/models/graybox_ws/outputs/interrogate"), Path("/data/models/graybox_ws/outputs/t4")
OUT, RES = ROOT / "results/units", ROOT / "results/t1"
j = lambda f: json.loads((RES / f).read_text())  # noqa: E731


def close(a, b, tol, what):
    assert abs(float(a) - float(b)) <= tol, f"{what}: rebuilt {a} vs saved {b}"


def tool_prior():
    t11 = importlib.import_module("81_t11_gpt_monitor")
    _, Ug, T = t11.gemma_train()
    return pd.Series(Ug.y.values.astype(int)).groupby(T.act.str.split(" ", n=1).str[0].values).mean()


def closed_agents(rate):
    from sklearn.metrics import roc_auc_score
    t8b, t11 = importlib.import_module("76_t8b_claude_all_actions"), importlib.import_module("81_t11_gpt_monitor")
    U = t8b.units()
    d = pd.DataFrame({"episode_id": U.episode_id, "scenario_id": U.scenario_id, "template": U.template, "tool": U.act.str.split(" ", n=1).str[0],
                      "y": U.y.astype(int), "READER": np.load(W / "t8b/reader.npz")["p"], "J_act": pd.read_parquet(W / "t8b/judge.parquet").p.values})
    d["TOOL_PRIOR"] = d.tool.map(rate).fillna(0).values
    close(within_auroc(d.y.values, d.READER.values, d.template.values), j("t8b_claude_all_actions.json")["auroc"]["READER"], 1e-4, "T8b READER")
    d.to_csv(OUT / "t8b_claude.csv.gz", index=False)
    U = t11.units()
    d = pd.DataFrame({"episode_id": U.episode_id, "scenario_id": U.scenario_id, "template": U.template, "tool": U.tool, "y": U.y.astype(int),
                      "READER": np.load(W / "t11/reader.npz")["p"], "J_act": pd.read_parquet(W / "t11/judge.parquet").p.values,
                      "TXT": np.load(W / "t11/txt.npz")["s"]})
    d["TOOL_PRIOR"] = d.tool.map(rate).fillna(0).values
    close(roc_auc_score(d.y, d.READER), j("t11_gpt_agent.json")["auroc_pooled"]["READER"], 1e-4, "T11 READER")
    d.to_csv(OUT / "t11_gpt.csv.gz", index=False)


def agentdojo():
    """T13b's units with the T13b reader, the judge, and T14's two fits (k = 0 and all 3,455 open-agent calls)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t7, t13b = importlib.import_module("73_t7_agentdojo"), importlib.import_module("86_t13b_analysis")
    UA = pd.concat([pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")], ignore_index=True)
    Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz")
    jj = [int(x) for x in Hd["layers"]].index(32)
    XC, yC = np.vstack([Hd["h"][:, jj], Hf["h"][:, jj]]).astype(np.float32), UA.y.values.astype(int)
    A7 = t7.units()
    X7, y7 = np.load(t7.OUT / "reader.npz")["h"].astype(np.float32), A7.y.values.astype(int)
    G = t13b.units()
    XG = np.load(t13b.OUT / "reader.npz")["h"].astype(np.float32)
    fits = {}
    for k, add in (("T14_zero", np.arange(0)), ("T14_full", np.arange(len(X7)))):
        m = make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=0.05, max_iter=4000))
        fits[k] = m.fit(np.vstack([XC, X7[add]]), np.concatenate([yC, y7[add]])).predict_proba(XG)[:, 1]
    conf = {json.loads(l)["key"] for f in ("conf_gpt6luna_none_attack.jsonl", "conf_gpt6luna_none_benign.jsonl") for l in open(W / "agentdojo_gpt" / f)}
    d = pd.DataFrame({"episode": G.episode, "suite": G.suite, "kind": G.kind, "split": np.where(G.episode.isin(conf), "conf", "dev"), "y": G.y.astype(int),
                      "READER_T13b": np.load(t13b.OUT / "reader.npz")["p"], "J_act": pd.read_parquet(t13b.OUT / "judge.parquet").p.values, **fits})
    close(within_auroc(d.y.values, d.READER_T13b.values, d.suite.values), j("t13b_gpt_agentdojo_none.json")["auroc_within_suite"]["READER"], 1e-4, "T13b READER")
    close(within_auroc(d.y.values, d.T14_full.values, d.suite.values), j("t14_agentdojo_labels.json")["curve"]["3455"]["auroc"], 1e-4, "T14 full")
    d.to_csv(OUT / "agentdojo_gpt_t13b_t14.csv.gz", index=False)


def pcs(Yd_full, Yf_full):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Yd_full)
    pca = PCA(128, random_state=0).fit(sc.transform(Yd_full))
    return pca.transform(sc.transform(Yd_full)), pca.transform(sc.transform(Yf_full))


def ridge_rec(Xd, Xf, Yd, alpha, dense):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    if dense:
        xs = StandardScaler().fit(Xd)
        Xd, Xf = xs.transform(Xd), xs.transform(Xf)
    return Ridge(alpha=alpha).fit(Xd, Yd).predict(Xf)


def probe_scores(Yd, yd, Yf, recs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000)).fit(Yd, yd)
    return {"probe_true": p.decision_function(Yf), **{f"probe_{k}": p.decision_function(v) for k, v in recs.items()}}


def replication_t15():
    from sklearn.metrics import r2_score
    t9 = importlib.import_module("77_t9_text_baseline")
    r = j("t15_latent_replication.json")
    Ud, Uf = pd.read_parquet(I / "gemma4_clean_units.parquet"), pd.read_parquet(I / "gemma4_fresh_units.parquet")
    Dd, Df = np.load(I / "gemma4_clean_decision_hidden.npz"), np.load(I / "gemma4_fresh_decision_hidden.npz")
    okd, okf = Dd["ok"].astype(bool), Df["ok"].astype(bool)
    Yd, Yf = pcs(Dd["h"][okd].reshape(okd.sum(), -1).astype(np.float32), Df["h"][okf].reshape(okf.sum(), -1).astype(np.float32))
    Hd, Hf = np.load(T4 / "clean_act.npz"), np.load(T4 / "fresh_act.npz")
    jj = [int(x) for x in Hd["layers"]].index(32)
    Td, Tf = t9.gemma_texts("clean"), t9.gemma_texts("fresh")
    tx = t9.TXT()
    tx.v = [t9.vec(), t9.vec()]
    Xtd, Xtf = tx._X(Td.ctx.values[okd], Td.act.values[okd], True), tx._X(Tf.ctx.values[okf], Tf.act.values[okf], False)
    recs = {"READER": ridge_rec(Hd["h"][okd, jj].astype(np.float32), Hf["h"][okf, jj].astype(np.float32), Yd, r["alpha"]["READER"]["chosen"], True),
            "TXT": ridge_rec(Xtd, Xtf, Yd, r["alpha"]["TXT"]["chosen"], False)}
    for k in recs:
        close(r2_score(Yf, recs[k], multioutput="variance_weighted"), r["r2_fresh"][k], 2e-4, f"T15 R2 {k}")
    sc = probe_scores(Yd, Ud.y.values[okd].astype(int), Yf, recs)
    np.savez_compressed(OUT / "t15_gemma_replication.npz", Y_true=Yf.astype(np.float16), Y_reader=recs["READER"].astype(np.float16),
                        Y_text=recs["TXT"].astype(np.float16), y=Uf.y.values[okf].astype(np.int8), template=np.array([str(x) for x in Uf.template.values[okf]], dtype=str),
                        scenario=np.array([str(x) for x in Uf.scenario_id.values[okf]], dtype=str), **{k: v.astype(np.float32) for k, v in sc.items()})


def replication_t17():
    from sklearn.metrics import r2_score
    m = importlib.import_module("96_t17_replication_agents")
    t9 = importlib.import_module("77_t9_text_baseline")
    r17 = j("t17_replication_agents.json")["agents"]
    for a in m.AGENTS:
        U = pd.read_parquet(m.units_path(a))
        Hd = np.load(m.OUT / f"{a}_decision_hidden.npz")
        Xr, T = m.reader_features(a), m.texts(a)
        ok, dev = Hd["ok"].astype(bool), m.is_dev(U.scenario_id.values)
        d, f = ok & dev, ok & ~dev
        Yfull = Hd["h"].reshape(len(U), -1).astype(np.float32)
        Yd, Yf = pcs(Yfull[d], Yfull[f])
        tx = t9.TXT()
        tx.v = [t9.vec(), t9.vec()]
        recs = {"READER": ridge_rec(Xr[d], Xr[f], Yd, r17[a]["alpha"]["READER"]["chosen"], True),
                "TXT": ridge_rec(tx._X(T.ctx.values[d], T.act.values[d], True), tx._X(T.ctx.values[f], T.act.values[f], False), Yd, r17[a]["alpha"]["TXT"]["chosen"], False)}
        for k in recs:
            close(r2_score(Yf, recs[k], multioutput="variance_weighted"), r17[a]["r2_fresh"][k], 2e-4, f"T17 {a} R2 {k}")
        sc = probe_scores(Yd, U.y.values[d].astype(int), Yf, recs)
        np.savez_compressed(OUT / f"t17_{a}_replication.npz", Y_true=Yf.astype(np.float16), Y_reader=recs["READER"].astype(np.float16),
                            Y_text=recs["TXT"].astype(np.float16), y=U.y.values[f].astype(np.int8), template=np.array([str(x) for x in m.stratum(U.template.values[f])], dtype=str),
                            scenario=np.array([str(x) for x in U.scenario_id.values[f]], dtype=str), **{k: v.astype(np.float32) for k, v in sc.items()})


def live():
    cols = ["instance_id", "scenario_id", "template", "arm", "status", "unsafe_episode", "goal", "blocked"]
    parts = []
    for test, f in (("T16", "t16/run.partial.jsonl"), ("T16b", "t16b/run.partial.jsonl"), ("T16c", "t16c/run.partial.jsonl")):
        R = pd.DataFrame([json.loads(l) for l in open(W / f)])
        R["held"] = R["held"] if "held" in R else 0
        R["blocked_would_be_unsafe"] = R["blocked_would_be_unsafe"] if "blocked_would_be_unsafe" in R else np.nan
        parts.append(R[cols + ["held", "blocked_would_be_unsafe"]].assign(test=test))
    d = pd.concat(parts, ignore_index=True)
    for t, key in (("T16", "t16_live_gate.json"), ("T16b", "t16b_claude_live_gate.json")):
        x = d[(d.test == t) & (d.arm == "gated")].unsafe_episode.mean()
        close(x, j(key)["unsafe_episode_rate"]["gated"], 1e-3, f"{t} gated rate")
    d.to_csv(OUT / "live_t16_t16b_t16c_episodes.csv.gz", index=False)


def tau2():
    """T18 (and T20 when it exists): every labelled write, with the four monitors' scores, rebuilt as in scripts/103."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t9 = importlib.import_module("77_t9_text_baseline")
    D = W / "tau2_claude"
    tr = pd.read_parquet(D / "t18_train.parquet")
    Htr = np.load(D / "t18_train_reader.npz")
    reader = make_pipeline(StandardScaler(), PCA(min(256, len(tr) - 1), random_state=0), LogisticRegression(C=0.05, max_iter=4000)).fit(Htr["h"].astype(np.float32), tr.y.values)
    txt = t9.TXT().fit(tr.ctx.values, tr.act.values, tr.y.values, tr.task_id.values)
    rate = tr.groupby("tool").y.mean()
    for name, test, feats, key in (("t18_tau2_claude", "t18_test_gb_tau2_claude_conf.parquet", "t18_test_gb_tau2_claude_conf_reader.npz", "t18_tau2_claude.json"),
                                   ("t20_tau2_two_closed", "t20_test.parquet", "t20_test_reader.npz", "t20_tau2_two_closed.json")):
        if not (D / feats).exists() or not (RES / key).exists():
            print("skip", name)
            continue
        te, H = pd.read_parquet(D / test), np.load(D / feats)
        d = pd.DataFrame({"sim_id": te.sim_id, "task_id": te.task_id, "tool": te.tool, "agent": te["agent"] if "agent" in te else "claude", "y": te.y.astype(int),
                          "READER": reader.decision_function(H["h"].astype(np.float32)), "TXT": txt.score(te.ctx.values, te.act.values),
                          "TOOL": te.tool.map(rate).fillna(tr.y.mean()).values, "CAPE0": H["cape0"]})
        close(within_auroc(d.y.values, d.READER.values, d.tool.values), j(key)["auroc_within_tool"]["READER"], 1e-4, f"{name} READER")
        d.to_csv(OUT / f"{name}.csv.gz", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rate = tool_prior()
    for f in (lambda: closed_agents(rate), agentdojo, live, tau2, replication_t15, replication_t17):
        f()
        print("done", getattr(f, "__name__", "closed_agents"), flush=True)
    print({p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(OUT.iterdir())})


if __name__ == "__main__":
    main()
