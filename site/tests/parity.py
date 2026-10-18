#!/usr/bin/env python
"""Parity of the in-browser text monitor (site/src/assets/tm-core.js) with the Python SDK, in headless Chromium.
1. render: SDK ctx_text/act_text vs TM.render on 300 structured Claude agent steps (T8b episodes) + synthetic edge cases
2. model: TextMonitor (scikit-learn, C=10) vs TM.fit on the same 1,200 Gemma dev units; scored on 600 held-out units:
   vocabulary sizes, Spearman correlation of decision values, AUROC of each
3. privacy filter: TextMonitor.fit(privacy=True) vs TM.fit with TM.privacyDrop on the same units: identical vocabularies
python site/tests/parity.py   (needs the playwright venv for the browser; run with the cape venv)
"""
import importlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.monitor import Step, TextMonitor, act_text, ctx_text  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# 1. structured steps from Claude episodes
E = pd.read_parquet("/data/models/graybox_ws/outputs/claude_agent/run.parquet").iloc[:120]
S = {json.loads(l)["instance_id"]: json.loads(l) for l in open("/data/models/cape_ws/outputs/capebench/core24b/scenarios.jsonl")}
struct = []
for ep in E.itertuples():
    spec, hist = S[ep.instance_id], []
    for e in ep.events:
        if not e["is_tool_call"] or e["tool"] is None:
            continue
        args = json.loads(e["args_json"] or "{}")
        struct.append({"instructions": spec["system_prompt"], "task": spec["task_prompt"],
                       "history": [{"tool": h.tool, "args": h.args, "result": h.result} for h in hist], "action": {"tool": e["tool"], "args": args}})
        hist.append(Step(e["tool"], args, e["tool_result"] or ""))
struct = struct[:300]
struct += [{"instructions": "Règles: ne pas envoyer 😀", "task": "tâche\tavec\u0007contrôle", "history": [{"tool": "t", "args": {"a": [1, 2.5, None, True], "é": {"x": "\"q\""}}, "result": "x" * 2500}],
            "action": {"tool": "send", "args": {"to": "a@b.c", "n": -3}}},
           {"instructions": "I" * 7000, "task": "T" * 9000, "history": [{"tool": "r", "args": {}, "result": "R" * 1990}] * 12, "action": {"tool": "__final__", "args": None}}]
expect = [{"ctx": ctx_text(x["instructions"], x["task"], [Step(h["tool"], h["args"], h["result"]) for h in x["history"]]),
           "act": act_text(x["action"]["tool"], x["action"]["args"])} for x in struct]

# 2. model parity data
t9 = importlib.import_module("77_t9_text_baseline")
U = pd.read_parquet("/data/models/graybox_ws/outputs/interrogate/gemma4_clean_units.parquet")
T = t9.gemma_texts("clean")
rng = np.random.default_rng(0)
idx = rng.permutation(len(U))[:1800]
tr, te = idx[:1200], idx[1200:]
rows_tr = [{"context": T.ctx[i], "action": T.act[i], "label": int(U.y[i])} for i in tr]
rows_te = [{"context": T.ctx[i], "action": T.act[i], "label": int(U.y[i])} for i in te]
tm = TextMonitor(C=10).fit([(None, None, None, None)] * 0, [], None) if False else None
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
mk = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)  # noqa: E731
v1, v2 = mk(), mk()
Xtr = sp.hstack([v1.fit_transform([r["context"] for r in rows_tr]), v2.fit_transform([r["action"] for r in rows_tr])]).tocsr()
lr = LogisticRegression(C=10, max_iter=4000).fit(Xtr, [r["label"] for r in rows_tr])
Xte = sp.hstack([v1.transform([r["context"] for r in rows_te]), v2.transform([r["action"] for r in rows_te])]).tocsr()
py_dec = lr.decision_function(Xte)

fx = {"struct": struct, "train": rows_tr, "test": rows_te}
tmp = Path(tempfile.mkdtemp())
(tmp / "fx.json").write_text(json.dumps(fx))
(tmp / "tm-core.js").write_text((ROOT / "site/src/assets/tm-core.js").read_text())
(tmp / "run.html").write_text("""<!doctype html><meta charset=utf-8><script src="tm-core.js"></script><script>
fetch('fx.json').then(r=>r.json()).then(fx=>{
  const rend = fx.struct.map(x=>TM.render(x));
  const t0 = performance.now();
  const m = TM.fit(fx.train.map(TM.render), fx.train.map(r=>r.label), 10);
  const dec = fx.test.map(r=>TM.decision(m, TM.features(m, TM.render(r))));
  const rt = fx.test.map(TM.render), t1 = performance.now();
  for (let k = 0; k < 3; k++) rt.forEach(r => TM.decision(m, TM.features(m, r)));
  const scoreMs = (performance.now() - t1) / (3 * rt.length);
  m.format = 'graybox-text-monitor/1'; m.threshold_logit = 0;
  const mp = TM.fit(fx.train.map(TM.render), fx.train.map(r=>r.label), 10, {drop: TM.privacyDrop});
  const decp = fx.test.map(r=>TM.decision(mp, TM.features(mp, TM.render(r))));
  window.OUT = {rend, dec, nvc: m.vc.idf.length, nva: m.va.idf.length, ms: performance.now()-t0, model: m, scoreMs,
                privacy: {vc: Object.keys(mp.vc.vocab), va: Object.keys(mp.va.vocab), dec: decp}};
});</script>""")
drv = tmp / "drive.py"
drv.write_text(f"""
import json, http.server, threading, functools
from playwright.sync_api import sync_playwright
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory="{tmp}")
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto(f"http://127.0.0.1:{{srv.server_port}}/run.html"); pg.wait_for_function("window.OUT", timeout=600000)
    open("{tmp}/out.json", "w").write(json.dumps(pg.evaluate("window.OUT"))); b.close()
srv.shutdown()
""")
subprocess.run(["/data/models/graybox_ws/venv_pw/bin/python", str(drv)], check=True,
               env={"PLAYWRIGHT_BROWSERS_PATH": "/data/models/graybox_ws/pw-browsers", "PATH": "/usr/bin:/bin"})
out = json.loads((tmp / "out.json").read_text())
same = [a["ctx"] == b["ctx"] and a["act"] == b["act"] for a, b in zip(out["rend"], expect)]
bad = [i for i, s in enumerate(same) if not s]
js_dec = np.array(out["dec"])
yte = np.array([r["label"] for r in rows_te])
(tmp / "model.json").write_text(json.dumps(out["model"]))
(ROOT / "site/tests/browser_model.json").write_text(json.dumps(out["model"]))
tmj = TextMonitor.load_json(tmp / "model.json")
py_from_js = np.array([tmj.decision(None, None, None, None, rendered=(r["context"], r["action"])) for r in rows_te])
res = {"render_identical": f"{sum(same)}/{len(same)}", "render_mismatch_idx": bad[:10],
       "vocab_py": [len(v1.vocabulary_), len(v2.vocabulary_)], "vocab_js": [out["nvc"], out["nva"]],
       "spearman_decision": round(float(spearmanr(py_dec, js_dec)[0]), 5), "max_abs_decision_diff": round(float(np.max(np.abs(py_dec - js_dec))), 4),
       "auroc_py": round(float(roc_auc_score(yte, py_dec)), 4), "auroc_js": round(float(roc_auc_score(yte, js_dec)), 4), "js_fit_ms": round(out["ms"]), "js_score_ms_per_step": round(out["scoreMs"], 3),
       "roundtrip_browser_model_scored_in_python_max_abs_diff": float(np.max(np.abs(py_from_js - js_dec)))}
tmp_priv = TextMonitor(C=10).fit([{"context": r["context"], "action": r["action"]} for r in rows_tr], [r["label"] for r in rows_tr], None, privacy=True)
pv = (set(tmp_priv.v[0].vocabulary_), set(tmp_priv.v[1].vocabulary_))
jv = (set(out["privacy"]["vc"]), set(out["privacy"]["va"]))
pdec = np.array([tmp_priv.decision(None, None, None, None, rendered=(r["context"], r["action"])) for r in rows_te])
res["privacy_filter"] = {"vocab_py": [len(pv[0]), len(pv[1])], "vocab_js": [len(jv[0]), len(jv[1])], "vocab_identical": pv == jv,
                         "spearman_decision": round(float(spearmanr(pdec, np.array(out["privacy"]["dec"]))[0]), 5),
                         "auroc_py": round(float(roc_auc_score(yte, pdec)), 4), "auroc_js": round(float(roc_auc_score(yte, out["privacy"]["dec"])), 4)}
if bad:
    i = bad[0]
    a, b = out["rend"][i]["ctx"], expect[i]["ctx"]
    k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    res["first_diff"] = {"i": i, "js": a[max(0, k - 60):k + 60], "py": b[max(0, k - 60):k + 60], "act_js": out["rend"][i]["act"][:120], "act_py": expect[i]["act"][:120]}
print(json.dumps(res, indent=1))
(ROOT / "site/tests/parity_result.json").write_text(json.dumps(res, indent=1))
