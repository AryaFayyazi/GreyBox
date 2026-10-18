#!/usr/bin/env python
"""After T19: ship the selected browser reader if the preregistered rule allows it (docs/prereg_T19_small_readers.md).
  F1 and F2 pass -> the "transfer" monitor of the browser tool; F1 passes, F2 fails -> offered for new kinds of tasks
  only; F1 fails -> nothing is shipped (the site says that no tested browser model transfers).
Writes results/tool/browser_reader.json: the pre-trained probe (graybox.embed_monitor format, trained on the T19
training steps only) and the numbers the page shows. Deployment check (reported): the browser computes float32 CPU
features, T19 used bf16 GPU features; the exported probe is scored on float32 CPU features of T13b and T7.
python scripts/105_export_browser_reader.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.embed import Encoder, fit_probe  # noqa: E402
from graybox.embed_monitor import to_json  # noqa: E402

DOWNLOAD = {"minilm": "90 MB", "mpnet": "440 MB", "qemb06": "1.2 GB"}   # fp16 ONNX: features match PyTorch (cosine 1.000); q8 only 0.855


def main():
    m = importlib.import_module("100_t19_small_readers")
    r = json.loads((ROOT / "results/t1/t19_small_readers.json").read_text())
    out = ROOT / "results/tool/browser_reader.json"
    if not r["F1_pass"]:
        out.write_text(json.dumps({"shipped": False, "reason": "T19 F1 failed: no tested browser reader transfers better than TF-IDF"}, indent=1))
        print("not shipped (F1 failed)")
        return
    enc = r["selected_browser_reader"]
    tr = m.data("train")
    probe = fit_probe(m.features(enc, "train"), tr.y.values)
    E = Encoder(enc, "cpu", m.HF, batch=16)
    check = {}
    for s in ("T13b", "T7"):
        d = m.data(s)
        z = probe.decision_function(E.encode(list(d.ctx), list(d.act)))
        check[s] = {"auroc_float32_cpu": round(float(m.auroc(d, z)), 4), "auroc_t19_bf16_gpu": r["auroc"][enc][s]}
        print(s, check[s], flush=True)
    A, c = r["auroc"], r["caught_label_free_5pct"]
    scope = "all tasks" if r["F2_pass"] else "new kinds of tasks only"
    note = (f"Browser reader ({enc}, {DOWNLOAD[enc]} once, then cached). Trained on our CAPE-Bench steps only. On a new kind of task "
            f"(GPT-6-luna on AgentDojo) it scores AUROC {A[enc]['T13b']:.3f}, against {A['TXT']['T13b']:.3f} for word counts and "
            f"{A['q8']['T13b']:.3f} for the GPU reader (preregistered test T19). On the closed agents' CAPE-Bench actions: "
            f"{A[enc]['T8b']:.3f} (Claude) and {A[enc]['T11']:.3f} (GPT). Use: {scope}.")
    d = to_json(enc, probe, threshold_logit=float("nan"), trained_on="T19 training steps (Gemma-4, CAPE-Bench, 17,623)")
    d.pop("threshold_logit")                       # the page sets the cut from the user's own actions (label-free)
    res = {"shipped": True, "encoder": enc, "web": d["web"], "download": DOWNLOAD[enc], "scope": scope, "note": note,
           "t19": {"auroc": {k: A[k] for k in (enc, "TXT", "q8")}, "caught_label_free_5pct": {k: c[k] for k in (enc, "TXT", "q8")},
                   "F1": r["F1_T13b_selected_minus_TXT"], "F2": r["F2_closedCAPE_selected_minus_q8"]},
           "deployment_check": check, "pretrained": d}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print("shipped", enc, scope, out)


if __name__ == "__main__":
    main()
