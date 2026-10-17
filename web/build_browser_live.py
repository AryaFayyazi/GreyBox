#!/usr/bin/env python
"""Build the in-browser Grey Box Live page (web/live_browser.html) with its data, for the TRANSCRIPT framing (L3d):
  the frozen readouts (models/live_tx/readouts_v4.npz: frame, cloud, will_refuse and harmful_latest),
  Claude's calibrated thresholds (95th percentile over its 44 single-turn calibration conversations, as scripts/57),
  the validation lines (results/l3/claude_transcript.json), the moderator prompt (scripts/50), and the recorded
  9-turn Claude session (scripts/58) as the fallback when the page cannot call Claude.
The page runs the same estimator as scripts/57: q = (yes + 0.5)/(yes + no + 1), θ = 2 asin sqrt q, logistic readouts.

python web/build_browser_live.py <out.html>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from graybox.batch import BATCH_PREAMBLE, BATCH_SUFFIX, TRANSCRIPT_HEAD, TRANSCRIPT_TAIL  # noqa: E402
from graybox.battery_v2m import BATTERY_V2M  # noqa: E402

CB = Path("/data/models/graybox_ws/outputs/livebench_claude_batch")        # L3c: replies, moderator, refusal labels
CT = Path("/data/models/graybox_ws/outputs/livebench_claude_transcript")   # L3d: transcript-framing answers
BUNDLE = Path("/data/models/graybox_ws/models/live_tx/readouts_v4.npz")


def theta_of(counts):
    c = np.asarray(counts, float)
    q = (c[..., 0] + 0.5) / (c[..., 0] + c[..., 1] + 1.0)
    return q, 2 * np.arcsin(np.sqrt(q))


def main(out):
    R = np.load(BUNDLE, allow_pickle=True)
    items = list(R["items"])
    assert items == [k for k, _ in BATTERY_V2M], "the one-call bundle must use all 88 items in battery order"
    con = {c["name"]: c for c in R["concepts"]}
    cal = pd.read_parquet(CT / "calib.parquet")
    _, th_cal = theta_of(np.array(cal.counts.map(lambda x: [list(v) for v in x]).tolist()))
    readouts = {}
    for name in ("will_refuse", "harmful_latest"):
        w, b = np.asarray(con[name]["w"], float), float(con[name]["b"])
        thr = float(np.quantile(1 / (1 + np.exp(-(th_cal @ w + b))), 0.95))
        readouts[name] = {"w": w.round(6).tolist(), "b": b, "threshold": round(thr, 4), "calibrated_95pct_benign": round(thr, 4)}
    readouts["will_refuse"]["threshold"] = 0.5      # the page's refusal forecast and alarm use 50% (see the alarm rule)
    mu, dirs, scale = np.asarray(R["mu"], float), np.asarray(R["dirs"], float), float(R["scale"])
    res = json.loads((ROOT / "results/l3/claude_transcript.json").read_text())
    f1, f2 = res["F1_harmful_refusal"], res["F2_claude_moderator"]
    validation = {
        "will_refuse": f"Readout fit only on open models. On {res['n_harmful']} held-out harmful requests to Claude (preregistered), it "
                       f"ranked which ones Claude refused at AUROC {f1['greybox_auroc']:.2f}, against {f1['moderation_auroc']:.2f} for a moderator's "
                       f"Safe/Controversial/Unsafe verdict. Against a graded moderator score, it is about as good, not better "
                       f"(prospective test: 0.87 vs 0.85).",
        "harmful_latest": "A text moderator reading the message is at least as good at this (open models: 0.97 vs 0.85–0.93); "
                          "shown for context.",
        "alarm": ("Tested prospectively on Claude (preregistered, 923 fresh conversations incl. real jailbreaks): 54% of alarms "
                  "were real cases of Claude going along with a harmful request, against 27% for the moderator alone, and the "
                  "alarm fired on 4% of benign messages. A strong graded moderator combined the same way does nearly as well "
                  "(49%): the grey box adds a small, real increment, not a replacement for moderation.")}
    rec = []
    if (CT / "live.parquet").exists():
        L = pd.read_parquet(CT / "live.parquet").sort_values("id", key=lambda s: s.str.split("_").str[1].astype(int))
        rf = pd.read_parquet(CB / "live_refusals.parquet").set_index("id").refused if (CB / "live_refusals.parquet").exists() else None
        import importlib
        script = importlib.import_module("27_live_record").SCRIPT
        for _, row in L.iterrows():
            c = np.array([list(v) for v in row.counts], float)
            q, th = theta_of(c)
            r = {}
            for name, ro in readouts.items():
                p = 1 / (1 + np.exp(-(th @ np.asarray(ro["w"]) + ro["b"])))
                r[name] = {"score": round(float(p), 4), "flag": bool(p >= ro["threshold"])}
            i = int(row.id.split("_")[1])
            rec.append({"user": script[i], "reply": row.reply, "mod": row.claude_mod, "q": q.round(4).tolist(),
                        "point": (((th - mu) @ dirs) * scale).round(4).tolist(), "r": r,
                        "answered": round(float(c[:, :2].sum() / c.sum()), 4),
                        "refused": (None if rf is None or rf.get(row.id, -1) < 0 else bool(rf[row.id]))})
    facts = (f"<div class='label'>What this is, and how far to trust it</div>"
             f"<p><b>Method.</b> A separate call reads the conversation as a transcript and answers the questions; the reply you see "
             f"never sees them. The two answer sets "
             f"give a probability per question, which places Claude in a space built from six open models. Readouts trained "
             f"only on those open models then forecast what Claude is about to do.</p>"
             f"<p><b>Tested on Claude (preregistered).</b> {validation['will_refuse']} {validation['alarm']}</p>"
             f"<p><b>How to read an alarm.</b> In a prospective test on Claude, about half of the alarms were real cases of "
             f"Claude going along with a harmful request, and the alarm caught about half of such cases. Treat an alarm as 'look closer', not as proof. "
             f"This cheap format (2 calls) ranks requests well but its probabilities are compressed on Claude: in the "
             f"recorded session it gave the poison request only a 45% chance of refusal, and Claude refused.</p>"
             f"<p><b>Limits.</b> A refusal shows up once the reply arrives, so the forecast matters most where you must act first "
             f"(agents, audits). Thresholds were calibrated through the API on 44 benign messages, and this page's own framing "
             f"may shift answers slightly. Each message uses 4 Claude calls from your account.</p>")
    import importlib as _il
    mod_prompt = _il.import_module("50_claude_batch").MOD_PROMPT
    G = {"model": "claude-haiku-5-5", "n_samples": 2, "questions": [q for _, q in BATTERY_V2M],
         "preamble": BATCH_PREAMBLE, "suffix": BATCH_SUFFIX, "tx_head": TRANSCRIPT_HEAD, "tx_tail": TRANSCRIPT_TAIL,
         "mod_prompt": mod_prompt, "mu": mu.round(6).tolist(), "dirs": dirs.round(6).tolist(),
         "scale": scale, "cloud": np.asarray(R["cloud"]).round(4).tolist(), "cloud_label": [str(x) for x in R["cloud_label"]],
         "readouts": readouts, "validation": validation, "facts": facts, "recorded": rec}
    tpl = (ROOT / "web/live_browser.html").read_text()
    blob = json.dumps(G, separators=(",", ":")).replace("</", "<\\/")
    Path(out).write_text(tpl.replace("<!--DATA-->", f"<script>window.GB={blob};</script>"))
    print("wrote", out, len(blob) // 1024, "KB;", len(rec), "recorded turns; thresholds",
          {k: v["threshold"] for k, v in readouts.items()})


if __name__ == "__main__":
    main(sys.argv[1])
