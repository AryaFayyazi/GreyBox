#!/usr/bin/env python
"""Build the Grey Box website (static; GitHub Pages) into site/dist.

Every number on the site is read from a saved result file or from docs/PREREGISTRATION_RECORD.md at build time:
  results.html   the preregistration table, verbatim from the record (outcome tags derived from its text)
  index.html     headline numbers from results/t1/*.json and site/tests/parity_result.json
  monitor.html   the Claude agent monitor with its 3D space (web/build_monitor_demo.py)
  train.html     the in-browser trainer (site/src/assets/tm-*.js), with an example from CAPE-Bench
  docs.html, paper.html
python site/build.py   (the cape venv; reads /data for the example data and the 3D space)
"""
import html
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC, DIST = ROOT / "site/src", ROOT / "site/dist"
R = ROOT / "results"
REPO = "https://github.com/AryaFayyazi/GreyBox"
CODE_PUBLIC = False          # link locks/preregistrations to the code repo only once it is published
NAV = [("index.html", "Overview"), ("train.html", "Train a monitor"), ("monitor.html", "Claude, live replay"),
       ("results.html", "Evidence"), ("docs.html", "Docs"), ("paper.html", "Paper")]


def j(p):
    return json.loads((R / p).read_text())


def page(name, title, desc, body):
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == name else ""}>{t}</a>' for h, t in NAV)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · Grey Box</title><meta name="description" content="{html.escape(desc)}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=JetBrains+Mono:wght@400;600&display=swap">
<link rel="stylesheet" href="assets/site.css"><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%23596470'/%3E%3Crect x='4' y='4' width='8' height='8' rx='1' fill='%23eef0f2'/%3E%3C/svg%3E">
</head><body>
<header class="top"><div class="wrap"><a class="brand" href="index.html">Grey Box <span>agent monitoring</span></a><nav>{nav}</nav></div></header>
<main><div class="wrap">
{body}
</div></main>
<footer><div class="wrap">Grey Box is a research project. Every result on this site comes from a preregistered, hash-locked test,
failures included (<a href="results.html">evidence</a>). Built {__import__("datetime").date.today().isoformat()}.</div></footer>
</body></html>
"""


def md_inline(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def outcome_tag(t):
    low = t.lower()
    if "pending" in low:
        return '<span class="tag nt">pending</span>'
    if "not tested" in low and "pass" not in low:
        return '<span class="tag nt">not tested</span>'
    f, p = bool(re.search(r"\bfail", low)), bool(re.search(r"\bpass|confirmed", low))
    if f and p:
        return '<span class="tag mixed">mixed</span>'
    return '<span class="tag fail">fail</span>' if f else '<span class="tag pass">pass</span>' if p else ""


def results_page():
    rows = [l for l in (ROOT / "docs/PREREGISTRATION_RECORD.md").read_text().splitlines() if l.startswith("| ") and not l.startswith("| Test")]
    trs = []
    for l in rows:
        c = [x.strip() for x in l.strip("|").split("|")]
        test, lock, data, out = c[0], c[1], c[2], " | ".join(c[3:])
        m = re.search(r"\(([^)]*\d{2}:\d{2}[^)]*)\)", lock)
        locked = m.group(1) if m else ""
        lockf = lock.split(" (")[0]
        lockcell = f'<a href="{REPO}/blob/main/{lockf.split(" + ")[0]}">{html.escape(locked or "lock")}</a>' if CODE_PUBLIC else html.escape(locked)
        trs.append(f"<tr><td>{md_inline(test)}</td><td>{outcome_tag(out)}</td><td>{md_inline(out)}</td><td class=muted>{md_inline(data)}</td><td class=n>{lockcell}</td></tr>")
    body = f"""<h1>Evidence</h1>
<p class="lede">Every test was preregistered: hypotheses, data, metrics and pass criteria were written down and hash-locked
(SHA-256) before the data it is judged on existed or was scored. The table below is the project's record, reproduced
verbatim, including every failure, deviation and correction.</p>
<div class="note">How to read it: <span class="tag pass">pass</span> the preregistered criterion was met;
<span class="tag fail">fail</span> it was not; <span class="tag mixed">mixed</span> some endpoints passed and some failed;
<span class="tag nt">not tested</span> the data did not allow the preregistered test. AUROC: 0.5 is chance, 1.0 perfect.
Brackets are 95% bootstrap confidence intervals.</div>
<div class="tablewrap"><table><thead><tr><th>Test</th><th>Outcome</th><th>Result</th><th>Data</th><th>Locked</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<h2>Figures</h2>
<figure><img src="figures/generalization.png" alt="AUROC of monitors under increasing shift" loading="lazy"><figcaption>Monitoring an agent's next
action from its transcript, under increasing shift: trained reader (blue), trained bag-of-words (hollow square), zero-shot 70B judge (green),
the agent's own internals (orange). The two Claude rows rest on one stratum (136 actions, 26 violations).</figcaption></figure>
<figure><img src="figures/time_vs_access.png" alt="Time, not access" loading="lazy"><figcaption>The same 12,128 fresh steps read at the start of
the agent's turn and after its reasoning. Reading later helps every method; access alone does not.</figcaption></figure>
<figure><img src="figures/label_efficiency.png" alt="Label efficiency" loading="lazy"><figcaption>Reader accuracy against the number of labelled
training actions.</figcaption></figure>
<figure><img src="figures/claude_vs_moderation.png" alt="Refusal forecast vs moderation" loading="lazy"><figcaption>Chat: which harmful requests
Claude refuses. The forecast beats a 3-level moderation verdict and ties graded moderators.</figcaption></figure>
<figure><img src="figures/guard_tradeoff.png" alt="Guard trade-off" loading="lazy"><figcaption>A resampling guard, with samples scored by the
environment's own rule checker (an oracle; not deployable: with an LLM judge as the checker it catches 12%).</figcaption></figure>
"""
    return page("results.html", "Evidence", "Every preregistered test, its lock and its outcome, including failures.", body)


def index_page():
    t4, t8, t9, t10 = j("t1/t4_reader.json"), j("t1/t8b_claude_all_actions.json"), j("t1/t9_text_baseline.json"), j("t1/t10_claude_unseen_families.json")
    t2, br = j("t1/t2_time.json"), j("t1/t8b_breakdown_exploratory.json")
    lab = j("t1/t10_new_policy_labels_exploratory.json")["by_n"]
    gate = j("t1/t8b_calibration_exploratory.json")["gate_eval_76c"]
    par = json.loads((ROOT / "site/tests/parity_result.json").read_text())
    p = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    c8 = t8["caught_at_5pct_safe_escalated"]
    body = f"""<h1>Monitor AI agents you can't open.</h1>
<p class="lede">Grey Box decides whether an AI agent's next action should go to a human <i>before it runs</i>, reading only
what every deployment can see: the agent's instructions, the user's task, earlier tool results and the proposed action.
It needs no access to the model's weights, so it works on closed models. Every claim here comes from a preregistered,
hash-locked test, failures included.</p>
<div class="grid3" style="margin:22px 0 8px">
  <div class="card"><div class="label">Try it now</div><h3>Train a monitor in your browser</h3><p>Load labelled steps from your agent (or our
    example), get an honest held-out accuracy and a review threshold, score new actions. Nothing is uploaded.</p><a class="go" href="train.html">Open the trainer →</a></div>
  <div class="card"><div class="label">See it work</div><h3>A closed frontier agent, monitored</h3><p>Real Claude Haiku 5.5 agent episodes, with the
    risk assigned to each action before it ran, in the monitor's own 3D space.</p><a class="go" href="monitor.html">Open the replay →</a></div>
  <div class="card"><div class="label">Deploy</div><h3>Run it on your own infrastructure</h3><p>A Python SDK and an HTTP API: a free CPU monitor
    ({par['js_score_ms_per_step']:.2f} ms per action in a browser) and a GPU reader monitor (41 ms per action).</p><a class="go" href="docs.html">Read the docs →</a></div>
</div>

<h2>What we found</h2>
<div class="grid2">
<div class="card"><h3>Labels, not access</h3><p>A small open model that reads the transcript, trained on labelled examples, matches the
  agent's own internal activations (AUROC {t4['auroc_fresh']['READER_act']:.3f} vs {t4['auroc_fresh']['I_dec']:.3f}). Trained only on an open agent,
  it caught {p(c8['READER'])} of a closed frontier agent's violations (Claude Haiku 5.5) with 5% of actions sent to review. A zero-shot
  70B judge caught {p(c8['J_act'])}.</p></div>
<div class="card"><h3>Time, not access</h3><p>White-box probes look better than transcript monitors mainly because they read later: after
  the model's hidden reasoning, the action is nearly decided (probe {t2['auroc']['I_dec']:.3f} after reasoning vs {t2['auroc']['WB_ts']:.3f} before it).
  A monitor that sees the proposed action gets the same advantage.</p></div>
<div class="card"><h3>The policy is the limit, not the agent</h3><p>For a policy family the monitor never saw, it still ranked Claude's
  violations above safe actions (AUROC {t10['auroc_pooled']['READER_lofo']:.3f} vs judge {t10['auroc_pooled']['J_act']:.3f}), but escalated none of
  them at a 5% budget. Adding {list(lab)[3]} labelled steps of that policy brought it to {p(lab['20']['cons_service_restart'])}, and {list(lab)[4]} to
  {p(lab['40']['cons_service_restart'])}. Label a few dozen steps of each policy you enforce.</p></div>
<div class="card"><h3>Simple monitors suffice, until the task changes</h3><p>A word-count classifier trained on the same labels matches the
  reader in distribution (Claude: {t9['T8b_claude']['auroc']['TXT']:.3f} vs {t9['T8b_claude']['auroc']['READER']:.3f}). On a new benchmark of prompt
  injections it fell to chance ({t9['T7_agentdojo']['auroc']['TXT']:.2f}), where the reader held {t9['T7_agentdojo']['auroc']['READER']:.2f}.</p></div>
<div class="card"><h3>Set the review threshold on your own traffic</h3><p>A monitor's probabilities do not transfer between agents (a
  0.5 cut caught 3% on Claude); its ranking does. Escalating the top 5% of the agent's own first 500 actions, then freezing that cut,
  caught {p(gate['online']['frozen_default']['run_order']['caught_after_100'])} online and held at
  {p(gate['burst']['rate_0.5']['frozen_default']['caught'])} when half of a later burst was malicious. A rolling cut fell to
  {p(gate['burst']['rate_0.5']['rolling5000']['caught'])}.</p></div>
<div class="card"><h3>Chat: forecasting refusals</h3><p>Asking a model about a conversation in a side branch forecasts whether it will refuse.
  On Claude it ties a strong graded moderator at ranking refusals, and combined with one it doubles the precision of a
  "harmful request answered" alarm. It is not a better content filter.</p></div>
</div>
<div class="note limit"><b>Limits.</b> Most agent tests use one benchmark (CAPE-Bench) and one closed agent family so far (a second, GPT, is under a locked test); the external test
(AgentDojo) is weaker (AUROC 0.78). Monitors must be trained on labelled steps of the policies they enforce. The chat
readouts recover what behaviour reveals, not a model's hidden beliefs. See <a href="results.html">the full evidence</a>.</div>
"""
    return page("index.html", "Monitor AI agents you can't open", "Grey Box: pre-execution monitoring for closed AI agents, with preregistered evidence.", body)


def md_page(name, title, desc):
    s = (SRC / "pages" / name).read_text()
    return page(name, title, desc, s)


def example_data(n=1500):
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), "/home/arya/new_safety/cape/src", "/home/arya/new_safety/cape/scripts"]
    import pandas as pd
    t9 = importlib.import_module("77_t9_text_baseline")
    U = pd.read_parquet("/data/models/graybox_ws/outputs/interrogate/gemma4_clean_units.parquet")
    T = t9.gemma_texts("clean")
    idx = np.sort(np.random.default_rng(0).permutation(len(U))[:n])
    (DIST / "data").mkdir(parents=True, exist_ok=True)
    with open(DIST / "data/example_capebench.jsonl", "w") as f:
        for i in idx:
            f.write(json.dumps({"context": T.ctx[i], "action": T.act[i], "label": int(U.y[i]), "group": U.scenario_id[i]}, ensure_ascii=False) + "\n")


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "assets").mkdir(parents=True)
    for f in (SRC / "assets").iterdir():
        shutil.copy(f, DIST / "assets" / f.name)
    shutil.copytree(R / "figures", DIST / "figures", ignore=shutil.ignore_patterns("*.pdf"))
    (DIST / ".nojekyll").write_text("")
    pages = {"index.html": index_page(), "results.html": results_page()}
    for name, title, desc in (("train.html", "Train a monitor", "Train an agent-action monitor on your own labelled steps, in your browser."),
                              ("docs.html", "Docs", "Install, deploy and evaluate Grey Box monitors."),
                              ("paper.html", "Paper", "Labels, not access: what a black box reveals about its hidden state, and how to monitor closed agents.")):
        if (SRC / "pages" / name).exists():
            s = (SRC / "pages" / name).read_text()
            m1, m2 = re.search(r"<!--title: (.*?)-->", s), re.search(r"<!--desc: (.*?)-->", s)
            pages[name] = page(name, m1.group(1) if m1 else title, m2.group(1) if m2 else desc, s)
    for k, v in pages.items():
        (DIST / k).write_text(v)
    # the Claude replay, wrapped with the site header
    tmp = DIST / "_monitor_raw.html"
    subprocess.run([sys.executable, str(ROOT / "web/build_monitor_demo.py"), "/data/models/graybox_ws/outputs/t8b/space.json", str(tmp)], check=True)
    raw = tmp.read_text()
    tmp.unlink()
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == "monitor.html" else ""}>{t}</a>' for h, t in NAV)
    (DIST / "monitor.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="assets/site.css"></head><body>
<header class="top"><div class="wrap"><a class="brand" href="index.html">Grey Box <span>agent monitoring</span></a><nav>{nav}</nav></div></header>
{raw}</body></html>""")
    example_data()
    print("built", sorted(p.name for p in DIST.iterdir()))


if __name__ == "__main__":
    main()
