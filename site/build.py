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
CODE_PUBLIC = True           # link locks/preregistrations to the code repo only once it is published
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
<figure><img src="figures/policy_labels.png" alt="Policy coverage" loading="lazy"><figcaption>A policy family removed from training. A credentials
policy with similar trained policies transferred with no labels (GPT, T12). A service-restart policy without them was missed until 20–40
labelled steps were added (Claude, T10, exploratory).</figcaption></figure>
<figure><img src="figures/claude_vs_moderation.png" alt="Refusal forecast vs moderation" loading="lazy"><figcaption>Chat: which harmful requests
Claude refuses. The forecast beats a 3-level moderation verdict and ties graded moderators.</figcaption></figure>
<figure><img src="figures/guard_tradeoff.png" alt="Guard trade-off" loading="lazy"><figcaption>A resampling guard, with samples scored by the
environment's own rule checker (an oracle; not deployable: with an LLM judge as the checker it catches 12%).</figcaption></figure>
"""
    return page("results.html", "Evidence", "Every preregistered test, its lock and its outcome, including failures.", body)


def index_page():
    t4, t8, t9, t10 = j("t1/t4_reader.json"), j("t1/t8b_claude_all_actions.json"), j("t1/t9_text_baseline.json"), j("t1/t10_claude_unseen_families.json")
    t2, br = j("t1/t2_time.json"), j("t1/t8b_breakdown_exploratory.json")
    t11 = j("t1/t11_gpt_agent.json")
    lab = j("t1/t10_new_policy_labels_exploratory.json")["by_n"]
    gate = j("t1/t8b_calibration_exploratory.json")["gate_eval_76c"]
    par = json.loads((ROOT / "site/tests/parity_result.json").read_text())
    p = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    c8 = t8["caught_at_5pct_safe_escalated"]
    body = f"""<h1>Monitor AI agents that you cannot open.</h1>
<p class="lede">Grey Box decides if an AI agent's next action must go to a human <i>before the action runs</i>. It reads only
the data that every deployment can see: the agent's instructions, the user's task, the earlier tool results and the
proposed action. It does not need the model's weights, so it works on closed models. Each claim on this site comes from a
preregistered, hash-locked test. We show the failures too.</p>
<div class="grid3" style="margin:22px 0 8px">
  <div class="card"><div class="label">Try it now</div><h3>Train a monitor in your browser</h3><p>Load labelled steps from your agent, or
    use our example. Get a held-out accuracy and a review threshold. Score new actions. Your data stays in your browser.</p><a class="go" href="train.html">Open the trainer →</a></div>
  <div class="card"><div class="label">See it work</div><h3>A closed frontier agent, monitored</h3><p>Real episodes of Claude Haiku 5.5 as an
    agent. See the risk that the monitor gave to each action before the action ran, in the monitor's own 3D space.</p><a class="go" href="monitor.html">Open the replay →</a></div>
  <div class="card"><div class="label">Deploy</div><h3>Run it on your own machines</h3><p>A Python SDK and an HTTP API. The CPU monitor
    takes {par['js_score_ms_per_step']:.2f} ms for each action in a browser. The GPU reader monitor takes 41 ms.</p><a class="go" href="docs.html">Read the docs →</a></div>
</div>

<h2>What we found</h2>
<div class="grid2">
<div class="card"><h3>Labels, not access</h3><p>A small open model reads the transcript. We train it on labelled examples. It is
  equal to or better than the agent's own internal activations (AUROC {t4['auroc_fresh']['READER_act']:.3f} vs {t4['auroc_fresh']['I_dec']:.3f}).
  We trained it only on an open agent. On two closed agents, on its training benchmark, with 5% of the actions sent to review, it caught {p(c8['READER'])}
  (Claude Haiku 5.5) and {p(t11['label_free_cut']['READER']['all'])} (GPT-6-luna) of the violations. A zero-shot 70B judge caught
  {p(c8['J_act'])} and {p(t11['label_free_cut']['J_act']['all'])}.</p></div>
<div class="card"><h3>Why access does not help: a theorem</h3><p>An agent's internal state is a function of the tokens that it read.
  For the label "does this action break the policy?", a monitor that reads the transcript and the proposed action is
  Bayes-sufficient. Internals read at or before the action cannot do better. White-box probes win against turn-start
  monitors only because they read later (probe {t2['auroc']['I_dec']:.3f} after the reasoning, {t2['auroc']['WB_ts']:.3f} before it).</p></div>
<div class="card"><h3>The limit is the policy, not the agent</h3><p>We removed a policy family from the training data. A credentials
  policy with similar trained policies still transferred: the monitor caught all of GPT's violations of it with no labels. A
  service-restart policy without similar ones did not: the monitor ranked Claude's violations well (AUROC
  {t10['auroc_pooled']['READER_lofo']:.3f}) but escalated none at a 5% budget. With 20 labelled steps it caught
  {p(lab['20']['cons_service_restart'])}; with 40, {p(lab['40']['cons_service_restart'])}. Labels from open agents work for closed
  ones: on prompt injections, labelled calls of open agents raised the catch rate on a closed agent from 36% to 92%. Label steps of each
  new type of policy.</p></div>
<div class="card"><h3>Simple monitors are enough, until the task changes</h3><p>A word-count classifier with the same labels is equal to
  the reader on the training benchmark (Claude: {t9['T8b_claude']['auroc']['TXT']:.3f} vs {t9['T8b_claude']['auroc']['READER']:.3f}).
  On a new benchmark of prompt injections it fell to chance ({t9['T7_agentdojo']['auroc']['TXT']:.2f}). The reader stayed at
  {t9['T7_agentdojo']['auroc']['READER']:.2f}.</p></div>
<div class="card"><h3>Set the review threshold from your own traffic</h3><p>A monitor's probabilities do not transfer between agents: a
  0.5 cut caught 3% on Claude. Its ranking transfers. We escalate the top 5% of the agent's first 500 actions and then freeze that
  threshold. It caught {p(gate['online']['frozen_default']['run_order']['caught_after_100'])} online. When half of a later burst was
  malicious, it still caught {p(gate['burst']['rate_0.5']['frozen_default']['caught'])}. A rolling threshold caught
  {p(gate['burst']['rate_0.5']['rolling5000']['caught'])}.</p></div>
<div class="card"><h3>Chat: forecast a refusal</h3><p>We ask a model about a conversation in a side branch. The answers forecast if the
  model will refuse. On Claude this forecast only ties a strong graded moderator. Together with a moderator, it doubles the
  precision of an alarm for harmful requests that the model answers. It is not a better content filter.</p></div>
</div>
<div class="note limit"><b>Limits.</b> Most agent tests use one benchmark (CAPE-Bench). Two closed agent families are tested (Claude, GPT).
On an external benchmark of prompt injections, the reader without labels of that policy was not significantly better than a zero-shot judge on
a closed agent (AUROC 0.840 vs 0.787); with labelled calls from open agents it reached 0.922. A monitor needs labelled steps of each type of
policy that it enforces. The chat readouts recover a model's behaviour, not its hidden beliefs. See <a href="results.html">all the evidence</a>.</div>
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
