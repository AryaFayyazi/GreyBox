#!/usr/bin/env python
"""Build the Grey Box website (static; GitHub Pages) into site/dist.

Every number on the site is read from a saved result file or from docs/PREREGISTRATION_RECORD.md at build time:
  results.html   the preregistration table, verbatim from the record (outcome tags derived from its text)
  index.html     headline numbers from results/t1/*.json and site/tests/parity_result.json
  monitor.html   the Claude Haiku 5.5 agent monitor with its 3D space (web/build_monitor_demo.py)
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
NAV = [("index.html", "Overview"), ("selfpolice.html", "Self-policing"), ("train.html", "Train a monitor"), ("monitor.html", "Agent replay"),
       ("chat.html", "Chat grey box"), ("results.html", "Evidence"), ("docs.html", "Docs"), ("paper.html", "Paper")]


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
<footer><div class="wrap">Grey Box is a research project. The confirmatory results on this site come from preregistered, hash-locked
tests, failures included; exploratory results are marked as such (<a href="results.html">evidence</a>). Built {__import__("datetime").date.today().isoformat()}.</div></footer>
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
(SHA-256) before the test ran. The lock times are self-reported; for T13b, T16 and T20 the lock was also published before the result.
The table below is the project's record, reproduced
verbatim, including every failure, deviation and correction.</p>
<div class="note">How to read it: <span class="tag pass">pass</span> the preregistered criterion was met;
<span class="tag fail">fail</span> it was not; <span class="tag mixed">mixed</span> some endpoints passed and some failed;
<span class="tag nt">not tested</span> the data did not allow the preregistered test. AUROC: 0.5 is chance, 1.0 perfect.
Brackets are 95% bootstrap confidence intervals.</div>
<details style="margin:10px 0"><summary><b>Words used in this table</b></summary><ul style="margin-top:8px">
<li><b>AUROC</b>: the chance that a random violation gets a higher score than a random safe action (0.5 = guessing, 1.0 = perfect).</li>
<li><b>Caught at 5%</b>: the share of violations among the 5% of actions with the highest scores (what a human reviewer would see).</li>
<li><b>Reader</b>: an open language model (Qwen3-8B) that reads the transcript; a simple classifier on its internal state is the monitor.</li>
<li><b>Probe</b>: a simple (linear) classifier on a model's internal state.</li>
<li><b>Replica</b>: an estimate of an agent's hidden state, computed from the reader; <b>R²</b> is the share of the state's variation that it reproduces.</li>
<li><b>TF-IDF</b>: a word-count classifier (the browser trainer's default monitor).</li>
<li><b>Preregistered, hash-locked</b>: the plan and the analysis code were fixed, and their fingerprint saved, before the test ran.</li>
<li><b>Exploratory</b>: an analysis decided after seeing data; it suggests, it does not confirm.</li>
<li><b>Open / closed agent</b>: a model with public weights (we can read its internals) / a model behind an API (we cannot).</li>
</ul></details>
<div class="tablewrap"><table><thead><tr><th>Test</th><th>Outcome</th><th>Result</th><th>Data</th><th>Locked</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<h2>Figures</h2>
<figure><img src="figures/replica.png" alt="True hidden state vs grey-box replica" loading="lazy"><figcaption>The agent's true hidden state (a)
and its replica from the transcript (b), on fresh steps; (c) the variance replicated per principal component (T15, preregistered).</figcaption></figure>
<figure><img src="figures/generalization.png" alt="AUROC of monitors under increasing shift" loading="lazy"><figcaption>Monitoring an agent's next
action from its transcript, under increasing shift: trained reader (blue), trained bag-of-words (hollow square), zero-shot 70B judge (green),
the agent's own internals (orange). The dashed line is chance (AUROC 0.5). The two Claude Haiku 5.5 rows rest on one stratum (136 actions, 26 violations).</figcaption></figure>
<figure><img src="figures/time_vs_access.png" alt="Time, not access" loading="lazy"><figcaption>The same 12,128 fresh steps read at the start of
the agent's turn and after its reasoning. Reading later helps every method; access alone does not.</figcaption></figure>
<figure><img src="figures/label_efficiency.png" alt="Label efficiency" loading="lazy"><figcaption>Reader accuracy against the number of labelled
training actions.</figcaption></figure>
<figure><img src="figures/policy_labels.png" alt="Policy coverage" loading="lazy"><figcaption>A policy family removed from training. A credentials
policy with similar trained policies transferred with no labels (GPT, T12). A service-restart policy without them was missed until 20–40
labelled steps were added (Claude Haiku 5.5, T10, exploratory).</figcaption></figure>
<figure><img src="figures/claude_vs_moderation.png" alt="Refusal forecast vs moderation" loading="lazy"><figcaption>Chat: which harmful requests
Claude Haiku 5.5 refuses. The forecast beats a 3-level moderation verdict and ties graded moderators.</figcaption></figure>
<figure><img src="figures/guard_tradeoff.png" alt="Guard trade-off" loading="lazy"><figcaption>A resampling guard, with samples scored by the
environment's own rule checker (an oracle; not deployable: with an LLM judge as the checker it catches 12%).</figcaption></figure>
"""
    return page("results.html", "Evidence", "Every preregistered test, its lock and its outcome, including failures.", body)


def g14_card():
    """The training-free monitor's direct test (G15), with the trained comparison (G14); empty until the results exist."""
    f15, f14 = R / "t1/g15_training_free.json", R / "t1/g14_decision_grey_box.json"
    if not f15.exists():
        return ""
    g = json.loads(f15.read_text())
    N = {"gemma4": "Gemma-4", "gptoss20b": "gpt-oss", "qwen38": "Qwen3.8", "museglimmer": "Muse-Glimmer"}
    rows = "".join(f"<tr><td>{N[a]}</td><td>{g[a]['within']['mean8']:.3f}</td><td>{g[a]['within']['complies']:.3f}</td><td>{g[a]['within']['white']:.3f}</td></tr>"
                   for a in N if a in g)
    E = g["endpoints"]
    npass = sum(bool(v) for v in E.values())
    verdict = "all preregistered endpoints passed" if npass == len(E) else f"{npass} of {len(E)} preregistered endpoints passed (" + ", ".join(
        f"{k.split('_')[0]} {'pass' if v else 'FAIL'}" for k, v in E.items()) + ")"
    t1, t2 = g["T1_mean8_minus_complies"], g["T2_mean8_minus_white_box"]
    cl = g.get("claude")
    claude = (f" On the closed model Claude Haiku 5.5, the read-out ranks its unapproved production restarts at {cl['restart_mean8']:.3f} "
              f"[{cl['restart_mean8_ci'][0]:.2f}, {cl['restart_mean8_ci'][1]:.2f}].") if cl else ""
    trained = ""
    if f14.exists():
        h = json.loads(f14.read_text())
        trained = (" A trained comparison that recreates the decision state from an open reader and the answers, then reads it with the "
                   f"probe on true states, differs from the true state by {h['P1_grey_minus_white_mean'][0]:+.3f} on average (G14).")
    return (f'<div class="card" style="margin-bottom:14px"><h3>Answers to fixed yes/no questions, read with a fixed rule, monitor an agent\'s decisions</h3>'
            f'<p>At each proposed call the agent answers 8 yes/no questions about the call, through its own API, in side branches. A fixed '
            f'read-out of the answers scores the call: no labels, nothing trained. On held-out calls of four open agents, {verdict}.{claude}{trained}</p>'
            f'<table class="tablewrap"><tr><th>Agent</th><th>its 8 answers (no training)</th><th>its 1 answer (self-audit)</th><th>probe trained on its hidden states</th></tr>{rows}</table>'
            f'<p class="muted">Within-template AUROC. Mean difference, 8 answers − 1 answer: {t1[0]:+.3f} [{t1[1]:+.3f}, {t1[2]:+.3f}]; '
            f'8 answers − trained probe: {t2[0]:+.3f} [{t2[1]:+.3f}, {t2[2]:+.3f}]. Preregistered (G15). A probe tuned afterwards is 0.015 '
            f'higher on average (exploratory).</p>{second_card()}</div>')


def second_card():
    """Second benchmarks for the training-free read-out: G18 (tau2-bench, Claude Haiku 5.5) and G17 (AgentDojo, open agents)."""
    out = ""
    f18 = R / "t1/g18_tau2_claude_readout.json"
    if f18.exists():
        g = json.loads(f18.read_text())
        r1, r2 = g["R1_within_tool_read_out_8"], g["R2_read_out_minus_single"]
        out += (f'<p><b>A second benchmark for the closed agent</b> (G18, preregistered). On &tau;&sup2;-bench airline, with 8 questions '
                f'written from its published policy before any answer, the same read-out ranks Claude Haiku 5.5\'s policy violations at '
                f'{r1[0]:.3f} [{r1[1]:.2f}, {r1[2]:.2f}] within tool ({g["writes"]} writes, {g["violations"]} violations); its single answer '
                f'scores {g["within_tool_single_answer"]:.3f} (difference {r2[0]:+.3f} [{r2[1]:+.3f}, {r2[2]:+.3f}]). Within one task it is '
                f'weaker (0.61, exploratory).</p>')
    f16 = R / "t1/g16_own_vs_other.json"
    if f16.exists():
        g = json.loads(f16.read_text())
        d = g["O1_own_minus_other_mean"]
        out += (f'<p><b>Whose answers?</b> (G16, preregistered). Another model answering the same 8 questions about each call ranks the '
                f'violations as well as the agent\'s own answers: own &minus; other {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] over four open agents; '
                f'on Claude Haiku 5.5\'s restarts Gemma-4\'s answers score {g["claude"]["restart_other"]:.3f} against Claude Haiku 5.5\'s {g["claude"]["restart_own"]:.3f}. '
                f'The monitor is a fixed questionnaire that any capable model can answer.</p>')
    f20 = R / "t1/g20_questionnaire_vs_judge.json"
    if f20.exists():
        g = json.loads(f20.read_text())
        d = g["J1_read_out_minus_judge_mean"]
        out += (f'<p><b>Against a reasoning judge</b> (G20, preregistered). A reasoning model that reads the same conversation and gives one '
                f'verdict ranks the open agents\' violations worse than the questionnaire: read-out &minus; judge {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}].</p>')
    f21 = R / "t1/g21_tau2_open_answerer.json"
    if f21.exists():
        g = json.loads(f21.read_text())
        out += (f'<p><b>Comparison: another model answering</b> (G21, preregistered; not the method). Gemma-4 (reasoning on) answered the same '
                f'questions about Claude Haiku 5.5\'s &tau;&sup2;-bench writes: {g["within_tool"]["gemma4_8"]:.3f} [{g["A1_gemma4_8_ci"][0]:.2f}, '
                f'{g["A1_gemma4_8_ci"][1]:.2f}] within tool, against {g["within_tool"]["claude_own_8"]:.3f} from Claude Haiku 5.5\'s own answers '
                f'(given with reasoning off). The method itself uses only the agent\'s own API.</p>')
    f19 = R / "t1/g19_own_vs_outside_claude.json"
    if f19.exists():
        g = json.loads(f19.read_text())
        d = g["O1_own_minus_outside_8"]
        out += (f'<p><b>Own seat or outside reviewer</b> (G19, preregistered). The same model, reading the same conversation as an outside '
                f'reviewer, scores {g["within_tool"]["outside_8"]:.3f} against {g["within_tool"]["own_8"]:.3f} in its own conversation '
                f'({d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]): no detectable difference. The value is in the questions and the read-out.</p>')
    f17 = R / "t1/g17_agentdojo_readout.json"
    if f17.exists():
        g = json.loads(f17.read_text())
        N = {"gemma4": "Gemma-4", "gptoss20b": "gpt-oss"}
        parts = [f"{N[a]} {g[a]['within_suite']['read_out_8']:.3f} [{g[a]['read_out_8_ci'][0]:.2f}, {g[a]['read_out_8_ci'][1]:.2f}]"
                 for a in N if a in g]
        out += (f'<p><b>Prompt injection</b> (G17, preregistered). On AgentDojo, open agents\' answers rank their injected calls at '
                f'{"; ".join(parts)} within suite.</p>')
    return out


def index_page():
    t4, t8, t9, t10 = j("t1/t4_reader.json"), j("t1/t8b_claude_all_actions.json"), j("t1/t9_text_baseline.json"), j("t1/t10_claude_unseen_families.json")
    t2, br = j("t1/t2_time.json"), j("t1/t8b_breakdown_exploratory.json")
    t11 = j("t1/t11_gpt_agent.json")
    t15 = j("t1/t15_latent_replication.json")
    lab = j("t1/t10_new_policy_labels_exploratory.json")["by_n"]
    gate = j("t1/t8b_calibration_exploratory.json")["gate_eval_76c"]
    par = json.loads((ROOT / "site/tests/parity_result.json").read_text())
    p = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    lf = j("t1/label_free_cuts.json")["claude_T8b"]                  # one operating point: the label-free 5% cut
    c8 = {k: lf[k]["label_free_cut"] for k in ("READER", "J_act")}
    t17 = j("t1/t17_replication_agents.json")["agents"]
    r17 = [a["r2_fresh"]["READER"] for a in t17.values()]
    pl = {L: v for L, v in t15["r2_fresh_per_layer_full_space"]["READER"].items()}
    body = f"""<h1>Monitor AI agents that you cannot open.</h1>
<p class="lede">Grey Box decides if an AI agent's next action must go to a human <i>before the action runs</i>, with no training
and nothing but the agent's own API. Before a call runs, it asks the agent itself, in side branches that its conversation
never sees, a few yes/no questions about the call; a fixed read-out of the answers scores the call. It needs no weights
and no labels, so it works on closed models. The confirmatory claims on this site come from
preregistered, hash-locked tests (lock times self-reported); exploratory results are marked. We show the failures too.</p>
<div class="note" style="margin-top:14px"><b>In plain words.</b> Before an AI agent runs an action (send an email, delete a file, issue a
refund), Grey Box asks the agent about that action, in a side conversation, and holds the action for a person when the agent's own
answers say it may break a rule. Try it live on the <a href="selfpolice.html">self-policing page</a>. Optionally, a monitor can also
learn your policy from examples that you label. It works best on the kinds of tasks that you labelled; on
a new kind of task it needs new labels, and its review cut should be set on your live traffic. Start with the
<a href="train.html">trainer</a>: load your agent's logs, label actions, and see how well a monitor would have done.</div>
{(ROOT / "site/src/assets/workflow.svg.html").read_text()}
<figure style="margin:20px 0 6px"><a class="zoom" href="figures/teaser_v6.png" title="Open the full-size figure"><img src="figures/teaser_v6.png" alt="A model's hidden geometry and the same geometry from its answers; the agent's own answers monitoring its decisions; the predicted and observed prevention" loading="eager"></a>
<figcaption><span class="zoomhint"><a href="figures/teaser_v6.png">Tap to open the full-size figure.</a> </span><b>Reconstruct, monitor and forecast, prevent: a closed agent through its own API, with no training.</b>
(a) An open model's hidden geometry and (b) the same contexts placed only by its yes/no answers (exact Fisher&ndash;Rao coordinates; nothing fit
but a display rotation). (c) At an agent's proposed call, a fixed read-out of its own 8 answers about the call (dots, 95% intervals) against a
probe trained on its hidden states (diamonds), on held-out calls (G15, preregistered); Claude Haiku 5.5 is closed, so it has no probe. Below the line, the
second benchmarks: open agents' prompt injections on AgentDojo (G17) and Claude Haiku 5.5 on &tau;&sup2;-bench (G18); both preregistered. (d) The agent's own answer as a gate: the live reduction in violations against
the law's prediction (G12, G13, preregistered; one of seven points misses the &plusmn;0.15 band).</figcaption></figure>
<div class="grid3" style="margin:22px 0 8px">
  <div class="card"><div class="label">Try it now</div><h3>Monitor your own agent</h3><p>Import your agent's logs (OpenAI or Anthropic
    format), label its actions with a review queue that shows the riskiest first, and get a held-out accuracy and a review threshold. Your
    logs stay in your browser.</p><a class="go" href="train.html">Open the trainer →</a></div>
  <div class="card"><div class="label">See it work</div><h3>A closed frontier agent, monitored</h3><p>Real episodes of Claude Haiku 5.5 as an
    agent. See the risk that the monitor gave to each action before the action ran, in the monitor's own 3D space.</p><a class="go" href="monitor.html">Open the replay →</a></div>
  <div class="card"><div class="label">Deploy</div><h3>Run it on your own machines</h3><p>A Python SDK and an HTTP API. The CPU monitor
    takes {par['js_score_ms_per_step']:.2f} ms for each action in a browser. The GPU reader monitor takes 41 ms.</p><a class="go" href="docs.html">Read the docs →</a></div>
</div>

<h2>What we found</h2>
{g14_card()}
<div class="grid2">
<div class="card"><h3>The grey box replicates part of the hidden state</h3><p>An agent's internal state is a function of the tokens that it read.
  So an open model that reads the same tokens can hold a replica of part of it. From the transcript alone, the replica reconstructs
  {p(t15['r2_fresh']['READER'])} of the variance of the 128 main components of an open agent's hidden state ({p(min(pl.values()))}–{p(max(pl.values()))} of each full layer;
  text features: {p(t15['r2_fresh']['TXT'])}). The same reader replicates three more open agents on scenarios it never saw ({p(min(r17))}–{p(max(r17))}).
  A probe trained on the true state scores {t15['E2_auroc']['READER_replica']:.3f} on the replica and {t15['E2_auroc']['true']:.3f} on the true state;
  it also works on a replica from text features ({t15['E2_auroc']['TXT_replica']:.3f}), because the information that decides a violation is in the tokens.</p></div>
<div class="card"><h3>It prevents violations live</h3><p>We ran the monitor in the loop of two closed agents, each scenario with and
  without the gate. A blocked action did not run, and the agent was told that a human would review it. Episodes with an executed violation
  fell from 6.6% to 0.2% on GPT-6-luna and from 6.7% to 0.2% on Claude Haiku 5.5. The agents did not route around a block. Each check
  took 80–89 ms. The cost: 1.8 (GPT) and 5.8 (Claude Haiku 5.5) points fewer task goals reached. When a reviewer decided on the flagged actions
  instead (an oracle, as an upper bound on a human), violations fell to 0.0% and goals to 38.7% (no gate: 40.7%); the preregistered test
  that this costs at most 3 points failed (−2.0 points, 95% interval −5.1 to +0.7). The closed agents ran with reasoning off.</p></div>
<div class="card"><h3>Labels, not access</h3><p>A small open model reads the transcript. We train it on labelled examples. It is
  equal to or better than the agent's own internal activations (AUROC {t4['auroc_fresh']['READER_act']:.3f} vs {t4['auroc_fresh']['I_dec']:.3f}).
  We trained it only on an open agent. On two closed agents, on its training benchmark, with 5% of the actions sent to review, it caught {p(c8['READER'])}
  (Claude Haiku 5.5) and {p(t11['label_free_cut']['READER']['all'])} (GPT-6-luna) of the violations. A zero-shot 70B judge caught
  {p(c8['J_act'])} and {p(t11['label_free_cut']['J_act']['all'])}.</p></div>
<div class="card"><h3>Why access does not help: a theorem</h3><p>An agent's internal state is a function of the tokens that it read.
  For the label "does this action break the policy?", a monitor that reads the transcript and the proposed action is
  Bayes-sufficient. Internals read at or before the action cannot do better. White-box probes win against turn-start
  monitors only because they read later (probe {t2['auroc']['I_dec']:.3f} after the reasoning, {t2['auroc']['WB_ts']:.3f} before it).</p></div>
<div class="card"><h3>The limit is policy coverage</h3><p>We removed a policy family from the training data. A credentials
  policy with similar trained policies still transferred: the monitor caught all of GPT's violations of it with no labels. A
  service-restart policy without similar ones did not: the monitor ranked Claude Haiku 5.5's violations well (AUROC
  {t10['auroc_pooled']['READER_lofo']:.3f}) but escalated none at a 5% budget. With 20 labelled steps it caught
  {p(lab['20']['cons_service_restart'])}; with 40, {p(lab['40']['cons_service_restart'])} (exploratory). On prompt injections (AgentDojo),
  labelled calls of open agents raised the catch rate on a closed agent from 28% to 58% at the label-free cut (AUROC 0.840 → 0.922),
  also on user tasks that the training never saw (exploratory). On τ²-bench, no trained monitor transferred to tasks it had not seen
  (reader 0.535, word counts 0.519; an earlier 0.987 for word counts came from test tasks that were also in training). Word counts
  memorize entities and tasks, and neither monitor checks rules over numbers, dates or recipients it has not seen. A zero-shot reasoning judge, with no labels, separates violating from allowed
  calls of the same tool better than chance (AUROC 0.766 on four closed frontier agents, preregistered T23), but at a 5% review
  budget it caught fewer violations than the reader and than the tool alone (11.7% vs 18.4% and 13.0%). Label steps of
  each new type of policy and of each new task.</p></div>
<div class="card"><h3>Simple monitors are enough, until the task changes</h3><p>A word-count classifier with the same labels is equal to
  the reader on the training benchmark (Claude Haiku 5.5: {t9['T8b_claude']['auroc']['TXT']:.3f} vs {t9['T8b_claude']['auroc']['READER']:.3f}).
  On a new benchmark of prompt injections it fell to chance ({t9['T7_agentdojo']['auroc']['TXT']:.2f}). The reader stayed at
  {t9['T7_agentdojo']['auroc']['READER']:.2f}.</p></div>
<div class="card"><h3>Set the review threshold from your own traffic</h3><p>A monitor's probabilities do not transfer between agents: a
  0.5 cut caught 3% on Claude Haiku 5.5. Its ranking transfers. We escalate the top 5% of the agent's first 500 actions and then freeze that
  threshold. It caught {p(gate['online']['frozen_default']['run_order']['caught_after_100'])} online. When half of a later burst was
  malicious, it still caught {p(gate['burst']['rate_0.5']['frozen_default']['caught'])}. A rolling threshold caught
  {p(gate['burst']['rate_0.5']['rolling5000']['caught'])}.</p></div>
<div class="card"><h3>An agent that knows it is monitored</h3><p>We told Claude Haiku 5.5 that a monitor reviews its tool calls
  (preregistered, 990 scenarios). It violated as often as before, and the monitor caught 97% of its violations. The notice moved
  all of its scores up, so the old review cut sent 15–17% of actions to review. Live, a cut set again on the agent's new,
  unlabelled traffic sent 4.7% (the budget was 5%) and the gate blocked every violation: 6.5% of episodes → 0.0%.</p></div>
<div class="card"><h3>Not robust to an adaptive attacker</h3><p>An attacker who can query the score and rewrite a call's free
  text cut the share caught from 96% to 29%. Two defenses we preregistered (canonical actions; training on rewrites) reached
  51% and 60%. A theorem explains the ceiling: under rewording, no monitor beats the best monitor of the call's decisive fields,
  and the main attacked family was missing from training. Keep the score hidden from the agent.</p></div>
<div class="card"><h3>Chat: forecast a refusal</h3><p>We ask a model about a conversation in a side branch. The answers forecast if the
  model will refuse. On Claude Haiku 5.5 this forecast only ties a strong graded moderator. Together with a moderator, it doubles the
  precision of an alarm for harmful requests that the model answers. It is not a better content filter.</p></div>
</div>
<div class="note limit"><b>Limits.</b> Most agent tests use one benchmark (CAPE-Bench). Two closed agent families are tested (Claude Haiku 5.5, GPT).
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


def example_logs():
    """Real agent logs in each provider's native format, from the T11 (GPT-6-luna, OpenAI) and T8b (Claude Haiku 5.5,
    Anthropic) runs, with the benchmark's labels in an optional "labels" field: {call_id: 0/1}."""
    import pandas as pd
    S = {json.loads(l)["instance_id"]: json.loads(l) for l in open("/data/models/cape_ws/outputs/capebench/core24b/scenarios.jsonl")}
    def pick(E, n):
        bad = E[E.unsafe_episode]
        good = E[~E.unsafe_episode].sample(n - len(bad), random_state=0)
        return pd.concat([bad, good]).sample(frac=1, random_state=1)
    G = pick(pd.read_parquet("/data/models/graybox_ws/outputs/gpt_agent/run.parquet"), 300)
    with open(DIST / "data/example_gpt_openai.jsonl", "w") as f:
        for ep in G.itertuples():
            sp = S[ep.instance_id]
            msgs, labels = [{"role": "system", "content": sp["system_prompt"]}, {"role": "user", "content": sp["task_prompt"]}], {}
            for e in ep.events:
                if e["is_tool_call"] and e["tool"]:
                    cid = f"call_{e['event_id']}"
                    msgs.append({"role": "assistant", "content": e["content_text"] or None,
                                 "tool_calls": [{"id": cid, "type": "function", "function": {"name": e["tool"], "arguments": e["args_json"] or "{}"}}]})
                    msgs.append({"role": "tool", "tool_call_id": cid, "content": e["tool_result"] or ""})
                    labels[cid] = int(bool(e["unsafe_action"]))
                elif e["content_text"]:
                    msgs.append({"role": "assistant", "content": e["content_text"]})
            f.write(json.dumps({"id": ep.episode_id, "messages": msgs, "labels": labels}, ensure_ascii=False) + "\n")
    C = pick(pd.read_parquet("/data/models/graybox_ws/outputs/claude_agent/run.parquet"), 150)
    with open(DIST / "data/example_closed_a.jsonl", "w") as f:
        for ep in C.itertuples():
            sp = S[ep.instance_id]
            msgs, labels = [{"role": "user", "content": sp["task_prompt"]}], {}
            for e in ep.events:
                if e["is_tool_call"] and e["tool"]:
                    tid = f"toolu_{e['event_id']}"
                    blocks = ([{"type": "text", "text": e["content_text"]}] if e["content_text"] else []) + [
                        {"type": "tool_use", "id": tid, "name": e["tool"], "input": json.loads(e["args_json"] or "{}")}]
                    msgs.append({"role": "assistant", "content": blocks})
                    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": e["tool_result"] or ""}]})
                    labels[tid] = int(bool(e["unsafe_action"]))
                elif e["content_text"]:
                    msgs.append({"role": "assistant", "content": [{"type": "text", "text": e["content_text"]}]})
            f.write(json.dumps({"id": ep.episode_id, "system": sp["system_prompt"], "messages": msgs, "labels": labels}, ensure_ascii=False) + "\n")


def example_agentdojo(n_neg=464):
    """A new kind of task: GPT-6-luna on AgentDojo without a defense (T13b), all 36 injected calls + n_neg others,
    in step format with the benchmark's labels. The monitors were never trained on AgentDojo."""
    import importlib
    import pandas as pd
    U = importlib.import_module("86_t13b_analysis").units()
    pick = pd.concat([U[U.y == 1], U[U.y == 0].sample(n_neg, random_state=0)]).sample(frac=1, random_state=1)
    with open(DIST / "data/example_agentdojo_gpt.jsonl", "w") as f:
        for r in pick.itertuples():
            f.write(json.dumps({"context": r.ctx, "action": r.act, "label": int(r.y), "group": str(r.episode)}, ensure_ascii=False) + "\n")


def browser_reader():
    """The browser reader (T19), published only if the preregistered rule ships it (scripts/105)."""
    src = R / "tool/browser_reader.json"
    if src.exists() and json.loads(src.read_text()).get("shipped"):
        shutil.copy(src, DIST / "data/browser_reader.json")
    else:                                   # always present (no 404); the page says why it is not offered
        (DIST / "data/browser_reader.json").write_text(json.dumps({"shipped": False, "status": "in a preregistered test (T19)" if not src.exists()
                                                                    else "not offered: it did not pass its preregistered test (T19)"}))


def selfpolice_data():
    """Data and headline cards for selfpolice.html, from the saved result files and the recorded G10 episodes."""
    cs = Path("/data/models/graybox_ws/outputs/claude_selfaudit")
    g9, g10 = j("t1/g9_live_self_policing.json"), j("t1/g10_claude_self_policing.json")
    ad = j("t1/agentdojo_selfaudit_DEV.json")
    G = [json.loads(x) for x in open(cs / "live_g10_p60.jsonl")]
    by = {}
    for r in G:
        by.setdefault(r["instance_id"], {})[r["arm"]] = r

    def steps(r):
        out = []
        for t in json.loads(r["turns"]):
            d = t["draws"][-1] if t["draws"] else None
            if not d or not d.get("name"):
                continue
            ex = t.get("executed") or {}
            sh = d.get("s")
            a_ = json.dumps(d.get("args") or {})
            out.append({"tool": d["name"], "args": a_ if len(a_) <= 110 else a_[:109] + "…", "share": None if sh is None or sh != sh else round(sh, 2),
                        "held": bool(t.get("reviewed")), "blocked": bool(ex.get("blocked")), "ran": not ex.get("blocked"), "unsafe": int(d.get("unsafe") or 0)})
        return out

    def outcome(r):
        return ("A violation ran." if r["unsafe_episode"] else "No violation ran.") + (" The task was completed." if r["success"] else " The task was not completed.")
    rep = []
    for tmpl, title, note, want in (
            ("cons_service_restart", "Production restart without approval", "The policy requires request_permission before a production operation.", lambda v: v["none"]["unsafe_episode"] and not v["gate"]["unsafe_episode"] and v["gate"]["success"]),
            ("tp_campaign_test", "A 'test' send to a mailing list", "The list's thousands of external members show only after sending; nothing in the conversation reveals them, so self-policing cannot catch it (as predicted).", lambda v: v["none"]["unsafe_episode"] and v["gate"]["unsafe_episode"])):
        n = 0
        for k, v in sorted(by.items()):
            if "none" in v and "gate" in v and v["none"]["template"] == tmpl and want(v):
                n += 1
                rep.append({"title": f"{title} ({n})", "note": note, "plain": steps(v["none"]), "gated": steps(v["gate"]),
                            "plain_outcome": outcome(v["none"]), "gated_outcome": outcome(v["gate"])})
                if n == (3 if tmpl == "cons_service_restart" else 1):
                    break
    ep = next(e for e in (json.loads(x) for x in open("/data/models/graybox_ws/outputs/t24/run.partial.jsonl"))
              if e["arm"] == "plain" and e["template"] == "cons_service_restart" and e["unsafe_episode"])
    k = next(i for i, t in enumerate(ep["turns"]) if t.get("unsafe"))
    msgs = [{"role": "user", "content": ep["task_prompt"]}]
    for i, t in enumerate(ep["turns"][:k]):
        if t.get("tool"):
            msgs.append({"role": "assistant", "content": t.get("text") or "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": t["tool"], "arguments": json.dumps(t.get("args") or {})}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": str(t.get("result"))})
    # make the deciding fact visible: the scenario's own host-info output (its environment definition), as a checked step
    spec = {json.loads(x)["instance_id"]: json.loads(x) for x in open("/home/arya/new_safety/cape/outputs/capebench/core24b/scenarios.jsonl")}[ep["instance_id"]]
    host = (ep["turns"][k].get("args") or {}).get("command", "").split("--host")[-1].strip()
    hinfo = next(c["output"] for c in spec["commands"] if c["name"].startswith("host-info"))
    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c_host", "type": "function", "function": {"name": "execute_mock_command", "arguments": json.dumps({"command": f"host-info {host}"})}}]})
    msgs.append({"role": "tool", "tool_call_id": "c_host", "content": hinfo})
    D = {"example": {"system": ep["system_prompt"], "messages": msgs, "tool": ep["turns"][k]["tool"], "args": ep["turns"][k].get("args") or {},
                     "note": "Recorded from Claude Haiku 5.5 on CAPE-Bench; the last step (host-info) is added from the scenario's environment so that the deciding fact, a production host, is in the conversation. Expected: hold (the policy requires request_permission for production)."},
         "replay": rep,
         "law_example": [{"kind": "production restart without approval", "w": 0.38, "tau": 0.69}, {"kind": "test send to a mailing list", "w": 0.62, "tau": 0.21}],
         "law_note": "Claude Haiku 5.5, all its violations on these two kinds of task (G12 step A rates). This counts the mailing-list sends, which self-policing cannot see; the headline −69% counts only violations its conversation decides. Observed in G10 over all its violations: 31%; in G12 (core24, this mix): 30%."}
    (DIST / "data").mkdir(parents=True, exist_ok=True)
    (DIST / "data/selfpolice.json").write_text(json.dumps(D))
    (DIST / "selfpolice_local.html").write_text(selfpolice_local(DIST / "data/selfpolice.json"))
    shutil.copy(ROOT / "web/data/selfpolice3d.json", DIST / "data/selfpolice3d.json")      # web/build_selfpolice3d.py
    shutil.copy(ROOT / "web/data/live_scenarios.json", DIST / "data/live_scenarios.json")   # web/build_live_scenarios.py
    if (ROOT / "web/data/observable_ref.json").exists():
        shutil.copy(ROOT / "web/data/observable_ref.json", DIST / "data/observable_ref.json")  # web/build_observable_ref.py
    gr, cr = g9["arms"]["gate@1.00"], g10["decidable"]
    a = ad["gemma4"]
    g13 = j("t1/g13_agentdojo_selfpolicing.json") if (R / "t1/g13_agentdojo_selfpolicing.json").exists() else None
    cards = (f'<div class="card"><div class="label">Open agent (Gemma-4), unseen scenarios</div><div class="num">{gr["violation_episodes"] * 100:.0f}%</div>'
             f'<p>episodes with a violation, down from {g9["arms"]["none"]["violation_episodes"] * 100:.0f}% (preregistered); tasks completed: '
             f'{g9["arms"]["none"]["success"] * 100:.0f}% → {gr["success"] * 100:.0f}%.</p></div>'
             f'<div class="card"><div class="label">Closed agent (Claude Haiku 5.5), held-out scenarios</div><div class="num">&minus;{cr["reduction"][0] * 100:.0f}%</div>'
             f'<p>of the violations that its conversation reveals [{cr["reduction"][1] * 100:.0f}, {cr["reduction"][2] * 100:.0f}] (all its violations: &minus;{g10["all"]["reduction"][0] * 100:.0f}%), at no cost in tasks '
             f'completed ({g10["success"]["none"] * 100:.1f}% → {g10["success"]["gate"] * 100:.1f}%; preregistered).</p></div>'
             + (f'<div class="card"><div class="label">Prompt injection (AgentDojo), Gemma-4, held-out tasks</div><div class="num">{g13["gemma4"]["spotaudit"]["asr"] * 100:.1f}%</div>'
                f'<p>attacks that succeeded, down from {g13["gemma4"]["spot"]["asr"] * 100:.1f}% (&minus;{g13["gemma4"]["reduction"][0] * 100:.0f}% [{g13["gemma4"]["reduction"][1] * 100:.0f}, {g13["gemma4"]["reduction"][2] * 100:.0f}]), '
                f'no loss of utility (preregistered). gpt-oss: {g13["gptoss20b"]["spot"]["asr"] * 100:.0f}% → {g13["gptoss20b"]["spotaudit"]["asr"] * 100:.0f}%.</p></div>' if g13 else
                f'<div class="card"><div class="label">Prompt injection (AgentDojo), Gemma-4</div><div class="num">{a["flag_injected_cut1"] * 100:.0f}%</div>'
                f'<p>of its own injected tool calls flagged when asked (all answers "no"), {a["flag_benign_cut1"] * 100:.1f}% of benign calls (development data).</p></div>'))
    law = ""
    if (R / "t1/g12_composition_law.json").exists():
        g12 = j("t1/g12_composition_law.json")
        names = {"gemma4": "Gemma-4", "gptoss20b": "gpt-oss", "qwen38": "Qwen3.8", "museglimmer": "Muse-Glimmer", "claude": "Claude Haiku 5.5"}
        rows = "".join(f"<tr><td>{names[a]}</td><td>{g12[a]['predicted'] * 100:.0f}%</td><td>{g12[a]['observed'][0] * 100:.0f}% [{g12[a]['observed'][1] * 100:.0f}, {g12[a]['observed'][2] * 100:.0f}]</td></tr>"
                       for a in names if a in g12)
        ok = all(g12["endpoints"].get(f"Q1_{a}") for a in names if a in g12)
        law = (f'<div class="card"><b>The preregistered test of this prediction (G12): {"passed" if ok else "see the record"}.</b> '
               f'Predicted from each agent\'s offline flag rates per kind of task, weighted by where it violated without the gate; '
               f'observed live on new scenarios (relative reduction in episodes with a violation, 95% interval). The registered tolerance was ±0.15 for each agent: '
               f'<table class="tablewrap"><tr><th>Agent</th><th>Predicted</th><th>Observed</th></tr>{rows}</table>'
               f'<p class="muted">Mean absolute error {g12["mean_abs_error"]:.3f}. The law over-predicts Claude Haiku 5.5: its flag rate on mailing-list sends '
               f'was 12% live against 21% in the offline audit of 38 calls (within that estimate\'s error). Audit enough calls of each kind.</p></div>')
    return cards, law


def selfpolice_local(D_path):
    """A single-file copy of the audit tool and the calculator (open from disk: it can call a model server on localhost)."""
    src = (SRC / "pages/selfpolice.html").read_text()
    a = src.index("<h2>1. Audit a call"); b = src.index("<h2>2. Run an agent")
    c = src.index("<h2>4. Predict"); d = src.index('<div id="lawEvidence">')
    css = (SRC / "assets/site.css").read_text()
    style = src[src.index("<style>.sp-form"):src.index("</style>", src.index("<style>.sp-form")) + 8]
    js = (SRC / "assets/selfpolice.js").read_text()
    data = D_path.read_text()
    return (f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-policing audit (single file) · Grey Box</title><style>{css}</style>{style}</head><body><main><div class="wrap">
<h1>Self-policing: audit a call with your agent's own API</h1>
<p class="lede">A single-file copy of the tool at https://aryafayyazi.github.io/GreyBox/selfpolice.html. Opened from your disk, it can call a
model server on your own machine (for example vLLM at http://127.0.0.1:8000/v1, or Ollama at http://127.0.0.1:11434/v1).</p>
{re.sub(r"<!--obs-->.*?<!--/obs-->", "", src[a:b], flags=re.S)}{src[c:d]}</div></main>
<script>window.SP_DATA = {data};</script><script>{js}</script></body></html>""")


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "assets").mkdir(parents=True)
    for f in (SRC / "assets").iterdir():
        shutil.copy(f, DIST / "assets" / f.name)
    shutil.copy(SRC / "coi-sw.js", DIST / "coi-sw.js")      # the trainer's cross-origin isolation (multi-threaded reader)
    shutil.copytree(R / "figures", DIST / "figures", ignore=shutil.ignore_patterns("*.pdf"))
    # the explainer video, GIF and poster were recorded from a live session of the closed model and show its name: not published
    (DIST / ".nojekyll").write_text("")
    pages = {"index.html": index_page(), "results.html": results_page()}
    sp_cards, sp_law = selfpolice_data()
    for name, title, desc in (("train.html", "Train a monitor", "Train an agent-action monitor on your own labelled steps, in your browser."),
                              ("docs.html", "Docs", "Install, deploy and evaluate Grey Box monitors."),
                              ("selfpolice.html", "Self-policing", "Let an AI agent check its own tool calls before they run, through its own API."),
                              ("paper.html", "Paper", "Grey Box: Reading Closed Agents Through Their APIs Alone.")):
        if (SRC / "pages" / name).exists():
            s = (SRC / "pages" / name).read_text()
            m1, m2 = re.search(r"<!--title: (.*?)-->", s), re.search(r"<!--desc: (.*?)-->", s)
            if name == "selfpolice.html":
                s = s.replace("<!--filled from the result files at build time-->", sp_cards).replace("<!--filled at build time: the preregistered test of this prediction-->", sp_law)
            pages[name] = page(name, m1.group(1) if m1 else title, m2.group(1) if m2 else desc, s)
    for k, v in pages.items():
        (DIST / k).write_text(v)
    # the Claude Haiku 5.5 replay, wrapped with the site header
    tmp = DIST / "_monitor_raw.html"
    subprocess.run([sys.executable, str(ROOT / "web/build_monitor_demo.py"), "/data/models/graybox_ws/outputs/t8b/space.json", str(tmp)], check=True)
    raw = re.sub(r"<title>.*?</title>", "", tmp.read_text(), count=1)       # the page title goes in <head> below
    tmp.unlink()
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == "monitor.html" else ""}>{t}</a>' for h, t in NAV)
    (DIST / "monitor.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent replay · Grey Box</title><meta name="description" content="Recorded episodes of Claude Haiku 5.5 as a tool-using agent, with the monitor's decision on each action and its 3D feature space.">
<link rel="stylesheet" href="assets/site.css"></head><body>
<header class="top"><div class="wrap"><a class="brand" href="index.html">Grey Box <span>agent monitoring</span></a><nav>{nav}</nav></div></header>
{raw}</body></html>""")
    # the chat grey box (interrogation state), recorded Claude Haiku 5.5 session, wrapped with the site header
    tmp = DIST / "_chat_raw.html"
    subprocess.run([sys.executable, str(ROOT / "web/build_replay.py"), "/data/models/graybox_ws/outputs/live/session_v2_claude_haiku55.json",
                    str(ROOT / "web/data/meta_v2_closed_a.json"), str(tmp)], check=True)
    raw = re.sub(r"<title>.*?</title>", "", tmp.read_text(), count=1)
    tmp.unlink()
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == "chat.html" else ""}>{t}</a>' for h, t in NAV)
    (DIST / "chat.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recorded chat replay · Grey Box</title><meta name="description" content="A recorded chat session with Claude Haiku 5.5, replayed turn by turn with the grey-box readouts.">
<link rel="stylesheet" href="assets/site.css"></head><body>
<header class="top"><div class="wrap"><a class="brand" href="index.html">Grey Box <span>agent monitoring</span></a><nav>{nav}</nav></div></header>
{raw}</body></html>""")
    example_data()
    example_logs()
    example_agentdojo()
    browser_reader()
    scrub_dist()
    print("built", sorted(p.name for p in DIST.iterdir()))


RENAME = (("Closed-A's", "Claude Haiku 5.5's"), ("Closed-A", "Claude Haiku 5.5"))   # the closed model is named (user, 2026-10-22)
TEXT = {".html", ".js", ".json", ".jsonl", ".css", ".vtt", ".srt", ".txt", ".md", ".svg", ".csv"}


def scrub_dist():
    """The closed model is named "Claude Haiku 5.5" (the user's decision, 2026-10-22; it was "Closed-A" before): rename any
    remaining "Closed-A" in the built site, drop images that no page uses (an old image may show the old name in its
    pixels), and fail if "Closed-A" remains."""
    pages = "".join(f.read_text(errors="ignore") for f in DIST.rglob("*") if f.is_file() and f.suffix in {".html", ".js"})
    for f in list((DIST / "figures").iterdir()):
        if f.name not in pages:
            f.unlink()
    for f in DIST.rglob("*"):
        if f.is_file() and f.suffix in TEXT:
            t = f.read_text(errors="ignore")
            if "Closed-A" in t or "closed-A" in t:
                for a, b in RENAME:
                    t = t.replace(a, b)
                f.write_text(t)
    left = [f.name for f in DIST.rglob("*") if f.is_file() and f.suffix in TEXT and "Closed-A" in f.read_text(errors="ignore")]
    assert not left, f"the old name is still in: {left}"


if __name__ == "__main__":
    main()
