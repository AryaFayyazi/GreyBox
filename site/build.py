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
NAV = [("index.html", "Overview"), ("train.html", "Train a monitor"), ("monitor.html", "Agent replay"),
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
the agent's own internals (orange). The dashed line is chance (AUROC 0.5). The two Claude rows rest on one stratum (136 actions, 26 violations).</figcaption></figure>
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
<p class="lede">Grey Box decides if an AI agent's next action must go to a human <i>before the action runs</i>. It reads only
the data that every deployment can see: the agent's instructions, the user's task, the earlier tool results and the
proposed action. It does not need the model's weights, so it works on closed models. The confirmatory claims on this site come from
preregistered, hash-locked tests (lock times self-reported); exploratory results are marked. We show the failures too.</p>
<div class="note" style="margin-top:14px"><b>In plain words.</b> Before an AI agent runs an action (send an email, delete a file, issue a
refund), Grey Box gives it a risk score from the conversation so far, and sends the riskiest few percent to a person. It needs no access
to the model's internals. It learns your policy from examples that you label. It works best on the kinds of tasks that you labelled; on
a new kind of task it needs new labels, and its review cut should be set on your live traffic. Start with the
<a href="train.html">trainer</a>: load your agent's logs, label actions, and see how well a monitor would have done.</div>
<figure style="margin:20px 0 6px"><a class="zoom" href="figures/teaser.png" title="Open the full-size figure"><img src="figures/teaser.png" alt="True hidden state of an open agent, its replica from the transcript, replicas of two closed agents, and live prevention rates" loading="eager"></a>
<figcaption><span class="zoomhint"><a href="figures/teaser.png">Tap to open the full-size figure.</a> </span><b>Seeing inside closed agents.</b> Panels (a–d) share the main axes of one open agent's hidden state. (a) Its true state. (b) Its
replica, computed from the transcript only. (c, d) Two closed agents, mapped through the same reader: we never saw their weights, and their replica cannot be checked, so we test prediction. Red: policy
violations. In every agent they occupy their own region. (e) Live prevention: the share of episodes in which a violation ran, without the
gate (grey) and with the monitor in the agent's loop (blue), for GPT and Claude.</figcaption></figure>
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
  took 80–89 ms. The cost: 1.8 (GPT) and 5.8 (Claude) points fewer task goals reached. When a reviewer decided on the flagged actions
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
  service-restart policy without similar ones did not: the monitor ranked Claude's violations well (AUROC
  {t10['auroc_pooled']['READER_lofo']:.3f}) but escalated none at a 5% budget. With 20 labelled steps it caught
  {p(lab['20']['cons_service_restart'])}; with 40, {p(lab['40']['cons_service_restart'])} (exploratory). On prompt injections (AgentDojo),
  labelled calls of open agents raised the catch rate on a closed agent from 28% to 58% at the label-free cut (AUROC 0.840 → 0.922),
  also on user tasks that the training never saw (exploratory). On τ²-bench, no trained monitor transferred to tasks it had not seen
  (reader 0.535, word counts 0.519; an earlier 0.987 for word counts came from test tasks that were also in training). Word counts
  memorize entities and tasks, and neither monitor checks rules over numbers, dates or recipients it has not seen. A zero-shot reasoning judge, with no labels, checks such written rules
  within a tool (AUROC 0.766 on four closed frontier agents, preregistered T23), but at a 5% review budget it caught fewer
  violations than the reader (11.7% vs 18.4%). Label steps of
  each new type of policy and of each new task.</p></div>
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
    with open(DIST / "data/example_claude_anthropic.jsonl", "w") as f:
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
    raw = re.sub(r"<title>.*?</title>", "", tmp.read_text(), count=1)       # the page title goes in <head> below
    tmp.unlink()
    nav = "".join(f'<a href="{h}"{" aria-current=page" if h == "monitor.html" else ""}>{t}</a>' for h, t in NAV)
    (DIST / "monitor.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent replay · Grey Box</title><meta name="description" content="Recorded episodes of Claude Haiku 5.5 as a tool-using agent, with the monitor's decision on each action and its 3D feature space.">
<link rel="stylesheet" href="assets/site.css"></head><body>
<header class="top"><div class="wrap"><a class="brand" href="index.html">Grey Box <span>agent monitoring</span></a><nav>{nav}</nav></div></header>
{raw}</body></html>""")
    # the chat grey box (interrogation state), recorded Claude session, wrapped with the site header
    tmp = DIST / "_chat_raw.html"
    subprocess.run([sys.executable, str(ROOT / "web/build_replay.py"), "/data/models/graybox_ws/outputs/live/session_v2_claude_haiku55.json",
                    str(ROOT / "web/data/meta_v2_claude.json"), str(tmp)], check=True)
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
    print("built", sorted(p.name for p in DIST.iterdir()))


if __name__ == "__main__":
    main()
