# Grey Box

[![ci](https://github.com/AryaFayyazi/GreyBox/actions/workflows/ci.yml/badge.svg)](https://github.com/AryaFayyazi/GreyBox/actions/workflows/ci.yml)

**Replicating agents' hidden states from their transcripts, to predict and prevent the violations of closed agents.**

![Seeing inside closed agents](https://raw.githubusercontent.com/AryaFayyazi/GreyBox/main/results/figures/teaser.png)

An agent's internal state is a function of the tokens that it read. So an open model that reads the same tokens can
hold a replica of part of it. From the transcript alone, the replica reconstructs 66% of the variance of the 128 main
components of an agent's hidden state (51–54% of each full layer; text features: 59%), and readouts trained on the
true state work on the replica (T15). Readouts also work on a replica from text features, because the information that
decides a violation is in the tokens. The same reader replicates three more open agents (46–54%, T17). We use the
reader to predict and block policy violations of closed agents (run with reasoning off) before they run.

Grey Box monitors AI agents that you cannot open. It decides if an agent's next action must go to a human
**before the action runs**. It reads only the data that every deployment can see: the agent's instructions, the
user's task, the earlier tool results and the proposed action. It does not need the model's weights, so it works on
closed models.

**Website: https://aryafayyazi.github.io/GreyBox/**
- Train a monitor on your own labelled agent steps, in your browser. The browser does not upload your data.
- See a monitor on a closed frontier agent (Claude Haiku 5.5), action by action, in 3D.
- Read every preregistered result, with the failures.

## Quick start

```
pip install "greybox-monitor[server,text]"     # import graybox; [reader] adds the GPU reader. (PyPI "graybox" is another project.)
```

| Monitor | Hardware and speed | Use it when |
|---|---|---|
| `TextMonitor` (TF-IDF and logistic regression) | CPU, about 0.2 ms for each action | your tools and policies are the same as in your labelled data |
| Reader monitor (Qwen3-8B reads the transcript; a linear probe) | 1 GPU, 41 ms for each action | your tasks change; it is more robust to new tasks |

1. Train a monitor on your own labelled logs (OpenAI or Anthropic conversations with a `"labels"` field
   `{call_id: 0/1}`), in seconds on a CPU, or in the [browser trainer](https://aryafayyazi.github.io/GreyBox/train.html).
   Example data to try: [300 GPT conversations](https://aryafayyazi.github.io/GreyBox/data/example_gpt_openai.jsonl),
   [150 Claude conversations](https://aryafayyazi.github.io/GreyBox/data/example_claude_anthropic.jsonl).
   ```
   python -m graybox.train_text labelled_logs.jsonl --out greybox-text-monitor.json   # --chat for agents that talk with a user
   ```
   It reports the AUROC with whole tasks held out, pooled and within tool, and the catch rate at the review budget. The
   privacy filter (on unless you pass `--no-privacy`) masks emails and URLs and drops terms with digits, codes or rare words;
   names that recur in your logs stay in the model file.
   `--eval-only --group-by <field>` reports the score with other groups held out (for example task families).
2. Put it in your agent's loop:
   ```python
   from graybox.guard import Guard
   from graybox.monitor import QuantileGate, TextMonitor

   guard = Guard(TextMonitor.load_json("greybox-text-monitor.json"))
   messages.append(reply)                         # the assistant reply that holds the calls (needed for parallel calls)
   for call in reply.tool_calls or []:            # OpenAI; check_anthropic(system, messages, block) for tool_use blocks
       v = guard.check_openai(messages, call)
       result = run_tool(call) if v.allow else v.block_message
       messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
   ```
   SDK objects (OpenAI messages, Anthropic content blocks) and plain dicts give the same scores.
3. On a new kind of task, a cut stored in the file can escalate almost nothing: on held-out task families it caught
   0 of 115 violations (three splits of the public GPT example, `scripts/114_cut_under_shift_public.py`). A live cut on
   the agent's own traffic, `Guard(monitor, gate=QuantileGate(0.05))` (or the server's `"gate": true`), keeps the review
   load at 5%, but it caught only 29 of 115 there (1 of 39 in one split): it does not restore detection. Label actions of the new tasks and retrain, or use the
   reader. A new gate sends every action to review until it has seen 100: seed it with `gate.fit(past_scores)`.

Train the reader monitor on your own labels with one command on a GPU:
`python -m graybox.train_reader labelled.jsonl --out my_reader_probe.joblib`, then load it with
`graybox.reader_load.load_reader`. The released probes are in
[release v0.2.0](https://github.com/AryaFayyazi/GreyBox/releases/tag/v0.2.0): `reader_probe.joblib` (CAPE-Bench) and
`reader_probe_cape_agentdojo.joblib` (CAPE-Bench and prompt injection). For the reader, always use `QuantileGate`: its
probability does not transfer between agents.

The browser trainer imports OpenAI and Anthropic agent logs, shows each action with the conversation before it, lets
you label actions (a review queue shows the riskiest first), and trains the monitor. Its importer and the guard build
the same steps as our research pipeline (979/979 GPT and 373/373 Claude units identical; `results/tool/`).

Self-host the HTTP API: `uvicorn graybox.monitor_server:app` (see `deploy/` for Docker). Your transcripts stay on your
machine.

## Main result

**Theory** (`docs/theory.md`, section T6). An agent's internal state is a function of the tokens that the agent read.
For an action-level label ("does this proposed action break the policy, given the facts that the agent saw?"), a
monitor that reads the transcript and the proposed action is Bayes-sufficient. No monitor that reads the agent's
internals at or before the action can do better. The remaining problem is statistical: learn the policy from
labelled examples.

**Evidence.** All tests below are preregistered and hash-locked. `docs/PREREGISTRATION_RECORD.md` gives each lock and
each outcome, with the failures and the deviations. AUROC 0.5 is chance; 1.0 is perfect.

| Test | Result |
|---|---|
| T2: timing | A probe on the agent's internals scores 0.979 after the agent's reasoning, and 0.841 at the start of the turn. The time of reading explains the white-box advantage. |
| T3: labels | Without labels, a 70B judge that scores resampled actions gets 0.782. The probe on the internals gets 0.976. |
| T4, T5: a transcript reader with labels | 0.994 on the same scenarios (the agent's internals: 0.979). 0.969 on unseen task families (internals: 0.976). |
| T6: another open agent | Trained on Gemma, applied to gpt-oss: 0.915 (zero-shot judge: 0.833). |
| T7: an external benchmark | AgentDojo prompt injection, no AgentDojo training: 0.78 (judge: 0.54). On gpt-oss it beats CAPE's white-box gate (0.82 vs 0.60). |
| T8b: a closed frontier agent | Claude Haiku 5.5, 3,128 actions. Trained only on Gemma, the reader catches 94.1% of the 68 violations when the top 5% of actions go to review, with a cut set without labels (95.6% with a cut set on labelled safe actions). The judge catches 0%. Claude ran with thinking off. Pooled AUROC 0.989 (judge 0.570; exploratory). The preregistered within-template AUROC (0.991 vs 0.633) uses only one stratum: 136 actions, 26 violations. |
| T9: reader or labels? | A TF-IDF monitor with the same labels is equal to the reader on the training benchmark (Claude 0.997 vs 0.991). On unseen task families the reader is better (0.969 vs 0.950). On AgentDojo, TF-IDF is at chance (0.50 vs 0.78). |
| T10: a policy that the monitor never saw | The reader still ranks the violations well (pooled 0.912; judge 0.570). But at a 5% budget it escalates 0 of 26 violations. With 20 to 40 labelled examples of that policy, it catches 85% to 94% (exploratory). |
| T11: a second closed family (GPT) | GPT-6-luna, 3,627 actions, 66 violations in 6 families. Pooled AUROC 0.996 (judge 0.778; tool identity alone 0.856). Within tool 0.994 (judge 0.956). At a 5% budget it catches 98.5% (judge 37.9%). All three preregistered endpoints pass. |
| T12: labels for a new policy, prospective | We held out a credentials policy. With no labels of it, the reader caught all 18 of GPT's violations of it. The preregistered H1 (labels improve the catch rate) fails: there was nothing to fix. A new policy that resembles a trained one can transfer; one that does not (T10) needs labels. |
| T13, T13b: a closed agent on an external benchmark | GPT-6-luna on AgentDojo. With the spotlighting defense, attacks succeed in 2.1% of episodes (4 injected calls): not tested. Without the defense, 36 injected calls: reader 0.840 vs judge 0.787 (+0.052 [−0.026, +0.130]). The preregistered test fails: on an external benchmark the reader is not significantly better than a zero-shot judge for this closed agent. |
| T14: labels of the new policy, from open agents | We added labelled AgentDojo calls of the two open agents to the training data, and tested on the closed GPT agent. AUROC 0.840 → 0.922 (judge 0.787); injected calls caught at the label-free 5% cut 28% → 58% (36% → 92% with a cut set on benign calls). Both preregistered endpoints pass. Exploratory: on the confirmation split only, whose user tasks the training calls never saw, 0.929 vs judge 0.786 (+0.143 [0.075, 0.227]). On this benchmark, labels of a policy transferred from open agents to a closed one. |
| T15: does the grey box replicate the hidden state? | From the transcript and the proposed action, the reader replica reconstructs 66.4% of the variance of the 128 main components of the agent's decision state (51–54% of each full layer; text features: 58.6%). A probe trained on the true state scores 0.987 on the replica and 0.984 on the true state; on a replica from text features it scores 0.983, so the information that decides a violation is in the tokens. |
| T17: replication on three more open agents | gpt-oss-20b (mixture of experts), Qwen3.8-27B (hybrid linear attention), Muse-Glimmer-30B (dense), on scenarios that the map never saw. The same reader replicates each agent's decision state better than text features: R² 0.49, 0.54, 0.46 (text 0.39, 0.44, 0.40). A probe trained on the true states works on every replica (0.900 → 0.924, 0.955 → 0.972, 0.929 → 0.969). All six endpoints pass. Replication is partial (about half of the variance), and the readout also works on a text replica. |
| T16: live prevention on a closed agent | The monitor in GPT-6-luna's loop (500 scenarios, with and without the gate). Episodes with an executed violation: 6.6% → 0.2% (−0.064 [−0.088, −0.044]). No violation after a block. Goals reached 47.4% → 45.6%. Round trip 89 ms. |
| T16b: live prevention on Claude | The same live test on Claude Haiku 5.5 (450 scenarios). Episodes with an executed violation: 6.7% → 0.2% (−0.064 [−0.087, −0.044]). No violation after a block. Goals reached 40.7% → 34.9%. Round trip 80 ms. |
| T20: a second outside benchmark (τ²-bench), two closed agents | Monitors trained on four frontier agents' published, labelled τ² trajectories; tested on Claude Haiku 5.5 and GPT-6-luna with a persistent customer (19 violations in 262 writes). Within tool: word counts 0.987 (G2 passes), reader 0.738 (G1 fails). But the test tasks also occur in training: see T21. |
| T21: τ²-bench, unseen tasks | T20 with every training write of a test task removed (554 writes of 28 other tasks remain). Within tool: reader 0.535, word counts 0.519; both preregistered endpoints fail (reader − word counts +0.017 [−0.514, +0.539]; reader − tool +0.035 [−0.230, +0.259]). Word counts with entities masked: 0.366. T20's word-count score was memorization of the tasks. |
| T22: a zero-shot reasoning judge on τ²-bench | Qwen3-32B with thinking reads the policy, the conversation and the proposed write; no labels. On T20's 262 writes: within tool 0.804 (T21's reader 0.535). Both preregistered endpoints fail: judge − tool +0.304 [−0.084, +0.427], judge − reader +0.268 [−0.074, +0.511]; only 7 tasks have a violation. |
| T23: the same judge on four closed frontier agents | T22's judge, unchanged, on the 894 published τ² writes of Claude 3.7 Sonnet, GPT-4.1, GPT-4.1-mini and o4-mini (227 violations, 29 tasks). Within tool: judge 0.766, reader trained on other tasks 0.646. K1 passes (judge − tool +0.266 [0.151, 0.338]); K2 fails (judge − reader +0.120 [−0.047, +0.274]). With exactly 5% of writes reviewed the reader catches more: 18.4% vs 11.7% (most possible 19.7%). The judge separates violating from allowed writes of the same tool without labels, but as a review queue it is below the tool alone (13.0%). |
| T16c: hold for review instead of block | On Claude (450 scenarios), escalated actions go to a reviewer (here an oracle; an upper bound on a human). Episodes with an executed violation: 6.7% → 0.0% (R1 passes). Goals 38.7% (no gate 40.7%, blocking 34.9%); the preregistered "at most 3 points lost" fails (−0.020 [−0.051, +0.007]). Exploratory: episodes in which the reviewer let a held action run kept their goals (60.8% vs 58.8%); the gap is in episodes where nothing was held, which we suggest is variation between runs (untested: the control was not concurrent). 0.20 reviews per episode. |
| Review threshold without labels (exploratory) | A threshold from the agent's own unlabelled traffic, frozen after 500 actions: 95.4% caught online, 1.9% of safe actions escalated. With a burst of violations, it stays at 94%; a rolling threshold falls to 36%. |

**Results that did not confirm** are in the record too. Examples: the agent's answers alone do not match the
white-box probe (T1, F3), and resampling scored by an LLM judge loses to the probe (T3).

## Limits

- Most agent tests use one benchmark, CAPE-Bench. On the external test (AgentDojo), the reader beats a zero-shot judge
  for open agents (0.78 vs 0.54), but not significantly for a closed agent (0.840 vs 0.787, T13b).
- On τ²-bench (T20: Claude Haiku 5.5 and GPT-6-luna with a customer who insists after a refusal; 262 writes, 19
  violations), monitors trained on four frontier agents' labelled trajectories: word counts reached within-tool AUROC
  0.987 (preregistered G2 passes), the reader only 0.738 (G1 fails; reader − word counts −0.250 [−0.518, −0.077]).
  But every test task also occurs in training. With those tasks removed (T21), neither monitor transfers: reader 0.535,
  word counts 0.519. T20's word-count score was memorization of tasks (codes, names, dates). τ² violations break
  explicit rules over facts of the case (dates, amounts, cabin); label-trained readouts do not learn these checks for
  new tasks. A zero-shot reasoning judge, which needs no labels, separates violating from allowed writes of the same tool
  (T23: 0.766 on four closed frontier agents), but at a 5% review budget it catches fewer violations than the reader. (T18, on Claude
  alone, had too few violations to test.)
- Catch rates are at exactly the stated budget. A first computation (score ≥ the 95th percentile) over-counted
  monitors with tied scores (the tool baseline, a judge's 0/100 answers); the record lists the corrections. The
  reader's numbers did not change.
- A monitor can miss a policy that is not like the policies in its training data (T10). Label a few dozen steps of
  each such policy. A policy that is like a trained one can transfer without labels (T12).
- The sufficiency corollary (Corollary 1 in the paper; Corollary 3 in `docs/theory.md`) applies to action-level labels. It does not apply to labels about later behaviour or about beliefs.
- Chat readouts recover the behaviour of a model, not its hidden beliefs.

## Chat: forecast a refusal before the reply

A *grey box* is the model's **interrogation state** Ψ(x): the probabilities of the answer "Yes" to a fixed list of
questions. We ask the questions in side branches of the conversation, so the real reply never sees them.
- Ψ recovers the hidden state of models with known internals: a program compiled into a transformer (21/21 states with a battery designed for it; 16/21 with a random one),
  Othello-GPT (R² 0.967) and 8 open chat models. L1 and L2 are preregistered; L0 is a development test.
- Readouts trained only on open models transfer to Claude: harmful request 0.993, refusal 0.997 AUROC (L3).
- As a refusal forecast on Claude, Ψ only ties a graded moderator (L3e: 0.87 vs 0.85, not significant). With a
  moderator, it doubles the precision of an alarm for harmful requests that the model answers (0.54 vs 0.27).
- It is not a content filter: a text moderator is better at the decision "is this message harmful?" (0.97 vs
  0.85–0.93).

**Grey Box Live in the browser** (`web/live_browser.html`) chats with Claude Haiku 5.5. It shows Claude's path in a
3D space shared with open models, the forecast "Claude will refuse", and the alarm "a moderator says Unsafe and
Claude will not refuse". Each message uses 4 Claude calls on the viewer's own account.

## Reproduce

| Path | Content |
|---|---|
| `src/graybox/` | the SDK: monitors, gate, batteries, adapters, servers, spend ledger |
| `scripts/` | the experiments, in order; each script's docstring gives its command |
| `docs/PREREGISTRATION_RECORD.md` | every confirmatory test, its lock and its outcome |
| `docs/theory.md` | the propositions and their proofs |
| `docs/research_ledger.md` | a dated log of each experiment, decision, failure and fix |
| `paper/main.tex` | the paper (ICML format); `paper/make_numbers.py` and `paper/make_record.py` write its result macros and its record appendix from the files below |
| `docs/paper_draft.md` | an early draft, kept for the record (superseded by `paper/main.tex`) |
| `results/` | the saved output of each analysis (summary statistics; the per-action data and GPU features are not public) |
| `site/` | the website builder and the browser-vs-Python parity tests |

Checks:
- `python scripts/99_verify_all.py` verifies each lock hash, re-derives each pass/fail from its saved interval, and
  prints the headline numbers. About 20 locked files (frozen models and per-action scores under `/data/models`) are
  not public, so from a clone it lists them as NOT PUBLIC (not as mismatches); the repository files all verify.
- `python scripts/98_audit_numbers.py paper/main.tex README.md` checks that each decimal in a document occurs somewhere
  in the saved results. This is a weak check (most short decimals occur somewhere); the paper's main result numbers
  are written from named result fields by `paper/make_numbers.py`.
- `pytest tests` runs the SDK tests (also in CI).

What you can reproduce from this repository alone:
- the SDK and its tests, the browser–Python parity tests, and the lock hashes of the repository files;
- the headline confirmatory numbers, recomputed from the per-unit release in `results/units/` (labels, strata,
  clusters and each monitor's score per action; the true and replicated states on 128 components; the live
  episodes): `python scripts/109_recompute_public.py` recomputes 67 numbers (AUROCs, catch rates, R², the T16c
  intervals) and checks them against `results/t1/` (CI runs it). `scripts/108_release_units.py` built the release
  and checked each array against the published value.

The experiments themselves (episodes, hidden states, reader features, CAPE-Bench internals) need data that is not
public, GPUs, and paid API calls.

Lock times are self-reported in each lock file. For T13b, T16 and T20 the lock was also published (git) before the
result; from T20 on, every lock is pushed before its test runs.

The experiments ran on a Slurm cluster, one GPU at a time (`slurm/run.sh`). Paid API runs share a file-locked spend
ledger with a hard cap.
