# Grey Box

[![ci](https://github.com/AryaFayyazi/GreyBox/actions/workflows/ci.yml/badge.svg)](https://github.com/AryaFayyazi/GreyBox/actions/workflows/ci.yml)

**Reconstruct a closed AI agent's hidden-state geometry through its own API, then monitor, forecast and prevent its
violations from that reconstruction: no other model, no training.**

Probes on an agent's hidden states catch unsafe actions that monitors of its text and chain of thought miss, but they
need the weights. Grey Box measures the state through the API instead. In side branches that the agent never sees, it
asks the agent a few fixed yes/no questions and samples each answer a few times. The questions are measurement
directions, as in state tomography; the model only answers yes or no. In the angles θ = 2 arcsin √P(yes), the
Fisher–Rao distance between answer laws is Euclidean distance, with a sampling noise that the theory fixes. The
distances are the reconstructed geometry of the hidden state; a fixed read-out of them, with no labels and nothing
trained, scores a call before it runs.

Preregistered results (hash-locked, locks published before the data; held-out data; failures reported in the record):
- **Reconstruct.** On eight open models, the answer geometry tracks each model's hidden geometry (G1). Part of that
  geometry is shared across models, and a triangulation through it, checked by leaving each open model out (mean error
  0.061; 7 of 8 inside the interval), estimates that Claude Haiku 5.5's own answers agree with its own unseen hidden geometry at
  0.495 [0.437, 0.554], the level of its agreement with open models' hidden geometries (G34; it assumes no own-model
  excess; over the open loadings' range, [0.32, 0.53]). At an agent's decision, 8 policy answers capture only a narrow
  slice of the decision state (G36).
- **Monitor.** At a proposed call, a fixed read-out of the agent's answers to 8 policy questions ranks violations
  within template at AUROC 0.82–0.99 on four open agents, close to a probe trained on their hidden states (G15), and
  better than a reasoning judge on open agents (G20). It carries to prompt injection (AgentDojo, 0.90–0.99; G17) and to
  two closed models: Claude Haiku 5.5 (0.909 on unapproved restarts, where its single answer scores 0.32; 0.704 on τ²-bench
  airline, G18) and GPT-6-luna (0.979 [0.946, 0.998], where its single answer scores 0.795; G38). With no label, the
  read-out's variance beyond its sampling noise tells where it works (Spearman 0.71 over 29 settings; G32).
- **Forecast.** Before its call exists, GPT-6-luna's answers about its next action rank the coming violation at 0.765
  [0.650, 0.909] (G38), and at 0.923 [0.811, 0.992] among calls at the same position, where position cannot help (G40: preregistered, 252 fresh episodes). Live, at an equal review rate, a gate on that state cut its violating episodes from 31.8% to 7.9% (a 75% reduction, against 44% for a gate on position; difference +0.31 [0.06, 0.54]; success unchanged; G39b), and the offline share of violating calls it holds predicted the 75% exactly (G42). On Gemma-4, whose pre-call read-out ranks weakly, such a gate did not prevent (G29b). Before its τ²-bench writes, Claude Haiku 5.5's pre-call answers did not rank the coming violation (0.536; G37). On three open agents, the pre-call answers rank the coming call among calls at the same step at 0.669 [0.594, 0.764] (G30); the state has a short memory (G31c) and the deciding fact enters at the call's step (G31b), and a forecast from earlier steps beats position on them (+0.21 [0.08, 0.32]; G27b).
- **Prevent.** As a gate, the agent's own answer prevented 82% of an open agent's violating episodes (G9) and 69% of
  those that Claude Haiku 5.5's transcript decides (G10; 31% of all); in a later test, a law fixed in advance predicted
  the prevention of five agents (mean error 0.035; G12); against prompt injection it cut attack success by 84% and 50% (G13). G39 tests a live gate on a closed
  agent's pre-call state against a gate on position.
- **Comparisons, not the method.** Another model answering the same questions does about as well on open agents
  (G16), and with reasoning on it reaches 0.989 on Claude Haiku 5.5's τ²-bench writes (G21): the questions read the geometry
  that models share. The method itself uses only the agent's own API.

Trained monitors (an open reader with a probe trained on labelled steps) are also in the package, as an option and as
the comparison in the paper; they need labels and do not transfer to new kinds of task without them.

**Website: https://aryafayyazi.github.io/GreyBox/**
- Train a monitor on your own labelled agent steps, in your browser. The browser does not upload your data.
- See a monitor on a closed frontier agent (Claude Haiku 5.5), action by action, in 3D.
- Read every preregistered result, with the failures.

**Model card:** [MODEL_CARD.md](MODEL_CARD.md) (training data, measured performance, limits of the released probes).

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
   [150 Claude Haiku 5.5 conversations](https://aryafayyazi.github.io/GreyBox/data/example_closed_a.jsonl).
   ```
   python -m graybox.train_text labelled_logs.jsonl --out greybox-text-monitor.json   # --chat for agents that talk with a user
   ```
   It reports the AUROC with whole tasks held out, pooled and within tool, and the catch rate at the review budget. The
   privacy filter (on unless you pass `--no-privacy`) masks emails and URLs and drops terms with digits, codes or rare words;
   names that recur in your logs stay in the model file.
   `--eval-only --group-by <field>` reports the score with other groups held out (for example task families).
   After a change of agent, prompt, tools or tasks, set the cut again on new, unlabelled logs:
   `python -m graybox.recalibrate new_logs.jsonl --monitor greybox-text-monitor.json --out recalibrated.json`.
   `--canonical` trains on the canonical action: free text that the agent writes about its own call (a message body, a
   subject, a reason) and quoted prose are removed, so rewording them cannot move the score.
   It is off by default because it has a measured cost (preregistered): on two closed agents the canonical reader caught
   51–59% of violations at a 5% budget instead of 94–98%, because a message body can be the leaked content itself. Use it
   only when your policies never depend on what free text says.
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
   In an agent framework, one line places the Guard before every tool call (`graybox.integrations`; install the extra
   of your framework):
   ```python
   ClaudeAgentOptions(hooks={"PreToolUse": [HookMatcher(hooks=[claude_agent_hook(guard, instructions=SYSTEM)])]})
   check, record = openai_agents_guardrails(guard)   # function_tool(..., tool_input_guardrails=[check], tool_output_guardrails=[record])
   ToolNode(tools, wrap_tool_call=langgraph_tool_wrapper(guard))
   ```
3. On a new kind of task, a cut stored in the file can escalate almost nothing: on held-out task families it caught
   0 of 115 violations (three splits of the public GPT example, `scripts/114_cut_under_shift_public.py`). A live cut on
   the agent's own traffic, `Guard(monitor, gate=QuantileGate(0.05))` (or the server's `"gate": true`), keeps the review
   load at 5%, but it caught only 29 of 115 there (1 of 39 in one split): it does not restore detection. Label actions of the new tasks and retrain, or use the
   reader. A new gate sends every action to review until it has seen 100: seed it with `gate.fit(past_scores)`.

4. Agents that call tools through MCP: run the monitor as a proxy in front of the MCP server (`pip install
   "greybox-monitor[mcp]"`), with no change to the agent:
   ```
   python -m graybox.mcp_proxy --monitor greybox-text-monitor.json --instructions policy.txt -- <your MCP server command>
   ```
   The proxy sees only the tool calls, not the conversation (on our GPT example the text monitor lost little from that;
   see the docs). `Guard(..., audit_log="decisions.jsonl")`, or the server's `GB_AUDIT_LOG`, records every decision.

Train the reader monitor on your own labels with one command on a GPU:
`python -m graybox.train_reader labelled.jsonl --out my_reader_probe.joblib`, then load it with
`graybox.reader_load.load_reader`. The released probes are in
[release v0.2.0](https://github.com/AryaFayyazi/GreyBox/releases/tag/v0.2.0): `reader_probe.joblib` (CAPE-Bench) and
`reader_probe_cape_agentdojo.joblib` (CAPE-Bench and prompt injection). For the reader, always use `QuantileGate`: its
probability does not transfer between agents.

The browser trainer imports OpenAI and Anthropic agent logs, shows each action with the conversation before it, lets
you label actions (a review queue shows the riskiest first), and trains the monitor. Its importer and the guard build
the same steps as our research pipeline (979/979 GPT and 373/373 Claude Haiku 5.5 units identical; `results/tool/`).

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
| T8b: a closed frontier agent | Claude Haiku 5.5, 3,128 actions. Trained only on Gemma, the reader catches 94.1% of the 68 violations when the top 5% of actions go to review, with a cut set without labels (95.6% with a cut set on labelled safe actions). The judge catches 0%. Claude Haiku 5.5 ran with thinking off. Pooled AUROC 0.989 (judge 0.570; exploratory). The preregistered within-template AUROC (0.991 vs 0.633) uses only one stratum: 136 actions, 26 violations. A prior that knows only each template's and tool's violation rate (fit out of fold) does as well on Claude Haiku 5.5: AUROC 0.992, 91.6% caught (exploratory; 42 of the 68 violations are in a template where every call violates). On GPT-6-luna and Gemma-4 the reader beats that prior. |
| T9: reader or labels? | A TF-IDF monitor with the same labels is equal to the reader on the training benchmark (Claude Haiku 5.5 0.997 vs 0.991). On unseen task families the reader is better (0.969 vs 0.950). On AgentDojo, TF-IDF is at chance (0.50 vs 0.78). |
| T10: a policy that the monitor never saw | The reader still ranks the violations well (pooled 0.912; judge 0.570). But at a 5% budget it escalates 0 of 26 violations. With 20 to 40 labelled examples of that policy, it catches 85% to 94% (exploratory). |
| T11: a second closed family (GPT) | GPT-6-luna, 3,627 actions, 66 violations in 6 families. Pooled AUROC 0.996 (judge 0.778; tool identity alone 0.856). Within tool 0.994 (judge 0.956). At a 5% budget it catches 98.5% (judge 37.9%). All three preregistered endpoints pass. |
| T12: labels for a new policy, prospective | We held out a credentials policy. With no labels of it, the reader caught all 18 of GPT's violations of it. The preregistered H1 (labels improve the catch rate) fails: there was nothing to fix. A new policy that resembles a trained one can transfer; one that does not (T10) needs labels. |
| T13, T13b: a closed agent on an external benchmark | GPT-6-luna on AgentDojo. With the spotlighting defense, attacks succeed in 2.1% of episodes (4 injected calls): not tested. Without the defense, 36 injected calls: reader 0.840 vs judge 0.787 (+0.052 [−0.026, +0.130]). The preregistered test fails: on an external benchmark the reader is not significantly better than a zero-shot judge for this closed agent. |
| T14: labels of the new policy, from open agents | We added labelled AgentDojo calls of the two open agents to the training data, and tested on the closed GPT agent. AUROC 0.840 → 0.922 (judge 0.787); injected calls caught at the label-free 5% cut 28% → 58% (36% → 92% with a cut set on benign calls). Both preregistered endpoints pass. Exploratory: on the confirmation split only, whose user tasks the training calls never saw, 0.929 vs judge 0.786 (+0.143 [0.075, 0.227]). On this benchmark, labels of a policy transferred from open agents to a closed one. |
| T15: does the grey box replicate the hidden state? | From the transcript and the proposed action, the reader replica reconstructs 66.4% of the variance of the 128 main components of the agent's decision state (51–54% of each full layer; text features: 58.6%). A probe trained on the true state scores 0.987 on the replica and 0.984 on the true state; on a replica from text features it scores 0.983, so the information that decides a violation is in the tokens. |
| T17: replication on three more open agents | gpt-oss-20b (mixture of experts), Qwen3.8-27B (hybrid linear attention), Muse-Glimmer-30B (dense), on scenarios that the map never saw. The same reader replicates each agent's decision state better than text features: R² 0.49, 0.54, 0.46 (text 0.39, 0.44, 0.40). A probe trained on the true states works on every replica (0.900 → 0.924, 0.955 → 0.972, 0.929 → 0.969). All six endpoints pass. Replication is partial (about half of the variance), and the readout also works on a text replica. |
| T16: live prevention on a closed agent | The monitor in GPT-6-luna's loop (500 scenarios, with and without the gate). Episodes with an executed violation: 6.6% → 0.2% (−0.064 [−0.088, −0.044]). No violation after a block. Goals reached 47.4% → 45.6%. Round trip 89 ms. |
| T16b: live prevention on Claude Haiku 5.5 | The same live test on Claude Haiku 5.5 (450 scenarios). Episodes with an executed violation: 6.7% → 0.2% (−0.064 [−0.087, −0.044]). No violation after a block. Goals reached 40.7% → 34.9%. Round trip 80 ms. |
| T20: a second outside benchmark (τ²-bench), two closed agents | Monitors trained on four frontier agents' published, labelled τ² trajectories; tested on Claude Haiku 5.5 and GPT-6-luna with a persistent customer (19 violations in 262 writes). Within tool: word counts 0.987 (G2 passes), reader 0.738 (G1 fails). But the test tasks also occur in training: see T21. |
| T21: τ²-bench, unseen tasks | T20 with every training write of a test task removed (554 writes of 28 other tasks remain). Within tool: reader 0.535, word counts 0.519; both preregistered endpoints fail (reader − word counts +0.017 [−0.514, +0.539]; reader − tool +0.035 [−0.230, +0.259]). Word counts with entities masked: 0.366. T20's word-count score was memorization of the tasks. |
| T22: a zero-shot reasoning judge on τ²-bench | Qwen3-32B with thinking reads the policy, the conversation and the proposed write; no labels. On T20's 262 writes: within tool 0.804 (T21's reader 0.535). Both preregistered endpoints fail: judge − tool +0.304 [−0.084, +0.427], judge − reader +0.268 [−0.074, +0.511]; only 7 tasks have a violation. |
| T23: the same judge on four closed frontier agents | T22's judge, unchanged, on the 894 published τ² writes of Claude Haiku 5.5 3.7 Sonnet, GPT-4.1, GPT-4.1-mini and o4-mini (227 violations, 29 tasks). Within tool: judge 0.766, reader trained on other tasks 0.646. K1 passes (judge − tool +0.266 [0.151, 0.338]); K2 fails (judge − reader +0.120 [−0.047, +0.274]). With exactly 5% of writes reviewed the reader catches more: 18.4% vs 11.7% (most possible 19.7%). The judge separates violating from allowed writes of the same tool without labels, but as a review queue it is below the tool alone (13.0%). |
| T16c: hold for review instead of block | On Claude Haiku 5.5 (450 scenarios), escalated actions go to a reviewer (here an oracle; an upper bound on a human). Episodes with an executed violation: 6.7% → 0.0% (R1 passes). Goals 38.7% (no gate 40.7%, blocking 34.9%); the preregistered "at most 3 points lost" fails (−0.020 [−0.051, +0.007]). Exploratory: episodes in which the reviewer let a held action run kept their goals (60.8% vs 58.8%); the gap is in episodes where nothing was held, which we suggest is variation between runs (untested: the control was not concurrent). 0.20 reviews per episode. |
| T24: a closed agent told it is monitored | Claude Haiku 5.5 on 990 scenarios, with and without an oversight notice, with the cut frozen on its earlier traffic. Caught 97.1% [92.9%, 100%] of the told agent's violations and 83.6% of the others (both endpoints pass). The notice moved all scores up: the frozen cut escalated 15.2% of actions instead of 5.3%, so set the cut again after a prompt change. Violation rate unchanged. |
| T28: set the cut again after a prompt change, live | The told agent in its tool loop (400 scenarios × gated / ungated), with the cut set from its earlier unlabelled scores. The new cut escalated 4.7% [3.5%, 6.1%] of actions (the old cut: 16.6%), and the gate cut episodes with an executed violation from 6.5% to 0.0% (both endpoints pass). After any change of prompt, tools or agent: `/v1/monitor/calibrate` on new traffic. |
| T25–T27: an attacker who can query the score | Rewriting only the free text of 134 real violations, with score feedback, cut the reader's catch rate from 96% to 29% (fails). Two preregistered defenses also failed: canonical actions (score cannot move, but clean catch fell to 51–59% because a message body can be the leaked content) and training on rewrites of both classes (60% caught after a fresh, defense-aware attack). The monitor is not robust to adaptive rewording: do not show the score to the agent. |
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
  (T23: 0.766 on four closed frontier agents), but at a 5% review budget it catches fewer violations than the reader. (T18, on Claude Haiku 5.5
  alone, had too few violations to test.)
- Catch rates are at exactly the stated budget. A first computation (score ≥ the 95th percentile) over-counted
  monitors with tied scores (the tool baseline, a judge's 0/100 answers); the record lists the corrections. The
  reader's numbers did not change.
- A monitor can miss a policy that is not like the policies in its training data (T10). Label a few dozen steps of
  each such policy. A policy that is like a trained one can transfer without labels (T12).
- The sufficiency corollary (Corollary 1 in the paper; Corollary 3 in `docs/theory.md`) applies to action-level labels. It does not apply to labels about later behaviour or about beliefs.
- Chat readouts recover the behaviour of a model, not its hidden beliefs.

## What a closed model's answers reveal: a shared latent geometry

Hidden coordinates cannot be read from outputs: an orthogonal change of the hidden space gives the same model (the
gauge symmetry). A **geometry** can be read, with no trained model.
- Ask yes/no questions after each context, each in its own call. In the angles ϑ = 2 arcsin √P(yes), the Fisher–Rao
  distance between two contexts' answer laws is Euclidean.
- From sampled answers on any chat API, the cross-fit of two sample halves estimates it without bias.

Preregistered results (each lock was published before its data; `docs/PREREGISTRATION_RECORD.md`):
- 8 open models: the answer geometry tracks the hidden geometry (Spearman 0.41–0.74), far better than the input text
  (0.22–0.30) or the model's own replies (0.22–0.44). More replies do not help; more questions do.
- **It is the geometry that models share, not the model's own.** On 1,000 fresh contexts, a model's answers match other
  models' hidden geometries as well as its own (own-model excess +0.002). Another open model's hidden state is the
  better stand-in for 6 of 7 models. This holds on new kinds of context (news, code, jailbreaks, prompt injections).
  Picozzi (2026) found the same sharing for next-token distributions.
- Claude Haiku 5.5, from 8 sampled answers to 24 questions, reveals the same shared geometry. Two independent halves
  of the samples agree at 0.95.
- The model's *own* final geometry needs full log-probabilities. Then it is exact: cosine error 7e-6 on Qwen3-0.6B
  above d(d+1)/2 contexts (the construction of Carlini et al. 2024).
- Why it matters: it is a measured reason why an open model can stand in for a closed one, which the monitors above use.
- Limits: agreement is lower on fresh contexts (0.22–0.64) and much weaker within one kind of context. Replies detect
  a *change* of state sooner than answers.

`pip install "greybox-monitor[geometry]"`, then `graybox-geometry contexts.txt --provider openai --model MODEL --out g.json`, or:
```python
from graybox.geometry import interrogate, crossfit_distances, mds
(kA, nA), (kB, nB) = [interrogate(ask, contexts, questions, samples=4) for _ in range(2)]   # ask(context, question) -> text
D = crossfit_distances(kA, nA, kB, nB)     # observable distances between contexts
X, eigenvalues = mds(D, 2)                  # coordinates, up to a rigid motion
```
Theory and proofs: `docs/theory_infogeom.md` (with Correction 1, after the lock).

## Self-policing: the agent's own API as the monitor (no other model, no labels)

Before a proposed tool call runs, ask the agent itself, in a side branch that its conversation never sees, whether
the call complies with its policy. Hold a flagged call for review. Try it in your browser:
https://aryafayyazi.github.io/GreyBox/selfpolice.html



```python
from graybox.selfaudit import SelfAudit, openai_asker, flag_rates
audit = SelfAudit(openai_asker(client, model="my-agent", tools=TOOLS), prompt="checklist", n=3,
                  reads={"read_file", "search"})          # your agent's read-only tools are not audited
r = audit.check(messages, "send_email", {"to": "x@partner.com", "body": "..."})
if r.flagged:                                          # unparsable answers also hold the call (unparsable="hold")
    hold_for_review(r)
```

Training-free observable state at the decision (8 questions, fixed read-out; the G15 test):

```python
from graybox.selfaudit import observable_state
st = observable_state(ask, messages, "send_email", {"to": "x@partner.com"}, n=4)   # ask: an openai_/anthropic_asker
st.theta, st.score        # 8 arcsine coordinates; higher score = more likely a violation (hold above your traffic's 95th pct)
```

When the user turn is a third party's (a customer), say in the system prompt where the check comes from, and read the
first line of each answer (the G18 settings):

```python
from graybox.selfaudit import TAU2_AIRLINE_QUESTIONS, parse_first_line
st = observable_state(ask, messages, "cancel_reservation", {"reservation_id": "GV1N64"}, n=4,
                      questions=TAU2_AIRLINE_QUESTIONS, parse=parse_first_line, operator_note=True)
```

- `messages` is the conversation **before** the assistant turn that proposes the call, in your provider's own format
  (for `anthropic_asker`, a "system" message is moved to Anthropic's `system` field). The question is appended.
- "checklist" asks the agent to check each numbered rule of its policy **and each constraint of the user's task**, so a
  call that breaks the task (e.g. emails the wrong person) is flagged too. It takes about 10 s per audit; "ask" about 1 s.
- `cut=1.0` (default): hold when every readable answer says "no". With `n=3`, `cut=0.67` also means "all" (2/3 < 0.67).
- Measure your agent first: audit some labelled calls and run `flag_rates(results, violates)` for τ (violations
  flagged) and φ (allowed calls flagged). `prevention_law(R, tau, phi, M)` gives the expected effect.

Preregistered results (hash-locked before the data, locks published; held-out scenarios):
- Gemma-4 (open, G9): episodes with an executed violation 27.9% → 5.0%; task success 33.5% → 43.7%.
- Claude Haiku 5.5 (closed, G10): violations that its conversation reveals fell by 69% [51, 88], at no cost in task
  success; violations decided by a fact it never checked (a mailing list's size) did not fall (7% [0, 17]), as the
  locked prediction said. All its violations: −31%. A plain "does it comply?" question did not work on Claude Haiku 5.5; checking the
  rules one by one did.
- It works when the agent judges its policy well. With the plain question (most answers "no"), agents flag 97%
  (Gemma-4), 87% (Muse-Glimmer), 47% (Qwen3.8), 37% (gpt-oss) and 5% (Claude Haiku 5.5) of their own violations.
- The effect is predictable (G12): each agent's offline flag rates per kind of task, weighted by where it violates,
  predicted the live reduction for five agents within the registered ±0.15 (Gemma-4 0.63, gpt-oss 0.17, Qwen3.8 0.01,
  Muse-Glimmer 0.27, Claude Haiku 5.5 0.30; mean error 0.035). Measure τ on enough calls of each kind (Claude Haiku 5.5 was over-predicted
  by 0.08 because one rate rested on 38 calls).
- A second benchmark (G13, prompt injection on AgentDojo, predictions fixed in advance): attack success 11.5% → 1.8%
  (Gemma-4) and 24.4% → 12.2% (gpt-oss; predicted reduction 0.58, observed 0.50), with no loss of utility. Gemma-4
  missed its predicted 1.00 by 0.16, just beyond the tolerance: most of its remaining attacks ran calls that its audit
  had flagged but our stand-in reviewer let through. Hold every flagged call for a person.

## Chat: forecast a refusal before the reply

A *grey box* is the model's **interrogation state** Ψ(x): the probabilities of the answer "Yes" to a fixed list of
questions. We ask the questions in side branches of the conversation, so the real reply never sees them.
- Ψ recovers the hidden state of models with known internals: a program compiled into a transformer (21/21 states with a battery designed for it; 16/21 with a random one),
  Othello-GPT (R² 0.967) and 8 open chat models. L1 and L2 are preregistered; L0 is a development test.
- Readouts trained only on open models transfer to Claude Haiku 5.5: harmful request 0.993, refusal 0.997 AUROC (L3).
- As a refusal forecast on Claude Haiku 5.5, Ψ only ties a graded moderator (L3e: 0.87 vs 0.85, not significant). With a
  moderator, it doubles the precision of an alarm for harmful requests that the model answers (0.54 vs 0.27).
- It is not a content filter: a text moderator is better at the decision "is this message harmful?" (0.97 vs
  0.85–0.93).

**Grey Box Live in the browser** (`web/live_browser.html`) chats with Claude Haiku 5.5. It shows Claude Haiku 5.5's path in a
3D space shared with open models, the forecast "Claude Haiku 5.5 will refuse", and the alarm "a moderator says Unsafe and
Claude Haiku 5.5 will not refuse". Each message uses 4 Claude Haiku 5.5 calls on the viewer's own account.

## Full preregistration record (every test, including failures)

We hash-locked every confirmatory test before its data existed. Each lock is a file `results/prereg_<name>_lock.json`
with the SHA-256 of the preregistration and its code. From T20 on, we pushed each lock to GitHub before its test ran,
so the commit history shows the order. We apply no correction for multiple tests. Each test answers its own registered
question. The paper reports the tests behind its claims. This section lists every test.

Tally (same rule as `paper/make_record.py`, on the current record):
- Method tests (G): 49 tests. 18 passed every primary endpoint, 17 passed some endpoints and failed others, 9 failed,
  2 stopped before any outcome, 1 reported without a pass/fail bar, 2 pending (G43, G45).
- Trained-comparison tests (T): 32 tests. 15 passed every primary endpoint, 7 passed some endpoints and failed others,
  7 failed, 0 stopped, 3 reported without a pass/fail bar (T8, T13, T18: not tested), 0 pending.

The rule reads words in the outcome text. "No detectable difference" counts as a failure. The seven chat tests
(L1–L3e) are not in the tally. The paper's appendix (`paper/appendix_record.tex`) was written before the G45 row, so it
gives 48 G tests and 1 pending.

### Tests that failed or were mixed

One row per test with a failed endpoint, a stop, or no test (too few cases). Intervals are 95%.

| Test | Question | What failed | What we learned |
|---|---|---|---|
| L3e | Does the Live tool's alarm forecast Claude Haiku 5.5's refusals, prospectively? | A2: grey box 0.873 vs graded Qwen3Guard 0.855, +0.018 [−0.023, 0.062]. A3: multi-turn 0.894 vs 0.906. | The grey box only ties a graded moderator; it adds precision in the alarm (A1: 0.535 vs 0.274). |
| T1 | Do the agent's answers (Ψ) match a white-box probe? | F3: Ψ − WB −0.021 [−0.054, 0.009]. F5: R2+Ψ − R16 −0.023 [−0.053, 0.006]. | The answers to 64 general questions at the start of the turn do not match a probe; resampling does (F1, F2 pass). |
| T3 | Can an LLM judge replace the oracle checker in resampling? | K1: judge-scored resampling 0.782 vs probe 0.846, −0.065 [−0.115, −0.019]. K3: resampling hurts the action judge, −0.049. | The label function is the bottleneck. |
| T8 | Does a Gemma-trained reader monitor Claude Haiku 5.5's actions after a reveal? | Not tested: 0 unsafe units under the post-reveal rule. | Claude violates at or without the reveal; T8b (all actions) replaced it and passed. |
| T9 | Does the reader beat TF-IDF trained on the same labels? | B1 (Claude): reader 0.991 vs TF-IDF 0.997, −0.006 [−0.027, +0.007]. | TF-IDF equals the reader in distribution; the reader is better on unseen families (B2 +0.019 [0.007, 0.032]). |
| T12 | How many labels does a new policy need (GPT)? | H1: 40 − 0 labels: 0.0. With 0 labels the reader caught 18/18. | This policy needed no labels, so labels had nothing to fix. |
| T13 | Reader vs judge for GPT-6-luna on AgentDojo with spotlighting | Not tested: attack success 2.1%, 4 injected calls (minimum 10). | The defense left too few attacks; T13b ran without it. |
| T13b | The same, without a defense | K1b: reader 0.840 vs judge 0.787, +0.052 [−0.026, +0.130]. | Labels of open agents fix it (T14 passes: 0.922). |
| T16c | Hold for review instead of block: is there a goal cost? | R2: goals 40.7% → 38.7%, −0.020 [−0.051, +0.007]; the lower bound misses the −0.03 margin. | Safety holds (R1: 6.7% → 0.0%); "no goal cost" is not shown. |
| T18 | Claude Haiku 5.5 on τ²-bench airline | Not tested: 9 violations, below the minimum of 15. | Too few violations; T20 used two agents and a persistent customer. |
| T19 | Do readers small enough for a browser or a CPU transfer? | F2: browser reader − 8B reader −0.097 [−0.115, −0.079], margin −0.05. | We offer the browser reader for new kinds of tasks only, and no CPU option. |
| T20 | τ²-bench, two closed agents, a persistent customer | G1: reader − tool +0.238 [−0.053, 0.431] (reader 0.738). Reported: reader − TF-IDF −0.250 [−0.518, −0.077]. | The test tasks were also in training; see T21. |
| T21 | τ²-bench, test tasks removed from training | H1: reader − TF-IDF +0.017 [−0.514, 0.539]. H2: reader − tool +0.035 [−0.230, 0.259]. | Neither monitor transfers to unseen tasks; T20's 0.987 was memorisation of the tasks. |
| T22 | A zero-shot reasoning judge (Qwen3-32B) on τ²-bench | J1: judge − tool +0.304 [−0.084, 0.427]. J2: judge − reader +0.268 [−0.074, 0.511]. | Large point estimate (0.804), but only 7 tasks have a violation. |
| T23 | The same judge on four frontier agents | K2: judge − reader +0.120 [−0.047, 0.274]. | The judge separates violating from allowed writes of one tool (K1 passes), but at 5% review it catches 11.7% vs the reader's 18.4%. |
| T25 | An adaptive attacker who can query the score | B1: reader catch 96.3% → 29.1% [21.6%, 36.6%]. B2: reader − TF-IDF −0.052 [−0.149, +0.037]. | The monitor is not robust to adaptive rewording; do not show the score to the agent. |
| T26 | A monitor that reads the canonical action | C1: 50.8% [42.5%, 59.0%] caught after attack. C2: −0.410 [−0.493, −0.311]. C3: −0.029 [−0.080, +0.010]. | The score cannot move, but clean catch falls; canonical mode is an option, not the default. |
| T27 | Adversarial training against the attack | D1: 59.7% [51.5%, 67.9%] caught after a defense-aware attack. D2: −0.037 [−0.078, +0.016]. D3: +0.012 [−0.050, +0.071]. | Hardening added about 2 points; defense against score-querying rewording is an open problem. |
| G8 | Does reading an agent's reasoning help a trained monitor? | C1: in-agent cot − act −0.005 [−0.018, +0.013], margin +0.01. C4 inconclusive: +0.025 [−0.033, +0.085]. | The reasoning gives no in-agent gain and hurts transfer (C2: −0.033 [−0.046, −0.017]). |
| G9 | Can an open agent police itself live through its own API? | L2: resampling needed 0.404 reviews per episode, against the 0.40 limit. | The gate works (27.9% → 4.95%); resampling needs slightly too many reviews. |
| G11 | Does an agent's offline self-audit predict its live self-policing? | Stopped: after 300 Qwen3.8 episodes neither arm had a violation. | Check base rates first; G11b replaced it. |
| G11b | The same, on templates where each agent violates | P1: Muse-Glimmer 0.23 [0.13, 0.35] vs 0.722; Qwen3.8 −0.06 [−0.23, 0.09] vs 0.167. P3: Qwen3.8 success −0.016 [−0.048, +0.016]. | The miss came from the template mix of the offline sample; the composition-weighted law (G12) then passed. |
| G13 | Does self-policing stop prompt injection on AgentDojo by the predicted amount? | A2 (Gemma-4): 0.843 vs predicted 1.00; miss 0.157 > 0.15. | Most remaining attacks ran calls that the audit flagged but the stand-in reviewer let run. |
| G16 | Does the monitor need the agent's own answers? | O1 (two-sided): own − other +0.012 [−0.018, 0.040]. | Any capable model can answer the questionnaire about the call; the title drops "own". |
| G19 | Does a closed agent's answer need its own conversation? | O1 (two-sided): own − outside reviewer +0.043 [−0.136, 0.226]. | The value is in the questions and the read-out, not in "own". |
| G23 | Does the theory predict, without labels, which questions monitor (Gemma-4 answering)? | F2: SNR-weighted − equal +0.008 [−0.177, 0.032]. F3: − raw-variance weights +0.005 [−0.177, 0.023]. No detectable difference. | F1 passes (0.745), but raw P(yes) variance does as well as the SNR. |
| G24 | Do the closed agent's own answers monitor it when it reasons? | E1: 0.568 [0.454, 0.771]. E2: on − off −0.051 [−0.246, 0.150]. | Reasoning does not help detectably; only 70% of answers parsed. |
| G27 | Does the state from earlier steps (Kalman filter) forecast the next call's violation? | K1: forecast − prior +0.013 [−0.115, 0.155]. K2: filtered − raw −0.022 [−0.073, 0.030]. K3: model misspecified (NIS/K 0.52). | The linear dynamics are unstable, and the step index alone scores 0.860; earlier steps do not forecast the call. |
| G27b | G27 replicated on three more open agents | K2: filtered − raw +0.018 [−0.021, 0.062]. | The forecast beats position (K1 +0.209 [0.080, 0.317]), but the filter adds nothing over raw answers. |
| G29 | Live prevention by the estimated state, before the call exists | Stopped before any episode outcome (only the cut, 4.796, was printed). | G27 showed unstable dynamics and a position confound; G29b replaced it. |
| G29b | A pre-call gate against a position gate and a random gate (Gemma-4) | P1: relative reduction −0.04 [−0.30, 0.16]. P3: pre − position −0.24 [−0.50, −0.03]. | Gemma-4's pre-call read-out ranks too weakly to prevent; the position gate reviewed 32% of calls. |
| G30 | Does the state before a call carry its violation beyond position, on three more agents? | W2: same-step agreement with the true state 0.072 [−0.004, 0.144]. | W1 passes (0.669 [0.594, 0.764]); tracking of the true state within a step is not shown. |
| G31 | Are the within-step effects limited by the measurement? | M2: D16 − R16 +0.006 [−0.064, 0.066]. M3: D16 − R4 +0.053 [−0.014, 0.129]. | More answers help (M1 passes); on open agents, dither adds nothing over repeats. |
| G31b | Does the state before the first call warn of a later violation? | E1: early-warning AUROC 0.498 [0.405, 0.585]. | No early warning; the deciding information enters at the call's step (H1 +0.163 [0.029, 0.302]). |
| G33 | The closed agent's own answers, dithered, on τ²-bench | Q1: D16 − G18's read-out +0.015 [−0.089, 0.104]. Q2: D16 − R16 +0.034 [−0.088, 0.157]. | Dither changed the measurement but not the ranking; a positive SNR does not imply a better ranking. |
| G35 | Does the monitor come from the reconstructed state? | L1: 0.044 [−0.008, 0.115]. L2: −0.011 [−0.058, 0.027]. | 20% [13%, 27%] of the read-out's power runs through the hidden violation coordinate (L3 passes); within a cell the answers are nearly unanimous. |
| G35b | The same with dithered answers | K1: −0.077 [−0.250, 0.195]. K2: D16 − R4 −0.041 [−0.274, 0.201]. | Not shown even with dither; the cells are small. |
| G36 | Do an agent's own answers at its calls reconstruct its decision-state geometry? | V2: inside the predicted interval for 1 of 4 agents (bar: 3 of 4). | Eight policy questions measure a narrow slice of the decision state; V1 and E1 pass. |
| G37 | The closed agent's state before its call, on τ²-bench | P1: 0.536 [0.434, 0.682]. P2: D4 − R4 +0.127 [−0.045, 0.253]. | Before its write exists, Claude Haiku 5.5's answers do not rank the coming violation. Run under a recorded deviation. |
| G39 | A live pre-call gate on GPT-6-luna against a position gate | P3: pre − position +0.227 [−0.190, 0.625]. P2: success +0.012 [−0.083, 0.107]; lower bound below −0.05. | P1 passes (26.2% → 9.5%); the gates had unequal review rates, so G39b matched them and passed. |
| G41 | Read the closed agent's state along the directions its call can violate | R1: relevant − full −0.014 [−0.059, 0.038]. R2: +0.008 [−0.021, 0.026]. | Answers from the agent's own seat stay near 0.70; the same model as an outside reviewer with reasoning reaches 0.918 (G26). |
| G42 | Does the prevention law predict a live pre-call gate? | L2 (position gate): predicted 0.167, observed 0.438 [0.256, 0.605]; error 0.271. | The law predicts the pre-call gate exactly (L1: 0.750 vs 0.750), but not a gate on turn index. |
| G44 | Does an agent's own state forecast its next violation better than another model? | F2 (three open agents): own − other −0.030 [−0.159, 0.082]. | It holds for GPT-6-luna (F1: +0.111 [0.041, 0.185]), not for the open agents pooled. |
| G45 | On two closed agents and fresh episodes, does the agent's own pre-call state forecast its next violation better than another model in its seat? | S1: own − other within template\|tool, pooled −0.190 [−0.238, −0.142] (GPT-6-luna 0.762 vs 0.759; Claude Haiku 5.5 0.360 vs 0.742); S3 within position −0.032 [−0.100, 0.038]. S2 passed (own within position 0.649 [0.570, 0.705]; GPT-6-luna 0.836). | G44's single-agent advantage (+0.111) did not replicate. Another model reads GPT-6-luna's next move as well as GPT-6-luna does; Claude Haiku 5.5's answers about its own next action point the wrong way. |
| G46 | Does the agent's own policy, read by sampling, forecast better than another model sampling in its seat (within template\|tool)? | R2: own − other −0.144 [−0.195, −0.094] (Gemma-4's draws 0.848 within template\|tool, 0.530 within position; own 0.704 / 0.708). R1 and R3 passed. | Pooled over steps, another model's draws track the step at which Claude violates; among calls at the same step only the agent's own draws forecast. |
| G46b | Does the theory predict, in size, how much another model loses? | B3: observed − predicted −0.046 [−0.062, −0.032] (observed 0.056, predicted 0.102). B2 passed (policy gap 0.095 [0.083, 0.107]). | The own draws were miscalibrated (0.173 vs 0.210) because G46's rebuilt histories drop refused extra tool calls; G46c repeats with the exact context. |
| G49 | Does a reminder timed by the agent's own policy forecast prevent more than the same reminder at random turns? | F2: forecast − random +0.048 [−0.012, 0.106] (violating episodes 15.0% vs 18.3%; reminders per turn 0.31 vs 0.39). F1 (0.778 reduction) and F3 (success +0.266) passed. | On Claude Haiku 5.5 a short policy reminder prevents most violations by itself; the forecast's timing saves reminders but does not separate from random at this size. |
| G44b | Is the agent's advantage specific to forecasting? | D1: −0.045 [−0.178, 0.082]. | It holds for the closed agent and Qwen3.8, not for gpt-oss-20b or Muse-Glimmer. |

44 rows (1 L, 17 T, 26 G). Two G tests count as mixed by the word rule, but every primary endpoint passed. G17's text
reports server errors. G22's "no detectable difference" is a reported, not a primary, comparison. G20 counts as "no
bar", but its J1 passed.

Other items for the record:
- A passed test with a limit: T10 passed both endpoints (0.912 vs 0.570), but at a 5% budget without labels of the
  policy it caught 0 of 26 violations of the unseen family.
- A design we dropped on development data, before any lock: for a hidden prompt change, eight sampled replies had more
  power than eight sampled answers in every model and condition.
- An early look at partial G45 Claude labels (53 of 328 episodes), while we drafted G46. G45's locked analysis is
  unchanged; we dropped G46 before any lock: [docs/incident_g45_early_look_2026-10-23.md](docs/incident_g45_early_look_2026-10-23.md).
- Analysis waiters fired before their data existed. G31c's analysis ran on one agent only; that file is kept as
  partial and is not G31c's result: [docs/incident_waiters_2026-10-22.md](docs/incident_waiters_2026-10-22.md).
- Deviations, each recorded before any result of its test:
  [G25](docs/deviation_G25_2026-10-22.md) (a crash on an empty read-out slice; endpoints unchanged),
  [G35](docs/deviation_G35_2026-10-22.md) (one gpt-oss-20b call without a valid state dropped),
  [G37](docs/deviation_G37_2026-10-22.md) (fixed credit; the dither arm ran on fewer writes).
  Other deviations, post-lock code edits and corrections are in the Notes of the record.
- Design order: several tests were designed after earlier results (the later τ²-bench tests, the reasoning judge, the
  canonical form, adversarial training). Each was locked before its own data existed, but their hypotheses were not
  independent of what we had seen.

The full table of all tests, passed and failed, with lock files: [docs/PREREGISTRATION_RECORD.md](docs/PREREGISTRATION_RECORD.md).

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
