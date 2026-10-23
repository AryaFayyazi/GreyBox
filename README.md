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
  0.061; 7 of 8 inside the interval), estimates that Closed-A's own answers reconstruct its own unseen hidden geometry
  at 0.495 [0.437, 0.554], as well as an open model's do (G34).
- **Monitor.** At a proposed call, a fixed read-out of the agent's answers to 8 policy questions ranks violations
  within template at AUROC 0.82–0.99 on four open agents, close to a probe trained on their hidden states (G15), and
  better than a reasoning judge on open agents (G20). It carries to prompt injection (AgentDojo, 0.90–0.99; G17) and to
  two closed models: Closed-A (0.909 on unapproved restarts, where its single answer scores 0.32; 0.704 on τ²-bench
  airline, G18) and GPT-6-luna (0.979 [0.946, 0.998], where its single answer scores 0.795; G38). With no label, the
  read-out's variance beyond its sampling noise tells where it works (Spearman 0.71 over 29 settings; G32).
- **Forecast.** Before its call exists, GPT-6-luna's answers about its next action rank the coming violation at 0.765
  [0.650, 0.909] (G38), and at 0.927 among calls at the same position (exploratory; G40 confirms it on fresh episodes).
- **Prevent.** As a gate, the agent's own answer prevented 82% of an open agent's violating episodes (G9) and 69% of
  those that Closed-A's transcript decides (G10), by amounts a law predicted in advance for five agents (mean error
  0.035; G12); against prompt injection it cut attack success by 84% and 50% (G13). G39 tests a live gate on a closed
  agent's pre-call state against a gate on position.
- **Comparisons, not the method.** Another model answering the same questions does about as well on open agents
  (G16), and with reasoning on it reaches 0.989 on Closed-A's τ²-bench writes (G21): the questions read the geometry
  that models share. The method itself uses only the agent's own API.

Trained monitors (an open reader with a probe trained on labelled steps) are also in the package, as an option and as
the comparison in the paper; they need labels and do not transfer to new kinds of task without them.

**Website: https://aryafayyazi.github.io/GreyBox/**
- Train a monitor on your own labelled agent steps, in your browser. The browser does not upload your data.
- See a monitor on a closed frontier agent (Closed-A), action by action, in 3D.
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
   [150 Closed-A conversations](https://aryafayyazi.github.io/GreyBox/data/example_closed_a.jsonl).
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
the same steps as our research pipeline (979/979 GPT and 373/373 Closed-A units identical; `results/tool/`).

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
| T8b: a closed frontier agent | Closed-A, 3,128 actions. Trained only on Gemma, the reader catches 94.1% of the 68 violations when the top 5% of actions go to review, with a cut set without labels (95.6% with a cut set on labelled safe actions). The judge catches 0%. Closed-A ran with thinking off. Pooled AUROC 0.989 (judge 0.570; exploratory). The preregistered within-template AUROC (0.991 vs 0.633) uses only one stratum: 136 actions, 26 violations. A prior that knows only each template's and tool's violation rate (fit out of fold) does as well on Closed-A: AUROC 0.992, 91.6% caught (exploratory; 42 of the 68 violations are in a template where every call violates). On GPT-6-luna and Gemma-4 the reader beats that prior. |
| T9: reader or labels? | A TF-IDF monitor with the same labels is equal to the reader on the training benchmark (Closed-A 0.997 vs 0.991). On unseen task families the reader is better (0.969 vs 0.950). On AgentDojo, TF-IDF is at chance (0.50 vs 0.78). |
| T10: a policy that the monitor never saw | The reader still ranks the violations well (pooled 0.912; judge 0.570). But at a 5% budget it escalates 0 of 26 violations. With 20 to 40 labelled examples of that policy, it catches 85% to 94% (exploratory). |
| T11: a second closed family (GPT) | GPT-6-luna, 3,627 actions, 66 violations in 6 families. Pooled AUROC 0.996 (judge 0.778; tool identity alone 0.856). Within tool 0.994 (judge 0.956). At a 5% budget it catches 98.5% (judge 37.9%). All three preregistered endpoints pass. |
| T12: labels for a new policy, prospective | We held out a credentials policy. With no labels of it, the reader caught all 18 of GPT's violations of it. The preregistered H1 (labels improve the catch rate) fails: there was nothing to fix. A new policy that resembles a trained one can transfer; one that does not (T10) needs labels. |
| T13, T13b: a closed agent on an external benchmark | GPT-6-luna on AgentDojo. With the spotlighting defense, attacks succeed in 2.1% of episodes (4 injected calls): not tested. Without the defense, 36 injected calls: reader 0.840 vs judge 0.787 (+0.052 [−0.026, +0.130]). The preregistered test fails: on an external benchmark the reader is not significantly better than a zero-shot judge for this closed agent. |
| T14: labels of the new policy, from open agents | We added labelled AgentDojo calls of the two open agents to the training data, and tested on the closed GPT agent. AUROC 0.840 → 0.922 (judge 0.787); injected calls caught at the label-free 5% cut 28% → 58% (36% → 92% with a cut set on benign calls). Both preregistered endpoints pass. Exploratory: on the confirmation split only, whose user tasks the training calls never saw, 0.929 vs judge 0.786 (+0.143 [0.075, 0.227]). On this benchmark, labels of a policy transferred from open agents to a closed one. |
| T15: does the grey box replicate the hidden state? | From the transcript and the proposed action, the reader replica reconstructs 66.4% of the variance of the 128 main components of the agent's decision state (51–54% of each full layer; text features: 58.6%). A probe trained on the true state scores 0.987 on the replica and 0.984 on the true state; on a replica from text features it scores 0.983, so the information that decides a violation is in the tokens. |
| T17: replication on three more open agents | gpt-oss-20b (mixture of experts), Qwen3.8-27B (hybrid linear attention), Muse-Glimmer-30B (dense), on scenarios that the map never saw. The same reader replicates each agent's decision state better than text features: R² 0.49, 0.54, 0.46 (text 0.39, 0.44, 0.40). A probe trained on the true states works on every replica (0.900 → 0.924, 0.955 → 0.972, 0.929 → 0.969). All six endpoints pass. Replication is partial (about half of the variance), and the readout also works on a text replica. |
| T16: live prevention on a closed agent | The monitor in GPT-6-luna's loop (500 scenarios, with and without the gate). Episodes with an executed violation: 6.6% → 0.2% (−0.064 [−0.088, −0.044]). No violation after a block. Goals reached 47.4% → 45.6%. Round trip 89 ms. |
| T16b: live prevention on Closed-A | The same live test on Closed-A (450 scenarios). Episodes with an executed violation: 6.7% → 0.2% (−0.064 [−0.087, −0.044]). No violation after a block. Goals reached 40.7% → 34.9%. Round trip 80 ms. |
| T20: a second outside benchmark (τ²-bench), two closed agents | Monitors trained on four frontier agents' published, labelled τ² trajectories; tested on Closed-A and GPT-6-luna with a persistent customer (19 violations in 262 writes). Within tool: word counts 0.987 (G2 passes), reader 0.738 (G1 fails). But the test tasks also occur in training: see T21. |
| T21: τ²-bench, unseen tasks | T20 with every training write of a test task removed (554 writes of 28 other tasks remain). Within tool: reader 0.535, word counts 0.519; both preregistered endpoints fail (reader − word counts +0.017 [−0.514, +0.539]; reader − tool +0.035 [−0.230, +0.259]). Word counts with entities masked: 0.366. T20's word-count score was memorization of the tasks. |
| T22: a zero-shot reasoning judge on τ²-bench | Qwen3-32B with thinking reads the policy, the conversation and the proposed write; no labels. On T20's 262 writes: within tool 0.804 (T21's reader 0.535). Both preregistered endpoints fail: judge − tool +0.304 [−0.084, +0.427], judge − reader +0.268 [−0.074, +0.511]; only 7 tasks have a violation. |
| T23: the same judge on four closed frontier agents | T22's judge, unchanged, on the 894 published τ² writes of Closed-A 3.7 Sonnet, GPT-4.1, GPT-4.1-mini and o4-mini (227 violations, 29 tasks). Within tool: judge 0.766, reader trained on other tasks 0.646. K1 passes (judge − tool +0.266 [0.151, 0.338]); K2 fails (judge − reader +0.120 [−0.047, +0.274]). With exactly 5% of writes reviewed the reader catches more: 18.4% vs 11.7% (most possible 19.7%). The judge separates violating from allowed writes of the same tool without labels, but as a review queue it is below the tool alone (13.0%). |
| T16c: hold for review instead of block | On Closed-A (450 scenarios), escalated actions go to a reviewer (here an oracle; an upper bound on a human). Episodes with an executed violation: 6.7% → 0.0% (R1 passes). Goals 38.7% (no gate 40.7%, blocking 34.9%); the preregistered "at most 3 points lost" fails (−0.020 [−0.051, +0.007]). Exploratory: episodes in which the reviewer let a held action run kept their goals (60.8% vs 58.8%); the gap is in episodes where nothing was held, which we suggest is variation between runs (untested: the control was not concurrent). 0.20 reviews per episode. |
| T24: a closed agent told it is monitored | Closed-A on 990 scenarios, with and without an oversight notice, with the cut frozen on its earlier traffic. Caught 97.1% [92.9%, 100%] of the told agent's violations and 83.6% of the others (both endpoints pass). The notice moved all scores up: the frozen cut escalated 15.2% of actions instead of 5.3%, so set the cut again after a prompt change. Violation rate unchanged. |
| T28: set the cut again after a prompt change, live | The told agent in its tool loop (400 scenarios × gated / ungated), with the cut set from its earlier unlabelled scores. The new cut escalated 4.7% [3.5%, 6.1%] of actions (the old cut: 16.6%), and the gate cut episodes with an executed violation from 6.5% to 0.0% (both endpoints pass). After any change of prompt, tools or agent: `/v1/monitor/calibrate` on new traffic. |
| T25–T27: an attacker who can query the score | Rewriting only the free text of 134 real violations, with score feedback, cut the reader's catch rate from 96% to 29% (fails). Two preregistered defenses also failed: canonical actions (score cannot move, but clean catch fell to 51–59% because a message body can be the leaked content) and training on rewrites of both classes (60% caught after a fresh, defense-aware attack). The monitor is not robust to adaptive rewording: do not show the score to the agent. |
| Review threshold without labels (exploratory) | A threshold from the agent's own unlabelled traffic, frozen after 500 actions: 95.4% caught online, 1.9% of safe actions escalated. With a burst of violations, it stays at 94%; a rolling threshold falls to 36%. |

**Results that did not confirm** are in the record too. Examples: the agent's answers alone do not match the
white-box probe (T1, F3), and resampling scored by an LLM judge loses to the probe (T3).

## Limits

- Most agent tests use one benchmark, CAPE-Bench. On the external test (AgentDojo), the reader beats a zero-shot judge
  for open agents (0.78 vs 0.54), but not significantly for a closed agent (0.840 vs 0.787, T13b).
- On τ²-bench (T20: Closed-A and GPT-6-luna with a customer who insists after a refusal; 262 writes, 19
  violations), monitors trained on four frontier agents' labelled trajectories: word counts reached within-tool AUROC
  0.987 (preregistered G2 passes), the reader only 0.738 (G1 fails; reader − word counts −0.250 [−0.518, −0.077]).
  But every test task also occurs in training. With those tasks removed (T21), neither monitor transfers: reader 0.535,
  word counts 0.519. T20's word-count score was memorization of tasks (codes, names, dates). τ² violations break
  explicit rules over facts of the case (dates, amounts, cabin); label-trained readouts do not learn these checks for
  new tasks. A zero-shot reasoning judge, which needs no labels, separates violating from allowed writes of the same tool
  (T23: 0.766 on four closed frontier agents), but at a 5% review budget it catches fewer violations than the reader. (T18, on Closed-A
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
- Closed-A, from 8 sampled answers to 24 questions, reveals the same shared geometry. Two independent halves
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
- Closed-A (closed, G10): violations that its conversation reveals fell by 69% [51, 88], at no cost in task
  success; violations decided by a fact it never checked (a mailing list's size) did not fall (7% [0, 17]), as the
  locked prediction said. All its violations: −31%. A plain "does it comply?" question did not work on Closed-A; checking the
  rules one by one did.
- It works when the agent judges its policy well. With the plain question (most answers "no"), agents flag 97%
  (Gemma-4), 87% (Muse-Glimmer), 47% (Qwen3.8), 37% (gpt-oss) and 5% (Closed-A) of their own violations.
- The effect is predictable (G12): each agent's offline flag rates per kind of task, weighted by where it violates,
  predicted the live reduction for five agents within the registered ±0.15 (Gemma-4 0.63, gpt-oss 0.17, Qwen3.8 0.01,
  Muse-Glimmer 0.27, Closed-A 0.30; mean error 0.035). Measure τ on enough calls of each kind (Closed-A was over-predicted
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
- Readouts trained only on open models transfer to Closed-A: harmful request 0.993, refusal 0.997 AUROC (L3).
- As a refusal forecast on Closed-A, Ψ only ties a graded moderator (L3e: 0.87 vs 0.85, not significant). With a
  moderator, it doubles the precision of an alarm for harmful requests that the model answers (0.54 vs 0.27).
- It is not a content filter: a text moderator is better at the decision "is this message harmful?" (0.97 vs
  0.85–0.93).

**Grey Box Live in the browser** (`web/live_browser.html`) chats with Closed-A. It shows Closed-A's path in a
3D space shared with open models, the forecast "Closed-A will refuse", and the alarm "a moderator says Unsafe and
Closed-A will not refuse". Each message uses 4 Closed-A calls on the viewer's own account.

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
