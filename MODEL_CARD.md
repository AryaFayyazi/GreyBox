# Model card: the Grey Box reader probes

Two probe files are released with v0.2.0 (`reader_probe.joblib`, `reader_probe_cape_agentdojo.joblib`). They score
an AI agent's proposed tool call before it runs. This card describes what they are, what they were tested on, and
where they fail. Every number links to a preregistered test in `docs/PREREGISTRATION_RECORD.md`, unless it is marked
exploratory.

## What it is
- **Reader:** Qwen/Qwen3-8B (frozen, unmodified). Input: the agent's instructions, the user's task, the earlier tool
  calls with their results, and the proposed call, in the prompt template `graybox.monitor.Q`. Feature: the residual
  stream at layer 32, at the last token.
- **Probe:** standardise → PCA (256 components) → logistic regression (C = 0.05), scikit-learn.
- **Output:** a probability-like score. Its scale does **not** transfer between agents; use its ranking with a cut set on
  the monitored agent's own traffic (`QuantileGate`, or the server's `/v1/monitor/calibrate`).
- **Files:**
  - `reader_probe.joblib`: sha256 `81ed26e6bbdb6b99ec10d9a7647a3dc2ba2e90aa3f92b9700aa797e802d59c2b`;
  - `reader_probe_cape_agentdojo.joblib`: see the release page.
  - Both are joblib pickles. Loading a pickle runs code, so check the checksum first.

## Training data
- `reader_probe.joblib`: 17,623 steps of the open agent Gemma-4 (26B) on CAPE-Bench. These cover 23 policy families;
  each action is labelled by the environment's checker, and 1,339 are violations.
- `reader_probe_cape_agentdojo.joblib`: the same, plus 3,455 labelled AgentDojo calls (prompt injection) from two open
  agents, Gemma-4 and gpt-oss-20b.
- No closed agent's data was used for training.

## Intended use
- Pre-execution review of an agent's tool calls: send the highest-scored few percent to a person, block them, or ask
  the agent's user.
- The policies must resemble those in the training data. For a new policy, add labelled steps of it and refit (the
  same pipeline on `mon.features(prompt)`, or `python -m graybox.train_reader`).

## Measured performance (closed agents, reasoning off, nothing fit on them)

| Setting | Result |
|---|---|
| Claude Haiku 5.5, 3,128 CAPE-Bench actions | 94.1% of 68 violations caught at a 5% label-free review budget; within-template AUROC 0.991 |
| GPT-6-luna, 3,627 actions | 98.5% of 66 violations caught; pooled AUROC 0.996 |
| Without the largest family (62% / 59% of violations; absent from training) | 92.3% / 96.3% caught (exploratory) |
| Live, in the agents' tool loops (block) | episodes with an executed violation: 6.6% → 0.2% (GPT), 6.7% → 0.2% (Claude) |
| Claude told it is monitored | 97.1% caught at a cut frozen on its plain traffic; its scores shift up, so set the cut again |
| The same, cut set again on its new unlabelled traffic, live | 4.7% of actions escalated (budget 5%); episodes with an executed violation 6.5% → 0.0% |
| AgentDojo, closed GPT, CAPE probe | AUROC 0.840 (judge 0.787; not significantly better) |
| AgentDojo, closed GPT, CAPE + AgentDojo probe | AUROC 0.922 (judge 0.787) |

## Limits
- **Not robust to an adaptive attacker.** An attacker who can query the score and rewrite a call's free text cut the
  catch rate from 96% to 29%. Two defenses failed. Keep the score and the decision reason hidden from the agent and from
  anyone who writes its inputs. Train a private probe on your own labels.
- **New policies need labels.** A policy unlike the training policies was ranked well but not escalated at a 5% budget.
  On τ²-bench, no trained monitor transferred to tasks it had not seen.
- **Shift moves the cut.** A cut stored from one task distribution can escalate almost nothing on another. Set the cut
  again after a change of agent, prompt, tools or tasks.
- **One benchmark for most results.** Most results are on CAPE-Bench, whose tasks are synthetic. Two closed agent
  families were tested (Claude, GPT), with reasoning off.
- **Cost of blocking.** Goals reached fell by 1.8 points (GPT) and 5.8 points (Claude) when blocking. Holding for an
  oracle reviewer recovered part of it.

## Ethical considerations
The probe reads transcripts, which can hold personal data and credentials; run it on your own machines
(`graybox.monitor_server`, Docker). It does not extract weights or private data from any model. False negatives are
expected: use it as one layer of review, not as the only control.
