# Grey Box

[![ci](https://github.com/AryaFayyazi/GreyBox/actions/workflows/ci.yml/badge.svg)](https://github.com/AryaFayyazi/GreyBox/actions/workflows/ci.yml)

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
pip install "graybox[server] @ git+https://github.com/AryaFayyazi/GreyBox"
```

| Monitor | Hardware and speed | Use it when |
|---|---|---|
| `TextMonitor` (TF-IDF and logistic regression) | CPU, about 0.2 ms for each action | your tools and policies are the same as in your labelled data |
| `ReaderMonitor` (Qwen3-8B reads the transcript; a linear probe) | 1 GPU, 41 ms for each action | your tasks change; it is more robust to new tasks |

```python
from graybox.monitor import TextMonitor, Step

tm = TextMonitor.load_json("greybox-text-monitor.json")      # trained in the browser trainer
if tm.escalate(system_prompt, user_task, history, ("send_message", args)):
    hold_for_review()
```

For the reader, use `QuantileGate` to set the review threshold. It takes the top 5% of the agent's first 500
actions, then it freezes the threshold. Do not use the raw probability: it does not transfer between agents.

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
| T8b: a closed frontier agent | Claude Haiku 5.5, 3,128 actions. Trained only on Gemma, the reader catches 95.6% of the 68 violations at a 5% review budget. The judge catches 0%. Pooled AUROC 0.989 (judge 0.570; exploratory). The preregistered within-template AUROC (0.991 vs 0.633) uses only one stratum: 136 actions, 26 violations. |
| T9: reader or labels? | A TF-IDF monitor with the same labels is equal to the reader on the training benchmark (Claude 0.997 vs 0.991). On unseen task families the reader is better (0.969 vs 0.950). On AgentDojo, TF-IDF is at chance (0.50 vs 0.78). |
| T10: a policy that the monitor never saw | The reader still ranks the violations well (pooled 0.912; judge 0.570). But at a 5% budget it escalates 0 of 26 violations. With 20 to 40 labelled examples of that policy, it catches 85% to 94% (exploratory). |
| T11: a second closed family (GPT) | GPT-6-luna, 3,627 actions, 66 violations in 6 families. Pooled AUROC 0.996 (judge 0.778; tool identity alone 0.856). Within tool 0.994 (judge 0.956). At a 5% budget it catches 98.5% (judge 37.9%). All three preregistered endpoints pass. |
| T12: labels for a new policy, prospective | We held out a credentials policy. With no labels of it, the reader caught all 18 of GPT's violations of it. The preregistered H1 (labels improve the catch rate) fails: there was nothing to fix. A new policy that resembles a trained one can transfer; one that does not (T10) needs labels. |
| T13, T13b: a closed agent on an external benchmark | GPT-6-luna on AgentDojo. With the spotlighting defense, attacks succeed in 2.1% of episodes (4 injected calls): not tested. Without the defense, 36 injected calls: reader 0.840 vs judge 0.787 (+0.052 [−0.026, +0.130]). The preregistered test fails: on an external benchmark the reader is not significantly better than a zero-shot judge for this closed agent. |
| T14: labels of the new policy, from open agents | We added labelled AgentDojo calls of the two open agents to the training data, and tested on the closed GPT agent. AUROC 0.840 → 0.922 (judge 0.787); injected calls caught at a 5% budget 36% → 92%. Both preregistered endpoints pass. Labels of a policy transfer from open agents to closed ones. |
| Review threshold without labels (exploratory) | A threshold from the agent's own unlabelled traffic, frozen after 500 actions: 95.4% caught online, 1.9% of safe actions escalated. With a burst of violations, it stays at 94%; a rolling threshold falls to 36%. |

**Results that did not confirm** are in the record too. Examples: the agent's answers alone do not match the
white-box probe (T1, F3), and resampling scored by an LLM judge loses to the probe (T3).

## Limits

- Most agent tests use one benchmark, CAPE-Bench. On the external test (AgentDojo), the reader beats a zero-shot judge
  for open agents (0.78 vs 0.54), but not significantly for a closed agent (0.840 vs 0.787, T13b).
- A monitor can miss a policy that is not like the policies in its training data (T10). Label a few dozen steps of
  each such policy. A policy that is like a trained one can transfer without labels (T12).
- Corollary 3 applies to action-level labels. It does not apply to labels about later behaviour or about beliefs.
- Chat readouts recover the behaviour of a model, not its hidden beliefs.

## Chat: forecast a refusal before the reply

A *grey box* is the model's **interrogation state** Ψ(x): the probabilities of the answer "Yes" to a fixed list of
questions. We ask the questions in side branches of the conversation, so the real reply never sees them.
- Ψ recovers the hidden state of models with known internals: a program compiled into a transformer (21/21 states),
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
| `docs/paper_draft.md` | the paper draft |
| `results/` | the saved output of each analysis; every number in the paper comes from these files |
| `site/` | the website builder and the browser-vs-Python parity tests |

Checks:
- `python scripts/99_verify_all.py` verifies each lock hash and prints the headline numbers.
- `python scripts/98_audit_numbers.py docs/paper_draft.md README.md` checks that each decimal in a document is in a
  saved result.
- `pytest tests` runs the SDK tests (also in CI).

The experiments ran on a Slurm cluster, one GPU at a time (`slurm/run.sh`). Paid API runs share a file-locked spend
ledger with a hard cap.
