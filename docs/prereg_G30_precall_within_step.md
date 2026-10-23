# Preregistration G30: the state before a call, beyond position, confirmed on three more agents

Locked, and published, before any analysis of G27b's answers: gpt-oss-20b's battery exists on disk and has not been
read (beyond counting parsed answers and server errors), and Qwen3.8's and Muse-Glimmer's batteries do not exist yet.
Code: scripts/200_g30_precall_within_step.py.

## Why
On Gemma-4, exploratory analyses after G27's locked analysis showed that the agent's answers before a call rank that
call's violation among calls at the same step, template and tool (0.674 [0.567, 0.776]), and that they track the true
hidden state within a step (same-step Spearman 0.158). An exploratory finding needs confirmation on new data.

## Data and scores (fixed; nothing fit; no label is used for the scores)
G27b's batteries: gpt-oss-20b (2,042 steps; parsed 74.9%: the model often tried to make a call; missing answers take
the neutral angle, which weakens the test), Qwen3.8 (750 steps), Muse-Glimmer (2,384 steps). Score: G27's raw pre-call
read-out, the signed sum of the step's 8 answer angles. Truth: the stored decision states, layer index 3.

## Endpoints (pooled: mean over the three agents; cluster bootstrap over scenarios within each agent, 2000, seed 3000)
- **W1 (primary).** Within template x tool x step, AUROC of the pre-call read-out: lower 95% bound > 0.50.
- **W2.** Spearman, over same-step pairs, between the answer geometry and the true decision-state geometry: lower 95%
  bound > 0.
- **Reported.** Per agent; strata with both classes and their violations; parse rates.

## Readings, fixed in advance
- W1 passes: on three more agents, the answers asked before a call carry its violation beyond its position.
- W2 passes: within a step, the answers through the API track the agent's true hidden state.
- Either fails: it does not replicate at this power.
