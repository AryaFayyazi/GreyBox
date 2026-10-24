# Preregistration G50: the policy forecast on an external benchmark (tau2-bench airline, Claude Haiku 5.5)

Locked, and published, before any G50 draw exists. Code: scripts/234_g50_tau2_policy_forecast.py (uses scripts/184's
questions, frame, operator note and parser, and scripts/101's context conversion, both earlier locked). Paid (Anthropic).

## Why
On CAPE-Bench, Claude Haiku 5.5's own sampled next actions, read by its own answers, forecast its coming violation where
its answers about its next action do not (G46: 0.708 within position vs 0.475). That result comes from one template.
On tau2-bench airline (Sierra's external benchmark), Claude's answers about its next write did not forecast its policy
violations (G37: 0.54 [0.43, 0.68]). G50 asks whether reading its policy by sampling does, on the same kind of writes.

## Design (fixed)
- Units: G18's 360 confirmation writes (the exact contexts tau2's LLMAgent sent; 50 NOT_ALLOWED, labelled by the
  airline policy oracle), sampled with G18's seed; label = the actual write's.
- own forecast (the method): before the write exists, K = 8 next actions drawn from the agent through its API with its
  own settings (thinking disabled, effort low, max_tokens 2048, the tau2 tools), in side branches of the context before the
  write; each distinct drawn tool call read by the agent's own answers to G18's 8 airline questions (4 answers each, G18's
  frame and operator note); forecast = mean over the 8 draws of G18's fixed read-out; a draw that is not a tool call
  scores the read-out's minimum. No label, no training, no other model.
- Reported: G18's at-call read-out on the same writes; the share of draws that are tool calls.

## Endpoints (cluster bootstrap over tau2 tasks, 2000 resamples, seed 5000)
- **P1 (primary).** Forecast, AUROC within tool (pair-weighted, G18's metric): lower 95% bound > 0.50.
- **Reported.** Forecast − at-call read-out within tool, with interval.

## Readings, fixed in advance
- P1 passes: on an external benchmark, before the write exists, the closed agent's own policy read by sampling forecasts
  its policy violation, where its answers about its next action did not (G37).
- P1 fails: the policy forecast does not carry to tau2-bench.

## Cost and conduct
About $0.009 per unit (about $3.2); Anthropic ledger cap 154.55. One process per provider; checkpoint per wave.
