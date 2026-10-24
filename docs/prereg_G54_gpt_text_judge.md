# Preregistration G54: a reasoning text monitor vs the agent's own policy on GPT-6-luna

Locked, and published, before any G54 judge answer exists (two smoke answers, not kept, tested the code). Code:
scripts/239_g54_gpt_text_judge.py (uses G48's units and own draws, G52's judge prompt and parser). Paid (OpenAI).

## Why
G52 found that, before the call, Claude Haiku 5.5 reasoning over its own transcript forecasts its coming violation at
chance (0.448) while its sampled policy reaches 0.891. Reviewers asked for the same baseline on the second closed agent,
whose self-report does forecast (G45: 0.836 within position). G54 runs it on G48's GPT-6-luna units.

## Design (fixed)
- Units: G48's (1,134 calls, 131 violations, six templates). Judge: GPT-6-luna, reasoning_effort medium, one call per
  unit, G52's system prompt, the transcript in the same format; "Probability: N" (0–100); unparsable -> median.
- Compared: G48's own forecast (8 sampled next actions read by its own answers) and G45's self-report.

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 5400)
- **J1 (primary).** Own forecast − judge, AUROC within template|tool|position: lower 95% bound > 0.
- **Reported.** Self-report − judge (within position, interval); all three within template|tool; judge tokens and latency.

## Readings, fixed in advance
- J1 passes: on GPT-6-luna too, the sampled policy forecasts better than a reasoning text monitor.
- J1 fails: for GPT-6-luna, whose words track its actions, a text monitor forecasts as well; reading the policy is needed
  where words and actions diverge (Claude Haiku 5.5), not everywhere.

## Cost and conduct
About $0.0005 per unit (about $0.6); OpenAI ledger cap 26.80.
