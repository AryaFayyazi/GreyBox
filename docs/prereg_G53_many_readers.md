# Preregistration G53: does the theory predict, reader by reader, how much another model loses before the call?

Locked, and published, before any G53 draw exists and before G48's other-model (Gemma-4) draws exist. Code:
scripts/238_g53_many_readers.py (scripts/226's other stage and scripts/230's quantities, both earlier locked). GPU only.

## Why
The proposition "the forecast is the agent's policy" predicts that a reader that forecasts with another policy loses
E[(sum_a (pi_bar - pi')(a | V) r(a, V))^2], so different readers should lose different, predictable amounts. G46b and G48
test this with one reader (Gemma-4). G53 adds three more open readers on G48's GPT-6-luna units, where the replay is
faithful (one tool call per turn; G48's calibration C0 passed: +0.015 [0.008, 0.023]).

## Design (fixed)
- Units: G48's (GPT-6-luna, 1,134 calls, 131 violations, six templates). Agent's own draws: G48's.
- Readers: gpt-oss-20b, Qwen3.8-27B, Muse-Glimmer-30B, each with its own vLLM settings (CAPE's actor table), 8 draws in the
  agent's seat from the replayed context (scripts/226's other stage unchanged); Gemma-4's draws come from G48.
- Per reader (units with >= 2 non-read draws from both the agent and the reader): observed Brier excess, the predicted
  excess and the policy gap (scripts/230's formulas, r from CAPE's checker on each drawn call).

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 5300)
- **T1 (primary).** Mean over the three new readers of observed − predicted: 95% interval inside [−0.02, +0.02].
- **Reported.** Per reader (four, with Gemma-4): observed, predicted, observed − predicted, policy gap, and the within-position
  AUROC of its draws' forecast; the number of readers whose interval lies inside ±0.02.

## Readings, fixed in advance
- T1 passes: the proposition predicts, in size, how much other readers lose before the call, across readers whose policies
  differ from the agent's by different amounts.
- T1 fails: the policy-gap account does not predict other readers' losses at this resolution.

## Conduct
One GPU job per reader, chained after G48's other stage (one GPU at a time); Muse-Glimmer with its vLLM 0.30 env.
