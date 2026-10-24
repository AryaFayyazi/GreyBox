# Preregistration G46c: the policy forecast and the policy-gap identity with the exact context

Locked, and published, before any G46c episode exists and before G46's or G46b's locked analyses have run. Code:
scripts/232_g46c_exact_context.py (it runs scripts/226's own and other stages, and scripts/230's share function, both
hashed in earlier locks). Paid (Anthropic) plus GPU.

## Why (a problem found before G46b's result)
G46 rebuilds each unit's conversation from its episode log, which keeps only the first tool call of each turn. When Claude
Haiku 5.5 emits several tool calls in one turn, the loop runs the first and refuses the others ("Not executed: one tool
call per turn"); the rebuilt history drops the refused calls. With G46's own draws complete (allowed after G46b's lock),
a calibration check showed the draws violating less often than the actual actions at position 2 (0.568 vs 0.700, 444
units), and agreeing at the other positions. The identity that G46b tests needs the draws to come from the same policy at
the same context as the actual action, so G46b may fail for this reason alone. G46c repeats the test with the exact
context: every episode logs the exact message list before each turn, and both models draw from it.

## Design (fixed)
- Claude Haiku 5.5, scripts/74's loop settings, no pre-call battery, all 41 x- scenarios of cons_service_restart, fresh
  seeds 160–171 (492 episodes); per turn the exact Anthropic messages before the call and the number of tool_use blocks.
- Units as G46 (executed non-read calls up to and including the first unsafe one; template|tool strata with both
  classes; position = ordinal among the episode's units).
- own: scripts/226's own_unit unchanged (8 draws from the exact context, each distinct call read by the agent's own 8
  answers x 4). other: scripts/226's other stage unchanged (Gemma-4, 8 draws), from the exact context converted to its chat
  format with every assistant tool call and every tool result kept.
- Policy rates as G46b: the unsafe share among each model's non-read draws; units need >= 2 non-read draws from each.

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 4620)
- **C0.** Calibration: mean Y − mean f_own (units with >= 2 non-read own draws): 95% interval inside [−0.05, +0.05].
- **R1.** Own forecast (scripts/226's read-out forecast), AUROC within template|tool|position: lower 95% bound > 0.50.
- **B3 (primary).** Observed − predicted Brier excess of the other model (scripts/230's formulas): interval inside ±0.02.
- **B2.** Policy gap with both sampling noises removed: lower 95% bound > 0.
- **Reported.** Share of turns with more than one tool_use block; B1 observed and the predicted excess.

## Readings, fixed in advance
- C0 passes: with the exact context, the agent's draws are calibrated to its actual actions (the G46 gap was the replay).
- B3 passes: the proposition predicts, in size, how much another model loses before the call.
- B2 passes: on this closed agent the policies differ, so the forecast needs the agent's own policy.
- C0 fails: draws from the agent's own API at the exact context do not reproduce its actual action rate; the identity
  cannot be tested this way.

## Cost and conduct
Episodes about $0.002 each (about $1); own about $0.004 per unit (about $6); Anthropic ledger cap 154.55. Runs after G49
(one process per provider). GPU: Gemma-4 after G46's other stage (one GPU chain).
