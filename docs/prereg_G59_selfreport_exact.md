# Preregistration G59: says vs does on the same episodes, with exact contexts

Locked, and published, before any G59 answer exists. Code: scripts/247_g59_selfreport_exact.py. Paid (Anthropic).

## Why
Reviewers noted that the paper's "says" for Claude Haiku 5.5 (0.47, G46) and its "does" (0.89, G46c) come from two sets
of episodes, and that G46's own draws scored 0.71 on rebuilt contexts. G59 measures the self-report on G46c's units, so
that says, the text monitors and does all use the same episodes and exact contexts.

## Design (fixed)
G46c's units (Claude Haiku 5.5, cons_service_restart, 41 scenarios, exact contexts). Before each unit's turn, G45's
pre-call battery unchanged (G27's frame and 8 forward questions, 4 answers each, appended to the last user turn;
tool_choice none, temperature 1, reasoning off). Self-report = G44's raw read-out (as in G45 and G46). Own draws = G46's
forecast on G46c's own stage.

## Endpoint (cluster bootstrap over scenarios, 2000 resamples, seed 5900; AUROC within template|tool|position)
- **S1 (primary).** Own draws − self-report: lower 95% bound > 0.
- Reported: the self-report's AUROC with its interval.

## Readings, fixed in advance
- S1 passes: on the same episodes, what Claude does (its sampled next actions) forecasts its violation better than what
  it says about its next action.
- S1 fails: on exact contexts, Claude's self-report forecasts as well as its sampled actions; the says–does gap of G46
  came from the rebuilt contexts.

## Cost
About $0.001 per unit (32 short requests with caching), about $1.7; Anthropic ledger cap 154.55.
