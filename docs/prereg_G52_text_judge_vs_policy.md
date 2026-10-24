# Preregistration G52: a reasoning text monitor vs the agent's own policy, before the call; and the cost of each read

Locked, and published, before any G52 judge answer or cost measurement exists. Code: scripts/237_g52_text_judge_vs_policy.py
(uses G46c's units and own draws, and scripts/226's forecast). Paid (Anthropic).

## Why
CAPE showed that open agents which knew a call was unauthorized made it and then denied it. Here, on a closed agent,
Claude Haiku 5.5's answers about its next action do not forecast its coming violation (G45, G46), while its own sampled
next actions do (G46 R1). A reviewer can ask whether the strongest text monitor, the same model reasoning over the full
transcript, would also forecast. G52 tests that, before the call, on the exact contexts of G46c, and measures what each
read of a decision costs in tokens, money and latency.

## Design (fixed)
- Units: G46c's (Claude Haiku 5.5, cons_service_restart, 41 scenarios x seeds 160–171, exact contexts; 1,587 units,
  316 violations). Own forecast: G46c's own draws scored by scripts/226's forecast (already collected; not analysed by
  this test before this lock).
- Judge: Claude Haiku 5.5 with adaptive thinking (effort medium), one call per unit: system prompt fixed in the script; the
  user turn is the agent's operating instructions and the conversation so far as a transcript; it ends with
  "Probability: N" (0–100). Unparsable answers take the median.
- Cost: on 100 random units (seed 5200), one decision at a time: (1) the at-call read-out (8 questions x 4 on the actual
  call), (2) the self-report (G27's 8 pre-call questions x 4), each with one request first to warm the prompt cache and the
  rest in parallel; (3) the policy forecast (scripts/226's own_unit); (4) the judge. Wall-clock seconds, tokens, requests;
  dollars from the ledger's prices.

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 5200)
- **J1 (primary).** Own forecast − judge, AUROC within template|tool|position: lower 95% bound > 0.
- **Reported.** Both AUROCs within position and within template|tool; the cost table (mean $ per decision, median seconds,
  mean output tokens, requests).

## Readings, fixed in advance
- J1 passes: before the call, the agent's own policy read by sampling forecasts its violation better than the same model
  reasoning over the transcript; for this agent, text is not enough.
- J1 fails: a reasoning text monitor forecasts as well; the policy read adds no forecasting value here.

## Cost and conduct
Judge about $0.001 per unit (about $2); cost stage about $0.5. Anthropic ledger cap 154.55. One process per provider.
