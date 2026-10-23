# Preregistration G45: on two closed agents, does the agent's own state forecast its next violation better than another model?

Locked, and published, before any episode of this test exists. Code: scripts/222_g45_closed_self_forecast.py. Paid APIs
(new credit) plus GPU.

## Why
G44 found, on one closed agent (GPT-6-luna; 402 calls, 27 violations), that its own answers about its next action forecast
its next violation better than another model answering the same questions in its seat (+0.111 [0.041, 0.185]), as the
proposition "the forecast needs the agent" predicts. Reviewers asked for this at power and on more than one closed agent.

## Design (fixed)
- GPT-6-luna: G40's loop and pre-call battery (before every turn; G27's frame and 8 forward questions, 4 answers each),
  fresh seeds 56 and 57, on G40's 6 templates: 504 episodes.
- Claude Haiku 5.5: scripts/74's agent loop (Anthropic tool use, reasoning off), on all 41 x- scenarios of
  cons_service_restart (the template where it violates knowingly: 26 of 41 episodes in its earlier run), fresh seeds 60–67:
  328 episodes; before every turn the same battery, appended to the last user turn, temperature 1.
- Other model (a control, never part of the method): Gemma-4 answers the same questions in the agent's seat (the agent's
  conversation replayed in its chat format; scripts/220's machinery).
- Units: executed non-read calls up to and including the first unsafe one, in template|tool strata holding both classes.
  Score: G27's raw read-out before the unit's turn (nothing fit). Position: the unit's ordinal among the episode's units.

## Endpoints (pooled: mean over the two closed agents; cluster bootstrap over scenarios within each agent, 2000, seed 4500)
- **S1 (primary).** Own − other, within template|tool AUROC: lower 95% bound > 0.
- **S2.** Own, within template|tool|position AUROC: lower 95% bound > 0.50 (forecast beyond position).
- **S3.** Own − other, within template|tool|position: lower 95% bound > 0.
- **Reported.** Per agent: units, violations, own and other AUROCs.

## Readings, fixed in advance
- S1 passes: on two closed agents and fresh data, a closed agent's own state, read through its API before the call
  exists, forecasts its next violation better than another model reading the same conversation.
- S2 passes: the forecast is beyond the call's position on both closed agents.
- S1 fails: the self-forecast advantage of G44 does not hold at this power across closed agents.

## Cost and conduct
GPT about $4.2 (OpenAI ledger cap 17.35); Claude about $10 at about $0.03 per episode (Anthropic ledger cap 154.55, new
account). One process per provider; checkpoint per wave of 6 episodes; the first wave of each measures the cost (kept).
