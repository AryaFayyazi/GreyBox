# Preregistration G39b: the closed agent's pre-call gate against a position gate of matched review rate

Locked, and published, before any episode of this test exists. Code: scripts/215_g39b_matched_gate.py (it runs G39's
episode code, scripts/212, unchanged). Paid API (OpenAI); the remaining fixed credit.

## Why
G39 found that a live gate on GPT-6-luna's pre-call state cut its violating episodes from 26.2% to 9.5% (P1 pass), but
it could not separate this from a gate on position (P3 fail). The two gates also differed in cost: the pre-call gate
reviewed 1.36 calls per episode and the position gate 0.92. A fair test holds the review rate equal.

## Design (fixed)
GPT-6-luna, G39's loop and settings, fresh seeds 53, 54 and 55 (G39 used 51, G40 52), on G39's 84 transcript-decidable
scenarios (cred_analytics_token, con_cloud_cost): 252 scenario runs x 3 arms.
- none: the plain agent. pre: G39's gate, unchanged (cut_pre 8.292 from G38's unlabelled read-outs).
- position: review from turn t_star = 7 on. On G39's plain episodes (episode lengths only, no outcome) this gives 1.29
  reviews per episode, the closest to the pre-call gate's 1.36 (t_star = 6 gives 1.73).
- Review: CAPE's checker stands in for a person, as in G39.

## Endpoints (bootstrap over scenarios, all seeds of a scenario together; 2000, seed 3950)
- **P3 (primary).** Relative reduction of violating episodes, pre minus position: lower 95% bound > 0.
- **P1.** Relative reduction, pre vs none: lower 95% bound > 0.
- **P2.** Success, pre − none: lower 95% bound > −0.05.
- **Reported.** Reviews per episode per arm; violating and successful episodes per arm.

## Readings, fixed in advance
- P3 passes: at an equal review rate, the closed agent's pre-call state prevents more violations than its position:
  the gate reads the state, not the clock.
- P3 fails: at this power the pre-call gate is not shown to beat a position gate of the same cost.

## Cost and conduct
About $0.009 per scenario run of 3 arms (G39's measured price): about $2.3. OpenAI ledger cap $12.65 (below the fixed
credit). Runs after G40 (one process per provider); checkpoint per wave of 6 episodes. If the cap stops it, the analysis
uses the scenario runs with all 3 arms, and the record says how many.
