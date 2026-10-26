# Preregistration G75: the geometry-only gate against random holds at a matched dose

Locked, and published, before any G75 episode. Code: scripts/267_g75_geo_vs_random.py. Paid (Anthropic).

## Why
G72's geometry-only gate (no label, no question sign) cut Claude Haiku 5.5's violating episodes from 65.9% to 44.9% at
0.171 holds per call, but G72 had no random-hold arm; in G67, random holds alone (0.234 per call) cut them to 42.9%. All
three round-17 reviewers asked whether the geometry beats holding calls at random at the same dose.

## Design (fixed)
Claude Haiku 5.5, cons_service_restart (41 scenarios), fresh seeds 360-371 (never used), two arms on the same units
(492 each), no human, checker or other model in the loop: geo (G72's gate, unchanged: same centre, cut and tie rule) and
random (the same hold on each non-read call not yet held with probability 0.21 by a hash of scenario, seed and turn;
r = 0.171 / (1 - 0.171 x 1.094) = 0.21, from G47's random arm's re-proposal factor, to match G72's realized geometry-gate
dose). A pilot (1 scenario, seeds 980-983, separate file, never analysed) checked both arms. Cluster bootstrap over
scenarios, 2000, seed 7500. Cost about $5; stop if the Anthropic ledger reaches $144.68 (budget, not data).

## Endpoints
- **R1 (primary).** Violating-episode rate, random - geo: lower 95% bound > 0.
- **R2 (dose check).** |holds per call, geo - random| <= 0.03.
- Reported: success in both arms.

## Readings, fixed in advance
- R1 and R2 pass: at a matched dose, a gate on the geometry alone prevents more violations than holding calls at random;
  the geometry carries the prevention, not only the act of holding.
- R1 fails: at this dose and size, the geometry-only gate is not shown to beat random holds; G72's prevention is then
  attributed to hold-and-recheck plus the reminder, and the paper says so (the signed read-out did beat random holds, G67).
