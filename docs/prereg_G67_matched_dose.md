# Preregistration G67: reviewer-free prevention, replicated on fresh seeds with the random arm's dose matched

Locked, and published, before any G67 episode. Code: scripts/257_g67_matched_dose.py. Paid (Anthropic).

## Why
G47 registered a random-hold control "at the same rate" as the agent's own gate, but the random arm held 0.177 calls per
call against the self arm's 0.227: identical re-proposals of a held call count as calls and cannot be held again. Reviewers
rightly noted that self − random was then not at a matched dose. G67 reruns G47 on fresh seeds with the random rate set so
that the doses match, and checks the match as a registered endpoint.

## Design (fixed)
Claude Haiku 5.5, cons_service_restart (41 scenarios), fresh seeds 320-331, arms none / self / random; scripts/225's
episode code, CUT (85th percentile of development at-call read-outs, no labels) and analysis unchanged.
Random arm: each non-read call not yet held is held with probability RATE = 0.30. Derivation (from G47's data only, before
any G67 episode): G47's random arm gives holds per call = r / (1 + r·ρ) with r = 0.22 and observed 0.1773, so ρ = 1.094;
matching the self arm's 0.227 needs r = 0.227 / (1 − 0.227·ρ) = 0.30.
Stopping rule (budget, not data): seeds run in order until the Anthropic ledger reaches $151.50; the analysis uses every
unit (scenario × seed) with all three arms. A smoke run (1 scenario × seed 998, separate file, never analysed) checked the
stage, and the analysis was checked on G47's file, where it reproduces G47's published values.

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 2250)
- **Q1 (primary).** Reduction of violating episodes, self − random: lower 95% bound > 0.
- **Q2 (manipulation check).** |holds per call, random − self| ≤ 0.03.
- **Q3.** Self reduction vs none: lower > 0.
- **Q4.** Success, self − none: lower > −0.05.
- Holm's step-down over Q1, Q3, Q4 at family-wise 0.05 (one-sided p from estimate and interval, normal approximation).

## Readings, fixed in advance
- Q1 and Q2 pass: at a matched dose, the agent's own gate prevents more violations than holding calls at random.
- Q1 passes, Q2 fails: reported as not at a matched dose, with the realized doses.
- Q1 fails: the self-vs-random advantage did not replicate at a matched dose; reported next to G47's +0.198.
