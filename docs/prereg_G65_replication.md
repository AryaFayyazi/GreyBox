# Preregistration G65: prospective replication of the closed-agent forecasting results on fresh episodes

Locked, and published, before any G65 episode exists. Code: scripts/254_g65_replication.py. Paid (Anthropic, OpenAI).

## Why
Reviewers noted that the main closed-agent forecasting results (G52, G55, G58, G59, G60, G61) were measured on one set of
episodes (G46c's), and that several of them were registered on that existing set after related results were known. G65
re-measures all of them on 492 NEW episodes, with every endpoint and the multiplicity correction fixed before any episode
is run. Nothing is chosen after the data.

## Design (fixed)
Claude Haiku 5.5 (reasoning off), cons_service_restart x 41 scenarios x seeds 300-311 (492 episodes; seeds never used
before), exact contexts. Every stage is an existing stage function, unchanged, pointed at new files: G46c's episode and
own-draw stages (8 own draws per unit, read by the agent's own answers), G59's self-report, G52's Claude text judge
(reasoning, adaptive effort medium), G55's GPT-6-luna text judge (reasoning medium), G58's GPT-6-luna read of each drawn
call. Units: G46c's (executed non-read calls up to and including the first violation; strata with both classes).
A smoke run (2 scenarios x seed 999, separate files, never analysed) checked the stages before this lock, and the analysis
code was checked on G46c's existing files, where it reproduces the published values.

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 6500; AUROC within template|tool|position unless noted)
- **P1 (primary).** Own draws - self-report: lower 95% bound > 0.
- **P2.** Own draws - Claude text judge: lower > 0.
- **P3.** Own draws - GPT-6-luna text judge: lower > 0.
- **P4.** Across steps (within template|tool): mean rank of GPT-6-luna's judge and the own draws - that judge alone: lower > 0.
- **P5.** Own draws: lower > 0.5.
- **P6a, P6b.** Loss prediction from checked draws (scripts/248's identity), observed - predicted Brier excess, for the
  Claude judge and for the GPT-6-luna judge: 95% interval inside ±0.02.
- **P7.** GPT-6-luna reading the own draws - GPT-6-luna reading the transcript: lower > 0.
- **Multiplicity, fixed here.** One-sided p for each of the 8 endpoints from its estimate and interval (normal
  approximation, SE = width/3.92; equivalence: the larger of the two one-sided p's); Holm's step-down at family-wise 0.05.
  An endpoint counts as confirmed only if it passes its bar and survives Holm.

## Readings, fixed in advance
- P1 confirmed: on fresh episodes, what Claude Haiku 5.5 does (its sampled next actions) forecasts its violation better
  than what it says. P2, P3 confirmed: better than two reasoning text monitors. P4: even across steps, the sampled actions
  add to the best text monitor. P6: the theory predicts each text monitor's loss from checked samples. P7: sampling, not
  the reader, is what forecasts.
- Any endpoint not confirmed is reported as a failed replication, next to its original value.

## Cost
About $5-8 (Anthropic) and $1 (OpenAI), from G52's measured per-decision costs and the smoke run.
