# Preregistration G46: the forecast lives in the agent's policy — its own sampled next actions vs its self-report vs another model

Locked, and published, before any G46 episode exists. Code: scripts/226_g46_rollout_forecast.py. Paid (Anthropic) plus GPU.

## Why
Proposition "the forecast needs the agent": before a call exists, the Bayes forecast of the coming violation is
f = sum_a pi_bar(a | V) r(a, V), the agent's own policy averaged over its hidden state; a reader with another policy loses
E[(sum_a (pi_bar - pi')(a | V) r(a, V))^2]. Two ways to read pi_bar through the agent's API, with no training:
(a) ask it about its next action (self-report; G27's pre-call battery), (b) sample its next action in side branches and
read each drawn call with its own answers. On GPT-6-luna (a) forecast beyond position (G40). On Claude Haiku 5.5 an
exploratory pilot on G45's development episodes (scripts/224, results/t1/rollout_forecast_pilot_exploratory.json;
398 units, 81 violations) found: self-report within position 0.45 [0.37, 0.51]; own draws read by own answers (1 answer
each) 0.68 [0.54, 0.84]; checker-labelled draws 0.72 [0.62, 0.84]. That is, the agent's policy forecasts the violation
but its self-report does not. G46 tests this on fresh episodes, at 4 answers per question, against another model
sampling in the agent's seat.

## Design (fixed)
- Agent: Claude Haiku 5.5, G45's episode code unchanged (scripts/74's loop, reasoning off, G27's pre-call battery of 8
  forward questions x 4 answers before every turn), on all 41 x- scenarios of cons_service_restart, fresh seeds 100–111
  (492 episodes).
- Units: executed non-read calls up to and including the first unsafe one, in template|tool strata holding both classes;
  position = the unit's ordinal among the episode's units.
- **own** (the method): before the unit's turn, K = 8 next actions drawn from the agent through its API (side branches,
  its loop's settings); each distinct drawn call read by the agent's own 8 at-call answers (graybox.selfaudit
  STATE_QUESTIONS and STATE_FRAME, 4 answers each); forecast = mean over the 8 draws of the fixed read-out; a draw that
  ends the episode scores the read-out's minimum. No label, no training, no other model.
- **other** (a control): Gemma-4 (thinking on) in the agent's seat, the conversation replayed in its chat format: its own
  8 draws, read by its own 8 answers x 4, scored the same way.
- **report** (a control): the agent's self-report, G27's raw read-out of the pre-call battery before the unit's turn.
- **ceiling** (reported): share of the agent's 8 draws that CAPE's checker calls unsafe.

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 4600)
- **R1 (primary).** own, AUROC within template|tool|position: lower 95% bound > 0.50.
- **R2.** own − other, AUROC within template|tool: lower 95% bound > 0.
- **R3.** own − report, AUROC within template|tool|position: lower 95% bound > 0.
- **Reported.** All four scores within template|tool and within position; units, violations, episodes.

## Readings, fixed in advance
- R1 passes: on a second closed agent, before the call exists, the agent's own policy, read through its API with no
  training and no other model, forecasts its coming violation beyond position.
- R2 passes: another model sampling in the agent's seat forecasts worse; the forecast needs the agent.
- R3 passes: the agent's policy forecasts what its self-report does not: asked what it will do, it does not say; sampled,
  it does.
- R1 fails: the policy forecast of the pilot does not hold on fresh episodes.

## Cost and conduct
Episodes about $0.009 each (about $4.4); own about $0.006 per unit (about $10 for about 1,600 units); Anthropic ledger cap
154.55. GPU: Gemma-4 for other, after G45's GPU stages. One process per provider; checkpoint per wave. If the budget or
an API error stops a stage, it resumes from its checkpoint; if it cannot finish, we analyse the complete units and record
this.
