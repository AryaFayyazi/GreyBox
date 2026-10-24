# Preregistration G60: the theory predicts, without labels, how much a text monitor loses

Locked, and published, before this analysis. Code: scripts/248_g60_text_monitor_loss.py. CPU; existing data.

## Why
Reviewers judged the theory elementary and noted that its one size prediction (G46b, another model's draws) failed,
because that replay dropped refused tool calls. Proposition 3 makes a sharper prediction for any forecaster that sees only
the context, such as a text monitor: its Brier loss against the agent's policy is E[(pi - q)^2], and the agent's own draws
estimate it with no label. G46c (exact contexts) and G48 (faithful replay) now give draws from the agent's real policy.

## Data (fixed; none analysed this way before this lock)
- claude_self: G46c's units (Claude Haiku 5.5, exact contexts, own draws) x G52's judge (Claude reasoning over the transcript).
- claude_gpt: the same units x G55's judge (GPT-6-luna reasoning over Claude's transcript).
- gpt_self: G48's units (GPT-6-luna, faithful replay, own draws) x G54's judge (GPT-6-luna reasoning over its transcript).
Units with at least 2 non-read draws; f = share of violating non-read draws (CAPE's checker), k = their number, q = the
judge's probability, Y = CAPE's label of the executed call.

## Quantities
observed = mean (Y - q)^2 - mean (Y - f)^2; predicted = mean (f - q)^2 - 2 mean f(1-f)/(k-1) (no label);
label-free policy gap = mean (f - q)^2 - mean f(1-f)/(k-1). Cluster bootstrap over scenarios, 2000 resamples, seed 6000.

## Endpoints
- **E1 (primary).** For each of the three pairs, observed - predicted: 95% interval inside ±0.02 (G46b's band).
- **E2.** Label-free policy gap, claude_self - gpt_self: lower 95% bound > 0 (the gap, computed with no label, separates
  the agent whose words do not track its actions from the one whose words do).
- Reported: each pair's observed and predicted excess, gap and calibration (mean Y - mean f).

## Readings, fixed in advance
- E1 passes: the theory predicts in size, from the agent's draws alone, how much each text monitor loses against the
  agent's policy, on both closed agents.
- E1 fails for a pair: the prediction does not hold there; the calibration term shows whether the draws were biased.
- E2 passes: the label-free gap confirms the two regimes (exploratory until now).
