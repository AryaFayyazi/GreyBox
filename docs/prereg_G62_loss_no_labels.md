# Preregistration G62: a text monitor's loss predicted with no label at all

Locked, and published, before this analysis. Code: scripts/251_g62_loss_no_labels.py. CPU; existing data.

## Why
G60 predicted three text monitors' Brier loss against the agent's policy with no label of the executed calls, but CAPE's
checker labelled every sampled action. Reviewers noted that a deployer has no checker for sampled actions. Here a reader
replaces the checker: GPT-6-luna's probability that each drawn call violates (G58's reads, reasoning medium).

## Data (fixed)
G46c's units (Claude Haiku 5.5, exact contexts). r = G58's probability for each non-read drawn call; f = mean r over the
unit's non-read draws (k of them, k >= 2), s2 = sample variance of r (ddof 1). q = the text monitor's probability:
claude_self (G52's Claude judge) and claude_gpt (G55's GPT-6-luna judge).

## Quantities
predicted = mean (f - q)^2 - 2 mean s2/k (no label anywhere); observed = mean (Y - q)^2 - mean (Y - f)^2 (Y, CAPE's label of
the executed call, only to score the prediction). Cluster bootstrap over scenarios, 2000 resamples, seed 6200.

## Endpoint
- **L1 (primary).** For each of the two pairs, observed - predicted: 95% interval inside ±0.02 (G60's band).
- Reported: calibration of the reader's draws (mean Y - mean f).

## Readings, fixed in advance
- L1 passes: with an LLM reading the agent's sampled actions, a deployer can predict a text monitor's loss against the
  agent's policy with no label at all.
- L1 fails: the reader's probabilities are not calibrated enough; the calibration term shows by how much.
