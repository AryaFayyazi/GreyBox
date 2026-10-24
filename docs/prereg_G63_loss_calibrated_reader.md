# Preregistration G63: the loss prediction with a reader recalibrated on a few labelled draws from other scenarios

Locked, and published, before this analysis. Code: scripts/252_g63_loss_calibrated_reader.py. CPU; existing data.

## Why
G62 failed: with GPT-6-luna's probability for each sampled action in place of CAPE's checker, the predicted loss of a text
monitor missed the observed by -0.226, because the reader overstates violations (calibration -0.195). The theory says the
prediction needs calibrated probabilities on the draws. A deployer can get them once: a reviewer labels a small set of
sampled actions from other tasks, and the reader is recalibrated on them.

## Design (fixed)
G46c's units (Claude Haiku 5.5, exact contexts). Scenarios split in two folds (seed 6300). For each fold, Platt scaling
(logistic regression on logit r, r clipped to [0.01, 0.99], C = 1e6) is fit on 200 distinct non-read drawn calls sampled
(seed 6301) from the OTHER fold, each with CAPE's checker label, and applied to this fold's reads. No label from the
scored fold and no label of any executed call is used. Then G62's quantities, pairs and bootstrap (seed 6302).

## Endpoint
- **L1 (primary).** For each of the two pairs (Claude's judge; GPT-6-luna's judge), observed - predicted: 95% interval
  inside ±0.02.
- Reported: calibration (mean Y - mean f).

## Readings, fixed in advance
- L1 passes: 200 labelled sampled actions from other tasks are enough for a deployer to predict a text monitor's loss
  against the agent's policy, with no label of the monitored tasks.
- L1 fails: the reader's error is not fixed by a monotone recalibration on other tasks.
