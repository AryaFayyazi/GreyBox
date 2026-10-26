# Preregistration G74: the fair white-box comparison, powered

Locked, and published, before any G74 draw. Code: scripts/266_g74_fair_whitebox_powered.py. GPU; no API cost.

## Why
G71-B compared, at the start of the turn (the moment a closed API forecasts from), the label-free API geometry with a
white-box probe on the turn-start hidden state: 0.776 vs 0.799 (signed API forecast 0.797) on 217 Gemma-4 units, but the
interval [-0.16, 0.12] was too wide for non-inferiority. The probe that read the state after the agent's reasoning scored
0.997. G74 measures the fair comparison on enough units.

## Design (fixed)
T2's 12,130 fresh Gemma-4 units minus G68's: every violating unit (265) plus 600 safe units drawn from the same 15
templates (numpy seed 7400): 865 units, 394 scenarios, every template stratum holding both classes. For each unit, the
agent's own 8 sampled next actions from the exact turn-start context, each distinct drawn call read by its own 8 at-call
answers (4 each): scripts/260's stage_draws, unchanged (G46's code, the agent in its own seat). Scores: the geometric
forecast (G57/G70's statistic: no label, no question sign), the signed own forecast (G46's), T1's frozen turn-start
white-box probe (scripts/25: layer 18, fit on the separate development set), CAPE's probe after reasoning (I_dec) and 16
checker-labelled resamples at turn start (reported). AUROC within template (T2's strata); cluster bootstrap over
scenarios, 2000, seed 7401. Checks before the lock (CPU): units build; replaying each unit's prior calls reproduces the
logged label of its call on 200 of 200 sampled units.

## Endpoints
- **F1 (primary).** Geometric − turn-start white box: lower 95% bound > −0.05 (non-inferiority).
- **F2.** Signed own forecast − turn-start white box: lower 95% bound > −0.05.

## Readings, fixed in advance
- F1 (or F2) passes: at the same moment, a closed agent's API, with no weights and no labels, forecasts its violations
  within 0.05 of a white-box probe on its hidden state; the white box's larger lead (0.997) comes from reading the agent's
  private reasoning, which no outside monitor of a closed agent can see.
- Both fail: the white box at turn start keeps a real lead; reported with its size as the price of having no weights.
