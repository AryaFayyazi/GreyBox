# Preregistration G57: forecasting by the reconstructed geometry alone

Locked, and published, before this analysis has been run. Code: scripts/242_g57_geometric_forecast.py. No API, no GPU.

## Why
G56 showed that a call's Fisher-Rao distance from the agent's typical state flags violations with no label and no sign.
G57 asks whether the same geometry forecasts the coming violation before the call: the agent's own sampled next actions,
placed in the geometry, should lie far from its typical state when it is about to violate.

## Design (fixed)
Units and draws: G46c's (Claude Haiku 5.5, exact contexts) and G48's (GPT-6-luna). Each distinct drawn call's 8 at-call
angles (4 answers each); centre = the median of all the agent's drawn calls (G56's); geometric forecast = mean over the
unit's 8 draws of the draw's distance from the centre (a draw that is not a call scores 0). No label, no sign.

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 5700)
- **G1 (primary, each agent).** Geometric forecast, AUROC within template|tool|position: lower 95% bound > 0.50, for
  Claude Haiku 5.5 and for GPT-6-luna.

## Readings, fixed in advance
- G1 passes for both: the reconstructed geometry of the agent's own sampled policy forecasts its coming violation, beyond
  the call's step, with no label and no sign.
