# Preregistration G61: across steps, do the agent's sampled actions add to text and to the step?

Locked, and published, before this analysis. Code: scripts/250_g61_draws_add_to_text.py. CPU; existing data.

## Why
A reviewer noted that on Claude Haiku 5.5, across steps, the step alone ranks violations at 0.82 and GPT-6-luna's text
judge at 0.90, above the sampled actions (0.80); the sampled actions win only within a step. The question that matters for
a deployer is whether sampling adds to what text (or the step) already gives, with the step not controlled.

## Data (fixed)
G46c's units (Claude Haiku 5.5, exact contexts, 1,587 calls, 316 violations). Scores: the own-draws forecast (G46's
forecast on G46c's own stage), G52's Claude text judge, G55's GPT-6-luna text judge, and the unit's position. A missing
score is set to that score's median. Combination with no fit and no label: the mean of the two scores' ranks over all
units.

## Endpoints (AUROC within template|tool, i.e. across steps; cluster bootstrap over scenarios, 2000 resamples, seed 6100)
- **A1 (primary).** GPT-6-luna judge + draws − GPT-6-luna judge alone: lower 95% bound > 0.
- **A2.** Claude judge + draws − Claude judge alone: lower 95% bound > 0.
- **A3.** Position + draws − position alone: lower 95% bound > 0.

## Readings, fixed in advance
- A1 passes: even across steps, the agent's sampled actions add to the best text monitor; what text misses is the policy.
- A1 fails: across steps, the best text monitor already holds what the draws add; the draws' advantage is only within a step.
