# Preregistration G70: forecasting from the reconstructed geometry on four open agents, against a white-box probe

Locked, and published, before this analysis of G68's data. Code: scripts/262_g70_open_geometry_vs_whitebox.py. CPU only.

## Why
The geometry-only forecast (G57: a sampled next action's Fisher-Rao distance from the agent's typical state, no label,
no question sign) forecast Claude Haiku 5.5's violations (0.892; replicated 0.866 in G66). It has not been tested beyond
Claude's one task family, and never against white-box access. G68 collected, on four open agents across their task
families, each agent's 8 sampled next actions read by its own at-call answers; that file has not been analysed with this
statistic. CAPE stored the agents' true hidden states at the decision step for part of these units.

## Design (fixed)
Units and draws: G68's (692 units; Gemma-4, gpt-oss-20b, Qwen3.8, Muse-Glimmer). Statistic G: G57's, unchanged
(scripts/242's theta; the agent's own centre = coordinate-wise median over all its drawn calls).
White box W: CAPE's stored decision state at the same step (layer 3), scripts/171's probe (PCA(32), logistic regression
C=0.1) fit on the agent's development units with every scenario among G68's units removed (4,544 / 2,789 / 2,286 / 2,527
training units). An agent enters W if its units with a stored state hold >= 15 violating and >= 15 safe (by this rule:
Gemma-4 and gpt-oss-20b; Qwen3.8 has 5 safe, Muse-Glimmer 7). AUROC within template|tool|step (G68's strata); cluster
bootstrap over scenarios within each agent, 2000 resamples, seed 7000.
Checks before the lock: the code reproduces G57's published Claude value (0.8916) on G46c's file; the probe fits on all
four agents (sizes only printed, no test score seen).

## Endpoints
- **G1 (primary).** Mean over the 4 agents of the geometric forecast's AUROC: lower 95% bound > 0.5.
- **W1.** Mean over eligible agents of (geometric − white-box probe), on units with a stored state: lower 95% bound > −0.10
  (non-inferiority).
- Holm over G1 and W1. Reported: each agent's geometric, signed own forecast, self-report and probe values, the number of
  agents whose geometric lower bound exceeds 0.5.

## Readings, fixed in advance
- G1 passes: the reconstructed geometry, with no label and no sign, forecasts violations on four open agents across their
  task families, as on Claude; geometry-only forecasting holds beyond one template and one agent.
- W1 passes: on the agents where it can be tested, the API geometry forecasts within 0.10 of a probe on the true hidden
  state. W1 fails: white-box access forecasts better; reported as the cost of having no weights.
