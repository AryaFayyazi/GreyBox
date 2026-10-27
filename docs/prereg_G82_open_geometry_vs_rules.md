# G82: across task families, does the label-free geometry read sampled actions better than a task-agnostic rule?

Locked 2026-10-27 before any G82 draw. Code: scripts/281_g82_open_geometry_vs_rules.py (scripts/260's draw stage and
scripts/262's geometric statistic, unchanged). GPU only (no API cost).

## Why
Review rounds 18-21: the forecast's signal comes from sampling the agent's next actions, and simple rules on the samples
match or beat the geometry within Claude's one task family (G79 F3 failed: generic rule 0.93 vs geometry 0.90). The only
evidence that the geometry adds something is exploratory, on existing draws (scripts/272 on G70's draws, four open
agents across task families: geometry 0.75, task-agnostic rule 0.45, checker 0.79). G82 tests it prospectively.

## Design (fixed)
- Agents: Gemma-4, gpt-oss-20b, Qwen3.8-27B, Muse-Glimmer-30B (Muse under vLLM 0.30, G68's recorded setting).
- Units: CAPE-Bench decision points from stored episodes no earlier draw used (episodes outside G68's units; not G74's
  units; pools: Gemma-4 clean + fresh, gpt-oss-20b clean, Qwen3.8 and Muse-Glimmer T17 units). Per agent, up to 150
  violating units at random and 150 safe units from the templates that hold a violation (numpy seed 8200 + agent index).
  Resulting sets: Gemma-4 300 (150 violations, 14 templates), gpt-oss-20b 300 (150, 14), Qwen3.8 289 (139, 11),
  Muse-Glimmer 300 (150, 8).
- Draws: scripts/260's stage unchanged: 8 sampled next actions from the exact turn-start context; each distinct drawn
  call read by its 8 at-call answers x 4.
- Scores per unit, AUROC within template (clusters = scenarios): geometric (scripts/262's statistic: mean distance of the
  drawn calls' angles from the agent's median over its read draws; no label, no question sign); task-agnostic rule (share
  of draws that are not CAPE read tools); reported only: CAPE's checker on the draws (a label per drawn call), the
  signed forecast.

## Endpoints (cluster bootstrap over scenarios within each agent, 2,000 resamples, seed 8201)
- **P1 (primary).** Mean over the four agents of (geometric - task-agnostic rule): lower 95% bound > 0.
- **P2.** Mean geometric AUROC over the four agents: lower 95% bound > 0.5 (G70's G1, prospectively).

## Readings, fixed in advance
- P1 passes: across task families, where no single tool is risky, the label-free geometry reads an agent's sampled
  actions better than a task-agnostic rule; this is what the geometry adds to sampling.
- P1 fails: across families, too, the geometry adds nothing beyond a generic read of the samples.
