# Preregistration G64: a label-free, rank-based test of the two regimes

Locked, and published, before this analysis. Code: scripts/253_g64_text_policy_agreement.py. CPU; existing data.

## Why
G60 E2 failed: the label-free Brier gap between a text monitor and the agent's policy did not separate Claude Haiku 5.5
from GPT-6-luna, because Brier mixes ranking with calibration and GPT-6-luna's judge is poorly calibrated. The regimes
differ in ranking. A rank-based, label-free measure: within each step, how well the text monitor's ranking agrees with the
agent's own sampled policy. An exploratory version (self-report vs policy) gave -0.01 vs +0.25; this test uses the
reasoning text monitors instead and is fixed before any value is computed.

## Data (fixed)
Claude Haiku 5.5: G46c's units, G52's judge (q), own-draws forecast (f). GPT-6-luna: G48's units, G54's judge, own-draws
forecast. Agreement = Spearman correlation of q and f within each template|tool|position stratum with at least 5 units,
averaged with weights equal to stratum size. No label is used.

## Endpoint (cluster bootstrap over scenarios, 2000 resamples, seed 6400)
- **A1 (primary).** Agreement, GPT-6-luna - Claude Haiku 5.5: lower 95% bound > 0.

## Readings, fixed in advance
- A1 passes: with no label, a deployer can tell that a text monitor tracks GPT-6-luna's policy but not Claude Haiku 5.5's.
- A1 fails: this label-free check does not separate the two agents.
