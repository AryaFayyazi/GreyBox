# Preregistration T5: agent internals vs a transcript reader, on unseen scenario families

Locked before any T4 or T5 result exists (T4's hidden stage has not started).

## Why
- CAPE's gate and every monitor in T1–T4 were trained on earlier episodes of the SAME scenarios they were tested on.
  A deployed monitor meets new kinds of tasks.
- T4 compares a transcript reader with the agent's internals in that same-scenario regime, with different probe
  pipelines.
- T5 makes the comparison fair and deployment-realistic: one identical probe pipeline for both feature sources, and
  test families never seen in training.

## Design
- **Features at the decision point:**
  - AGENT: Gemma-4's own residual stream at CAPE's point (layers 14/18/21/24/26);
  - READER: Qwen3-8B reading the transcript and the proposed action (T4's READER_act features).
- **Probe, identical for both:** per layer, standardize → PCA(256) → logistic (C=0.05); the score is the mean of the 5
  per-layer logits. No layer selection.
- **Family regime:** 5 folds over the 23 scenario families (sorted, round-robin). Train on development units of the
  other families; score the fresh units of the held-out families; pool over folds.
- **Same regime (reported):** train on all development units, score all fresh units.
- **References:** T3's zero-shot judge on the actual action (J_act); CAPE's frozen gate (I_dec).
- Code: scripts/70_t5_family_holdout.py.

## Endpoints (within-template AUROC on fresh units, scenario-cluster bootstrap, 1,000 resamples)
- **P1.** Family regime: AUROC(READER) − AUROC(AGENT) > −0.02: lower 95% bound > −0.02. A transcript reader is
  non-inferior to the agent's own internals on unseen task families.
- **P2.** Family regime, T3 subset: AUROC(AGENT) − AUROC(J_act), two-sided. Reading fixed in advance: agent probe
  better / judge better / no significant difference.
- **Reported:** same-regime READER − AGENT; family-regime READER − J_act; every AUROC.
