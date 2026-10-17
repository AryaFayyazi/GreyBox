# Preregistration T15: does the grey box replicate the agent's hidden state?

Locked before any T15 number exists. No map from reader features to agent states has been fit.

## Question
The project's original aim: read a black box's latent state from outside, replicate it, and use the replica for
safety. Proposition 5 says that an agent's internal state is a function of its tokens. So an open model reading the
same tokens may hold a replica of it. T15 measures how much of an agent's true hidden state a linear map recovers
from the reader's latent state, from the transcript alone. It also tests whether white-box readouts work on the
replica. We use open agents, where the true state is known.

## Data
- **Agent:** Gemma-4 (26B) on CAPE-Bench.
  - Development: 5,493 steps; fresh: 12,130 steps (T1, T4).
  - Only steps with a valid decision-point state are used (the `ok` masks).
- **Target:** the agent's hidden state at the decision point (after its reasoning, before the action): 5 layers
  (14, 18, 21, 24, 26) × 2,816 dimensions, standardized on development. Reduced to its top 128 principal components,
  fit on development; the share of variance they keep is reported.
- **Replica sources** (each from the transcript + proposed action only):
  - READER: Qwen3-8B layer-32 features (T4; the shipped monitor's layer).
  - TXT: TF-IDF of context and action (T9's features).

## Map
Ridge regression from the source to the 128 target components. Alpha ∈ {1, 10, 100, 1000, 10000} is chosen by
5-fold GroupKFold over scenario_id on development (mean R²). The map is fit on all development steps and applied to
the fresh steps.

## Endpoints (bootstrap over fresh scenario_id, 1,000 resamples)
- **E1, replication.** Variance-weighted R² of the reconstructed fresh states over the 128 components: R²(READER) −
  R²(TXT), lower 95% bound > 0. Seed 151.
- **E2, readouts work on the replica.** A probe is trained on TRUE development states (standardize → logistic, C =
  0.05, on the 128 components) to predict the unsafe label. It is applied (i) to the true fresh states and (ii) to the
  READER-reconstructed fresh states. Within-template AUROC (ii) − (i): lower 95% bound > −0.02 (non-inferiority).
  Seed 152.
- **Reported:**
  - R² values; R² per layer;
  - the share of variance kept by the 128 components;
  - E2 with TXT reconstructions;
  - (exploratory) the same map and probe applied to the closed agents' actions. These are Claude T8b and GPT T11, with
    reader features from scripts/78 and 81. This gives their pooled AUROC and catch rate "seen through Gemma's latent
    space".
- **Readings, fixed in advance.**
  - E1 and E2 pass: the reader's latent state is a replica of the agent's hidden state, from the transcript alone,
    good enough that white-box readouts work on it.
  - E1 passes, E2 fails: partial replication.
  - E1 fails: text features replicate the state as well as the reader.

Code: scripts/88_t15_latent_replication.py (CPU; threads pinned to 1 for the probes).
