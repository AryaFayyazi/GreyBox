# Addendum 1 to preregistration T1-agents (dated 2026-10-15 ~20:40 PDT, before any fresh episode exists)

## Why
After the original lock, development analysis (exploratory, clean units) found the two black-box signals strongly
complementary. AUROC: Ψ + 2 rollouts 0.905, matching R16 (0.908); Ψ + 4 rollouts 0.919; the best monitor overall is
Ψ + R16 (0.943). Ψ + 4 rollouts beats the strongest reader probe (Qwen3.8-27B, 0.861) by +0.058 [0.038, 0.079]. We add
confirmatory endpoints for these cheap black-box monitors.

## What changes
- **New frozen objects:** models/t1/frozen_combos_addendum1.pkl. Logistic regression on [Ψ features, logit R_K], with
  R_K = (k + 0.5)/(K + 1) from the FIRST K of the 16 rollouts, fit on the development clean units, K ∈ {2, 4}.
- **scripts/25_t1_confirm.py:** new code computes F4 and F5 when the addendum file exists. The code for F1–F3, the
  data and the frozen original monitors are unchanged. The original lock keeps its hashes; this addendum locks the
  edited script and the new file.
- **New endpoints** (same data, AUROC and bootstrap as F1–F3):
  - F4: AUROC(R4 + Ψ) − AUROC(WB) > 0, one-sided bootstrap p < 0.05 (four black-box rollouts plus answers beat the
    agent's own white-box probe).
  - F5: lower 95% bound of AUROC(R2 + Ψ) − AUROC(R16) > −0.02 (two rollouts plus answers are not worse than sixteen
    rollouts).
- F4 and F5 are a separate family from F1–F3, with no Holm correction across families (each family is pre-specified).
- The 27B reader probe is not part of the fresh confirmation. Its fresh hidden states are not in the job chain. It
  stays a development comparison, reported as such.
