# Preregistration G71: the fair white box, the slice the answers estimate, and a situation-battery monitor

Locked, and published, before any of these analyses is computed. Code: scripts/263_g71_fair_whitebox_and_slice.py. CPU.

## Why
G70 found a white-box probe forecasting 0.16 better than the API geometry. CAPE's stored decision state, which that probe
read, sits AFTER the agent's own hidden reasoning for the turn (scripts/96 decision_contexts; scripts/60), while a closed
API forecasts from the start of the turn; T2 already measured +0.138 for reading after the reasoning. The fair white box
reads the turn-start state. Separately, Theorem 1(c) says a battery estimates the hidden geometry only in the metric it
pulls back (a slice of rank <= its size), which predicts G36's weak full-state agreement (0.01-0.16). And G43 (32
situation questions, GPU, queued) can test whether a broader battery's geometry alone flags violations at the call.

## Design (fixed; details in the code)
- **B.** G68's Gemma-4 units with T2's turn-start states (217 units, 81 violations). Geometric forecast: G70's, unchanged.
  Fair white box: T1's frozen turn-start probe (scripts/25; layer 18; fit on the separate development set). Reported on the
  same units: CAPE's probe after reasoning (I_dec), 16 checker-labelled resamples at turn start (R16_ts), the agent's
  signed own forecast and self-report. AUROC within template|tool|step; cluster bootstrap over scenarios, 2000, seed 7100.
- **D.** G36's tables (G15's held-out calls; 600/599/373/600 calls). Ridge (alphas 1e-2..1e4) from PCA(64) of the decision
  state (layer 3; PCA fit on all of the agent's stored states, no labels) to the 8 answer angles, cross-fit by scenario
  (5 folds). Spearman of the answer geometry (Theorem 5's debiased distances) with (i) the predicted-angle geometry, (ii)
  the full-state cosine geometry, (iii) the mean over 20 random rank-8 linear maps of the same PCA(64) state.
  Bootstrap over calls, 200, seed 7101. No label used.
- **A.** G43's answers (when all 4 agents exist): G56's statistic on the 32 situation angles (distance from the agent's
  median), AUROC within template; reported: the 8-question and 40-question versions. Cluster bootstrap over scenarios,
  2000, seed 7102.
Checks before the lock: the loaders reproduce T2's published within-template turn-start values (probe 0.8413, R16 0.9038)
on T2's units, and G36's published own agreement on all 4 agents (0.0098, 0.0386, 0.0543, 0.161). No endpoint computed.

## Endpoints
- **B1.** Geometric − fair (turn-start) white box, Gemma-4: lower 95% bound > −0.05 (non-inferiority).
- **D1.** Slice − full-state agreement: lower 95% bound > 0 on at least 3 of 4 agents.
- **D2.** Slice − random rank-8 null: lower 95% bound > 0 on at least 3 of 4 agents.
- **A1.** Mean over agents of the 32-question geometric monitor: lower 95% bound > 0.5.
B1 and A1 are Holm-corrected together when A1 exists; D1, D2 are count endpoints over per-agent intervals.

## Readings, fixed in advance
- B1 passes: at the same moment (the start of the turn), the label-free API geometry forecasts within 0.05 of a white-box
  probe; G70's gap is the reasoning the probe read, not the weights. B1 fails: the white box at turn start is better; the
  price of having no weights is real and reported with its size.
- D1 and D2 pass: the 8 answers estimate a linear slice of the decision state (the pulled-back metric), as Theorem 1(c)
  predicts, not the full state; G36's weak full-state agreement is the expected result. Either fails: reported as is.
- A1 passes: a broader battery's geometry alone, with no label and no sign, flags violations at the call.
