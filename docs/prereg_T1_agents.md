# Preregistration T1-agents: black-box sufficiency for agent violations on fresh CAPE episodes

Locked before any fresh episode was generated. The fresh episodes are job gb_fresh_minus / gb_fresh_plus, at the end
of the GPU chain.

## Claim
For an agent's policy violation at the start of a turn (a behaviour-defined label):
- (a) re-sampling the agent's turn forecasts the violation better than a trained probe on the agent's own hidden
  state (Proposition 3);
- (b) the agent's answers to the battery (grey box) are not worse than that probe by more than a small margin.

## Development evidence (scripts/17, results/t1/gemma4_clean.json; 5,493 clean units from seeds 6–9)
Within-(template × tool) AUROC: R16 0.909; WB L18 0.837; Ψ 0.859; TXT 0.868. R16 − WB_L18 = +0.072 [0.048, 0.096].

## Data (generated after this lock)
- Gemma-4-26B-A4B on CAPE-Bench core24b, CAPE's own harness (scripts/20 wrapper). x− seeds 40–43; x+ seeds 40–41.
- Units (scripts/15, episode mode, seed 0): 2,400 x− and 400 x+ episodes, sampled uniformly. Every clean turn is kept
  (post-reveal non-read actions up to and including the first violation). Selection depends only on the past.
- On the fresh units, the job chain runs: Ψ (scripts/01, battery v1), hidden states (scripts/02), and R16 (scripts/04,
  K = 16, CAPE's checker).

## Frozen monitors (fit on the development clean units; models/t1/frozen_monitors.pkl, hashed)
WB: PCA-256 + logistic on layer 18. Ψ: logistic on θ + log-odds. TXT: TF-IDF + logistic.

## Endpoints (scripts/25 stage=confirm; within-stratum AUROC on all fresh units; scenario bootstrap, 2,000 resamples)
- **F1:** upper 95% bound of AUROC(WB) − AUROC(R16) < +0.02.
- **F2:** AUROC(R16) − AUROC(WB) > 0, one-sided bootstrap p < 0.05.
- **F3:** lower 95% bound of AUROC(Ψ) − AUROC(WB) > −0.03.
- Claim (a) holds iff F1 and F2 pass. Claim (b) holds iff F3 passes.
- **Secondary (reported regardless):** each contrast; Brier scores after Platt recalibration on a random 20% of fresh
  units; TXT; the selection check (TXT − R16 on clean units vs on the outcome-selected subset of the same data).

## Reporting
All results are reported whatever their direction. Failures go to docs/PREREGISTRATION_RECORD.md with cause and fix.
