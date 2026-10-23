# Preregistration G34: does a closed model's own answer geometry reconstruct its own hidden geometry? Triangulation

Locked, and published, before the quantities below are computed. Code: scripts/206_g34_triangulation.py. CPU only;
existing data; nothing is trained; the estimate for the closed model uses only its own sampled answers. Open models'
hidden states validate the estimator; they are never part of the method.

## Why
On open models we can check the reconstruction: a model's answer geometry tracks its hidden geometry (G1). On a
closed model we cannot, because its hidden states are not visible. G6 found that hidden geometries of different
models share a large common part (mean Spearman 0.56 between open models) and that a model's answers track other
models' hidden geometries as well as its own (excess +0.002). A one-factor model turns these facts into an estimator.

## Model (stated before the data)
On the vector of all context-pair distances: H_m = lambda_m C + e_m (hidden geometry of model m), A_a = kappa_a C + f_a
(answer geometry of answerer a), with idiosyncratic parts independent across models (G6: no own excess). Then
rho(H_a, H_b) = lambda_a lambda_b, rho(A_a, H_b) = kappa_a lambda_b, and the unseen rho(A_m, H_m) = kappa_m lambda_m.
lambda: rank-1 minres fit of the off-diagonal hidden-hidden correlations; kappa_a: least squares of rho(A_a, H_b) on
lambda_b over open b. The closed model's lambda_c is not identified: it takes the median open lambda, and the open
range gives an interval.

## Data (fixed)
G2's 192 confirm contexts and 24 questions. Closed-A: its 8 sampled answers per context and question (G2). Open
models (G1's 8): answers sampled to the same design (8 draws per context and question from the exact P(yes), seed
3400; Theorem 5's debiased distances); hidden states at G1's development-chosen layer.

## Endpoints
- **V1 (primary; validation).** Leave each open model out in turn, as if it were closed, and predict rho(A_m, H_m)
  from the other 7 hidden geometries. Mean absolute error over the 8 models < 0.10.
- **V2.** The observed rho(A_m, H_m) lies inside its predicted interval for >= 6 of the 8 models.
- **E1 (the estimate).** Closed-A's triangulated rho(A_c, H_c), with a bootstrap over contexts (500, seed 3401):
  lower 95% bound > 0.
- **Reported.** E2: whether the closed estimate lies within the open models' own range; kappa and lambdas; the
  own-excess residuals; the hidden-hidden fit residual; the same leave-one-out on G6's published matrices (1,000 fresh
  contexts, 88 exact answers).

## Readings, fixed in advance
- V1 and V2 pass: the triangulation predicts how well an unseen model's answers reconstruct its hidden geometry, to
  within 0.10. E1 then estimates this for the closed model, from its own answers alone.
- V1 fails: a one-factor shared geometry does not predict the unseen agreement well enough; E1 is reported without a
  claim.

## Known before locking
Closed-A's answers against open hidden geometries on these contexts (G2, reported: 0.45–0.51); G6's matrices; G1's
own-model agreements with 88 exact answers on G1's confirm split. Not computed: any quantity with the open models'
sampled answers on these 192 contexts and 24 questions, and any leave-one-out prediction. A synthetic check (exact
one-factor correlations; removed) verified that the code recovers kappa exactly.
