# Preregistration G36: triangulation at the agent's decision

Locked, and published, before any reader hidden state is computed. Code: scripts/209_g36_decision_triangulation.py.
Nothing is trained; the estimate for the closed agent uses only its own answers; open models' hidden states are
reference geometries and validation, never part of the method.

## Why
G34 checked, on generic contexts, how well a closed model's own answers reconstruct its own hidden geometry. The
monitor works at an agent's proposed call. This test moves the triangulation (Proposition, triangulation) to those
calls: how well does an agent's own answer geometry at its calls reconstruct its own decision-state geometry?

## Data (fixed)
G15's held-out calls: Gemma-4 (600), gpt-oss-20b (599; the call without a valid state dropped, as in G15), Qwen3.8
(373), Muse-Glimmer (600), Closed-A (495). Answer geometry: Theorem 5's debiased Fisher–Rao distances of the agent's
own 8 answers x 4 (G15's batteries). Own decision state (open agents only): the stored state, layer index 3 (cosine).
Reference geometries: G1's 8 open models (each at G1's development-chosen layer, its own chat template, last prompt
token) read each call's transcript and proposed action with scripts/69's reader prompt (cosine). Texts were rebuilt
before locking (2,667 calls; no hidden state or outcome computed).

## Estimator (fixed)
Per agent's call set: the readers' loadings lambda_r from the rank-1 minres fit of their pairwise Spearman matrix;
the agent's answer loading kappa = least squares of rho(answers, reader r) on lambda_r; for open agents, the own
decision-state loading lambda_own = least squares of rho(own state, reader r) on lambda_r. Predicted own agreement
rho(answers, own state) = kappa x median lambda_own of the other open agents (interval: their range). Closed-A:
kappa x median lambda_own of the 4 open agents (interval: their range).

## Endpoints (bootstrap over calls within each agent, 500, seed 3600)
- **V1 (primary; validation).** Leave each open agent out: mean absolute error of the predicted own agreement over
  the 4 open agents < 0.10.
- **V2.** The observed own agreement inside its predicted interval for >= 3 of the 4 open agents.
- **E1.** Closed-A's triangulated own agreement at its calls: lower 95% bound > 0.
- **Reported.** Per agent: observed and predicted own agreement, loadings, answers against each reader.

## Readings, fixed in advance
- V1 and V2 pass: at the calls the monitor scores, the triangulation predicts how well an unseen agent's answers
  reconstruct its decision-state geometry; E1 estimates it for the closed agent from its own answers alone.
- V1 fails: at the decision, a one-factor shared geometry does not predict the unseen agreement well enough; E1 is
  reported without a claim.

## Known before locking
G35's within-cell results (answers vs decision states, same template and label: about 0); G34's results; the
exploratory 145 (a Qwen3-8B reader's geometry vs agents' decision geometry over all pairs: 0.17–0.35). Not computed:
any agent's own answer geometry against its decision states over all pairs, and any reader of these 2,667 calls other
than Qwen3-8B's earlier features (not used here). A synthetic check (removed) verified the estimator.
