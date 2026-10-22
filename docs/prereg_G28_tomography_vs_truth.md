# Preregistration G28: does the state estimated from the agent's answers track its TRUE hidden state?

Locked, and published, before G27's battery exists. Code: scripts/195_g28_tomography_vs_truth.py.

## Why
G27 estimates an agent's state from its own answers (a Kalman filter over the steps of an episode, identified with no
label). For an open agent the true hidden state is known, so the estimate can be checked against it. The true states
are used only to validate; the estimator never sees them.

## Data
G27's steps of Gemma-4 whose true decision state is stored (G14/T17 decision states; layer index 3, the layer of
G15's white-box probe): about 916 steps.

## Endpoints (geometry: Spearman between all pairwise Euclidean distances; no fitted map; cluster bootstrap over
scenarios, 1000)
- **T1 (primary).** Filtered estimate − raw answers, in agreement with the true geometry: lower 95% bound > 0.
- **T2.** The forecast (made before the step, from earlier steps only) agrees with the true geometry: lower 95%
  bound > 0.
- **Reported.** The three agreements on all such steps and on steps after the first.

## Readings, fixed in advance
- T1 passes: combining the agent's answers over its trajectory, through the identified dynamics, recovers more of its
  true hidden geometry than its answers at one step.
- T2 passes: the agent's next hidden state is predictable, in its geometry, from its earlier answers alone.

## Known before locking
G1 and G6 (answer geometry tracks hidden geometry on contexts); G15's decision-state probes. No G27 answer exists.
