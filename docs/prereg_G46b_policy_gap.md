# Preregistration G46b (addendum to G46): does the theory predict, in size, how much another model loses?

Locked, and published, before any of Gemma-4's draws exist and before any analysis of the agent's draws. Code:
scripts/230_g46b_policy_gap.py (reads G46's outputs; uses G46's locked code). No extra data.

## Why
G44 found that another model in GPT-6-luna's seat forecast worse (+0.111); on G45's fresh GPT-6-luna episodes it forecast
as well (0.759 vs 0.762 within template and tool; seen before this lock while G45's locked analysis waits for its Claude
half). The proposition "the forecast needs the agent" does not say that another model always loses: it says the loss is
E[(sum_a (pi_bar - pi')(a | V) r(a, V))^2], the squared gap between the two policies' violation rates. G46's draws give
both policies' violation rates at every state, with r known (CAPE's checker on each drawn call). So the theory makes an
exact prediction that can fail.

## Quantities (per unit; K = 8 draws each)
- A unit is an executed non-read call, so its label Y is conditioned on the action being a non-read call. So each policy's
  violation rate is taken among its non-read draws: f_own = unsafe share among the agent's (Claude Haiku 5.5's) k_own
  non-read draws (of 8); f_oth = the same among Gemma-4's non-read draws in its seat. Units need at least 2 non-read draws
  from each model (k_own, k_oth >= 2).
- Observed excess D_obs = mean (Y − f_oth)^2 − mean (Y − f_own)^2.
- Predicted excess D_pred = mean (f_own − f_oth)^2 − 2 mean f_own (1 − f_own)/(k_own − 1). Derivation: the agent's draws and
  its actual action come from the same policy at the same state (the state is a function of the tokens; the draws use the
  loop's settings), so E[(Y − f_own)^2] = E[(Y − f)^2] + E f(1 − f)/k_own and E[(f_own − f_oth)^2] = E[(f − f_oth)^2] + E f(1 − f)/k_own,
  with f(1 − f)/k estimated without bias by f_own(1 − f_own)/(k_own − 1); f = P(unsafe | V, a non-read call).

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 4610)
- **B3 (primary).** D_obs − D_pred: 95% interval inside [−0.02, +0.02] (the theory predicts the other model's loss).
- **B2.** The policy gap G = mean (f_own − f_oth)^2 − mean f_own(1 − f_own)/(k_own − 1) − mean f_oth(1 − f_oth)/(k_oth − 1),
  an unbiased estimate of E(f − f')^2 with both sampling noises removed: lower 95% bound > 0 (the two policies differ, so the
  forecast needs the agent). (D_pred includes the other model's own sampling noise, so D_pred > 0 alone does not show this;
  a simulation before this lock confirmed E[D_obs] = E[D_pred] with equal and with different policies.)
- **Reported.** D_obs, D_pred, G, mean f_own, mean f_oth, mean Y.

## Readings, fixed in advance
- B3 passes: the proposition predicts, in size, how much another model loses before the call; where the policies agree
  (as G45 suggests for Gemma-4 and GPT-6-luna) it loses nothing, and where they differ it loses the predicted amount.
- B3 passes and B2 passes: on Claude Haiku 5.5, the forecast needs the agent, by the predicted amount.
- B3 fails: the policy-gap account does not predict the other model's loss; the paper says so.
