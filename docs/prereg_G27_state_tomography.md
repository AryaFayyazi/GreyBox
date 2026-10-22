# Preregistration G27: state tomography and forecasting of an agent through its own answers

Locked, and published, before any answer of this battery exists. Code: scripts/194_g27_state_tomography.py.

## Why
The method should estimate the hidden state of an agent from its API outputs alone, with no other model and no
training, and use the estimate to predict, forecast and prevent violations. Theorem 1 gives the measurement model: a
yes/no answer law is a measurement of the state; its arcsine angle has a sampling noise fixed by the number of answers.
G27 adds the dynamics: across the steps of an episode, the agent's observable state follows a linear-Gaussian
state-space model, identified from unlabelled trajectories (instrumental variables, consistent whatever the measurement
noise), filtered by a Kalman filter, and used to forecast the next call before the agent proposes it.

## Measurements (fixed; no label is used)
- Agent: Gemma-4 (26B-A4B), its own vLLM API, thinking on, its standard sampling.
- Episodes: CAPE-Bench held-out (G15's "fresh" set): 150 with a violating call and 150 without (seed 2700); every
  tool-call step, up to and including the first violating call, at most 12 steps: 2,505 steps.
- At each step, BEFORE the step's call, in a side branch of the agent's conversation: 8 forward-looking yes/no
  questions about its next action (scripts/194 QUESTIONS, one per kind of CAPE-Bench violation), 4 answers each.

## The estimator (fixed)
y_t = arcsine angles; R_t = diag(1/(m_tj + 1/2)) (theory). z_{t+1} = F z_t + c + w; y_t = z_t + v. F by the lag-2
instrumental-variable estimator; Sigma_z = F^-1 Cov(y_{t+1}, y_t); Q, P0 projected to the PSD cone; cross-fit by
scenario (5 folds: the model of each episode is identified on the other folds). Scores at each call (the fixed sign
vector of the questions): forecast sigma . E[z_t | y_<t] (before the call, from earlier steps only); filtered
sigma . E[z_t | y_<=t]; raw sigma . y_t; prior: the same filter with no measurement (a function of t only).
Statistic: AUROC within strata of template and tool; cluster bootstrap over scenarios (2000).

## Endpoints
- **K1 (primary).** On calls after the first step, forecast − prior (the information that the agent's earlier answers
  add to a forecast that knows only the step index): lower 95% bound > 0.
- **K2 (two-sided).** Filtered − raw on all calls (does the filter improve on the step's own answers?).
- **K3 (reported; model check).** Mean normalized innovation squared / K (1 if the model is calibrated); the noise
  that IV implies over the theory's noise, per question.
- **Reported.** Forecast, filtered, raw and prior AUROCs; caught at a 5% review budget of steps.

## Readings, fixed in advance
- K1 passes: the agent's state, estimated from its own answers at earlier steps, forecasts its violation before the
  call is proposed, beyond the step index; a gate can act before the call.
- K1 fails: at this power the estimated trajectory does not forecast the next call beyond the step index.
- K3 near 1 supports the measurement model; far from 1 says where it fails.

## Known before locking
G15 (call-level answers of the same episodes' calls, with the call shown). A dry run on random answers (removed) showed
that a forecast score can pass on the step index alone; K1 is therefore stated against the prior. On synthetic
linear-Gaussian systems with binomial answers (n = 4) the IV estimator recovered F (0.81 for 0.80; 0.93 for 0.95)
where the uncorrected regression gave 0.26 for 0.80.
