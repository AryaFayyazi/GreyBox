# Preregistration G27b: state tomography and forecasting, replicated on three more open agents

Locked, and published, before any answer of these batteries exists and before G27's result exists. Code:
scripts/197_g27b_replication.py (calls G27's locked scripts/194 unchanged for questions, frame, conversation prefix,
measurement, angles, identification and Kalman filter).

## Why
G27 tests, on Gemma-4, whether an agent's state estimated from its own answers at earlier steps forecasts its next
call's violation before the call is proposed. A method claim needs more than one agent. G27b runs the same test on the
three other open agents.

## Data (fixed)
gpt-oss-20b, Qwen3.8-27B, Muse-Glimmer-30B, each through its own vLLM API with its standard sampling; CAPE-Bench
held-out episodes of each (their G15 sets); min(150, available) episodes with a violating call and as many without
(seed 2700): gpt-oss 150+150 (2,042 steps), Qwen3.8 43+43 (750 steps; the fewest violations, the least power),
Muse-Glimmer 124+124 (2,384 steps). Every tool-call step up to and including the first violating call, at most 12.
G27's 8 forward-looking questions, 4 answers each, before each step's call.

## Estimator (G27's, unchanged)
Instrumental-variable identification from unlabelled trajectories, cross-fit by scenario (5 folds), Kalman filter with
the theory's noise; scores: forecast, filtered, raw, prior (the filter with no measurement).

## Endpoints (pooled: mean over the three agents of the per-agent difference; cluster bootstrap over scenarios within
each agent, 2000, seed 2790)
- **K1 (primary).** Forecast − prior, within template and tool, on calls after the first step: lower 95% bound > 0.
- **K2 (two-sided).** Filtered − raw, within template and tool, all calls.
- **Reported, per agent.** The four AUROCs; NIS/K; the noise that IV implies over the theory's; caught at a 5% review
  budget; G28's check (agreement of the forecast, filtered and raw geometries with the true decision-state geometry,
  layer index 3, on the steps whose true state is stored).

## Readings, fixed in advance
- K1 passes: the forecast from earlier answers carries information beyond the step index on the three other agents too.
- K1 fails: it does not replicate at this power (Qwen3.8 has 43 violating calls).

## Known before locking
G15 and G16 on these agents' held-out calls (call-level answers). No G27 or G27b answer exists. A dry run on random
answers (removed) checked that the code runs and gives no signal.
