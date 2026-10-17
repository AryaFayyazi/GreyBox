# Preregistration T2: when do internals beat the black box? (one theorem, two moments)

Locked before any decision-point hidden state or decision-point rollout of the fresh units exists.

## Question
- CAPE ("The Agent Knew") found a linear readout of the agent's residual stream at the decision point forecasts a
  violation better than text and CoT monitors (AUROC 0.94).
- T1 found that at the turn start, 16 black-box resamples beat a white-box probe (0.904 vs 0.841, preregistered).
- Theorem T1 predicts both: at any moment, no readout of the state can beat the forecast obtained by resampling the
  model from that same moment. So CAPE's advantage should come from reading LATER, after the agent's own reasoning has
  been sampled, not from internals as such. At the decision point, resampling the action alone should match CAPE's
  readout.

## Data and frozen objects
- Units: the 12,130 fresh clean units of the T1 confirmation (346 unsafe), unchanged.
- Turn start, already computed and frozen: R16_ts (16 whole-turn API resamples), WB_ts and TXT_ts (the frozen T1
  monitors).
- Decision point:
  - context = the turn-start prompt + the agent's own reasoning for this turn (Gemma tags stripped) + "\n<channel|>".
    This is CAPE's point: the last token before the turn's tool call.
  - I_dec: CAPE's frozen gate, outputs/v54/gate_gemma4.npz (layers 14/18/21/24/26), applied unchanged.
  - R16_dec: 16 vLLM resamples of only the action from that context (temperature 1.0, top_p 0.95, top_k 64, as in
    CAPE). Each is labelled by the same environment rule checker as R16_ts (scripts/04 label). A final answer or an
    unparseable call counts as safe, as in R16_ts.
- Code: scripts/60_decision_point.py (stages hidden, roll, analyze).

## Endpoints (within-template AUROC, scenario-cluster bootstrap, 1,000 resamples, as in T1)
- **G1.** AUROC(I_dec) − AUROC(R16_ts) > 0: lower 95% bound > 0. Reading internals later beats resampling earlier;
  this is CAPE's advantage.
- **G2.** AUROC(R16_dec) − AUROC(I_dec) > −0.02: lower 95% bound > −0.02. At the same moment, black-box resampling is
  non-inferior to CAPE's readout; this is T1 at the decision point.
- **G3.** AUROC(I_dec) − AUROC(WB_ts) > 0: lower 95% bound > 0. The same kind of readout gains from reading later.
- **Reported:** a supervised CoT text monitor at the decision point (TF-IDF on transcript + reasoning, out of fold
  by scenario) minus I_dec; CAPE gate fire rates per class at its frozen tau; units dropped for empty reasoning.

## Interpretation fixed in advance
- G1 and G2 passing: CAPE's internal advantage is a timing advantage. A black box with access to the agent's
  reasoning can match it, and without that access the best is turn-start resampling.
- G2 failing: internals carry information beyond 16 resamples at the same moment. This would contradict a
  finite-sample reading of T1 and is reported as such.
