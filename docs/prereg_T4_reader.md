# Preregistration T4: does a reader trained on the same labels match the agent's own internals?

Locked before any T4 hidden state exists.

## Why
- T3 showed CAPE's probe (0.976) beats a zero-shot judge on the actual action (0.910). But the probe was trained on
  labelled examples and the judge was not.
- The fair black-box comparison is an open reader model, trained on the same kind of labels, that reads only what a
  deployment of a closed agent can see.
- If it matches the probe, the lever is labels (amortizing the label function), not access to the agent's internals.
  If it falls short, the agent's internals carry information beyond its transcript.

## Design
- **Reader:** Qwen3-8B (another family than the Gemma-4 agent). It reads T3's judge context (operating instructions,
  task, earlier tool calls with results) plus the proposed action, as a review question.
- **Probe:** a linear probe (standardize → PCA 256 → logistic, C=0.05) on the reader's residual stream at the last
  prompt token, at the layers at 50/62/72/82/90% depth. The layer is chosen on the development units only, by
  scenario-grouped 5-fold CV.
- **Training:** the development clean units (gemma4_clean, 5,493: earlier episodes of the same scenarios). This is the
  same regime as every frozen monitor in T1/T2 and as CAPE's gate.
- **Test:** once on the 12,130 fresh units. The agent's own internals are never used by the reader.
- **Variants:** READER_act (the action alone); READER_cot (the action plus the agent's reasoning for that turn,
  truncated to its last 4,000 characters).
- Code: scripts/69_t4_reader_action.py (stages hidden, analyze).

## Endpoints (within-template AUROC on fresh units, scenario-cluster bootstrap, 1,000 resamples)
- **M1.** AUROC(READER_act) − AUROC(I_dec, CAPE's frozen probe at the decision point) > −0.02: lower 95% bound
  > −0.02 (non-inferiority).
- **M2.** AUROC(READER_act) − AUROC(J_act, T3's zero-shot judge) > 0 on the T3 subset: lower bound > 0. Amortizing
  the label function beats a zero-shot judge.
- **Reported:** READER_cot − I_dec; the dev CV AUROC per layer.
