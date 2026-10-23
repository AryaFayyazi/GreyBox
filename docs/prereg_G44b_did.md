# Preregistration G44b: is the agent's advantage specific to forecasting, as the theory predicts?

Locked, and published, while G44's other model is answering and before any G44 outcome exists. Code:
scripts/221_g44b_did.py. CPU only.

## Why
If G44 finds that an agent's own answers forecast its next violation better than another model's, a reader could say the
other model is simply a weaker reader. The proposition "the forecast needs the agent" predicts something sharper: at the
call, where the action is in the transcript, any capable reader matches the agent (G16 found this for the same other
model and the same agents); before the call, only the agent's own policy gives the Bayes forecast. So the agent's
advantage should appear before the call and not at it.

## Design (fixed)
Open agents gpt-oss-20b, Qwen3.8, Muse-Glimmer; the same other model (Gemma-4) answering in the agent's seat in both:
- at the call: G16's data (G15's held-out calls, 8 questions with the call shown, within-template AUROC; G16's code);
- before the call: G44's data (steps in template x tool x step strata holding both classes, 8 forward questions).
DiD_a = (own − other) before the call − (own − other) at the call.

## Endpoint (independent cluster bootstraps over scenarios in each data set, 2000, seed 4450)
- **D1.** Pooled DiD (mean over the three agents): lower 95% bound > 0.
- **Reported.** Per agent: the two gaps and their difference.

## Readings, fixed in advance
- D1 passes: the agent's advantage over another reader is specific to forecasting its own next action, as the theory
  predicts: it reflects the agent's own policy, not a weaker reader.
- D1 fails: the gap before the call is not larger than at the call at this power.

## Known before locking
G16's at-call results (own − other: +0.023, +0.062, −0.040). No G44 answer of the other model has been read.
