# Preregistration L2: black-box recovery of open LLMs' hidden states and concept readings (held-out contexts)

Written 2026-10-15 ~22:10 PDT. The 30% confirmation split of the L2 contexts (1,440 contexts, stratified by source,
seed 0; scripts/11) has never been analysed. Locked before the confirmation stage runs.

## Models and data
- 8 open chat models from 7 families: Qwen3-8B, Qwen2.5-7B, Llama-3.1-8B, Gemma-3-4B, Mistral-7B, Phi-4-mini,
  OLMo-3-7B, gpt-oss-20b.
- Battery v2 (88 items, sha 9e6c1a80…). Ψ from the logprob tier (θ + log-odds).
- True hidden state at the answer-start token: float32 (Gemma-3 re-extracted after an fp16 overflow).

## Development evidence (results/l2/l2_dev.json; 3,360 dev contexts)
- Hidden-state R² (top-64 PCs, mid layer), Ψ_self − text embedding: +0.123, 8/8 models.
- Own sentiment-probe logit, R²: Ψ_self − mean single other model's Ψ = +0.053, 7/8 models (vs the best single
  other: +0.025, 6/8). Ψ_self − text: about +0.45.
- Not claimed: raw-variance self-specificity. Ψ_self and the other models' averaged Ψ tie on hidden PCs.

## Frozen objects (fit only on dev; models/l2/frozen_<model>.npz, hashed)
For each model:
- the PCA frame of the mid-layer hidden state;
- ridge decoders Ψ_self / text / text+Ψ_self / text+Ψ_others-mean → the hidden PCs;
- the sentiment white-box probe (logistic on dev SST-2 contexts);
- ridge decoders Ψ_self / text / each single other model's Ψ → that probe's logit.

## Endpoints (scripts/30 stage=confirm; context bootstrap, 2,000 resamples; Holm over H1–H3, α = 0.05)
- **H1:** mean over models of [R²(Ψ_self) − R²(text)] on the hidden PCs > 0.
- **H2 (self-specificity, budget-matched):** mean over models of [R²(Ψ_self → own sentiment-probe logit) − mean over
  single other models o of R²(Ψ_o → the same logit)] > 0.
- **H3:** mean over models of [R²(Ψ_self) − R²(text)] on the sentiment-probe logit > 0.
- Secondary (reported regardless): per-model values, the best-single-other comparison, and Ψ_others-mean on the
  hidden PCs.

## Deviation note
An earlier draft of scripts/30 defined H2 on raw hidden-variance self-specificity. That definition was replaced
before any confirmation data were analysed, because the development data showed no raw-variance self-specificity
(above). The change is recorded in docs/research_ledger.md (2026-10-15 ~21:20).
