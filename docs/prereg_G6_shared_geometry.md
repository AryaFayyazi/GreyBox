# Preregistration G6: the observable geometry is shared across models, and it generalizes to new kinds of context

Locked, and published, before any G6 model output exists. Code: scripts/136_g6_contexts.py (contexts, built),
scripts/137_g6_shared_geometry.py (data and analysis; it reuses scripts/132's generation code and scripts/129's
geometry code). No model is trained or fine-tuned.

## Why
An independent review of our paper found, and we verified on the G1 confirm contexts, that a model's answer geometry
matches OTHER models' hidden geometries as well as its own:
- the own-model diagonal of the 8×8 matrix ρ(answers of A, hidden of B) has no excess over row + column effects
  (−0.002);
- no question category shows an excess, including questions about the model's own behaviour;
- another open model's hidden geometry matches a model's hidden geometry better than that model's own answers do.

The hypothesis: **what outputs reveal of a model's state is the geometry that models share.** This is also the
measured reason why an open model can stand in for a closed one. G6 tests it on fresh contexts, including four kinds
of context never used in this project.

Known before locking: all earlier results (G1–G5 pass); the exploratory analyses above (G1 confirm contexts); a dry
run of this analysis on G4's 480 development contexts, with two old kinds standing in for new ones
(results/t1/g6_shared_geometry_DRYRUN_dev.json): S1 −0.0016 [−0.0069, +0.0004]; S2 5 of 7 (gpt-oss and Mistral
intervals include 0); S3 8 of 8. The G6 contexts exist (1,000; 0 overlap with L2), but no model has seen them.

## Design
- **Contexts** (scripts/136, numpy seed 6): 100 per kind.
  - Fresh items of L2's six kinds: SST-2 reviews; LLM-LAT harmful and benign requests; BoolQ validation (L2 used
    train); GSM8K test (L2 used train); WikiText-103 validation (L2 used train).
  - Four new kinds: AG News articles (test); MBPP programming tasks (test); JailbreakBench behaviours (harmful and
    benign goals); deepset prompt injections (injections and benign prompts).
- **Models:** the 8 open chat models of G1, no system message. One pass per model with scripts/132's code: top-20
  log-probabilities after each of the 88 battery-v2 questions (vLLM seed 0); 8 replies (temperature 1, 128 tokens;
  seed 1); hidden states at L2's 6 layers (HF, bf16).
- **Layer:** G1's dev-chosen layer per model (fixed; no new choice).
- **Geometries:** answers = Fisher–Rao over 88 questions, exact; hidden = 1 − cosine; text = 1 − cosine of
  all-MiniLM-L6-v2 embeddings of the contexts; replies = 1 − cosine of the mean embedding of 8 replies.
- **Statistic:** Spearman over all pairs of contexts; bootstrap over contexts (500).

## Endpoints
- **S1 (no own-model advantage: the observable geometry is shared).** The 8×8 matrix R[A, B] = ρ(answers of A,
  hidden of B). Diagonal excess = mean over A of (R − row means − column means + grand mean)[A, A]. Bootstrap
  (seed 1360). Pass (equivalence): the 95% interval lies inside [−0.02, +0.02].
- **S2 (an open model's hidden geometry is the better stand-in).** For each model A other than Qwen3-8B (the reader
  of our monitors): ρ(Qwen3-8B hidden, A hidden) − ρ(A answers, A hidden). Seeds 1370 + i. Pass: lower 95% bound > 0
  for at least 5 of the 7 models.
- **S3 (the answer geometry generalizes to new kinds of context).** On the 400 contexts of the four new kinds:
  ρ(A answers, A hidden) − ρ(text, A hidden). Seeds 1380 + i. Pass: lower 95% bound > 0 for at least 6 of 8.
- **Reported:** the full 8×8 matrix; hidden-vs-hidden agreement; replies; per-kind agreement.

## Readings, fixed in advance
- **S1–S3 pass:**
  - A model's outputs reveal the geometry that models share, not a model-specific geometry. This holds on fresh and on
    new kinds of context.
  - The shared geometry is better estimated by an open model's hidden state than by the model's own answers. So
    reading a closed model through an open model is not an assumption but a measured property.
  - The answer geometry remains the training-free, output-only measurement that shows it.
- **S1 fails, with a positive excess:** the answers do carry a model-specific geometry, and the shared-geometry claim
  is withdrawn.
- **S2 fails:** an open model's state is not a better stand-in than the model's own answers; the paper says so.
- **S3 fails:** the answer geometry does not transfer to new kinds of context.

## Limits stated in advance
- Eight open models, 7–20B parameters, one layer each. Claude cannot be tested on fresh contexts: no API budget is
  left. For Claude the paper reports the G2 data only.
- "Shared" is measured at the level of geometry over these contexts. Models still differ in their answers (levels of
  P(yes)), and those differences are not tested here.
