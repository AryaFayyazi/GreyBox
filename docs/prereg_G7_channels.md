# Preregistration G7: what each output channel reveals of a model's own state

Locked, and published, before any G7 model output on the fresh contexts exists. Theory: docs/theory_channels.md.
Code: scripts/138_g7_channels.py (contexts built; data and analysis), reusing scripts/132 (answers) and scripts/129
(geometry). No model is trained or fine-tuned.

## Why
- G6 found that a model's yes/no answers reveal the geometry that models share, with no own-model excess.
- Theorem 6 says that full log-probabilities reveal the model's own final geometry exactly.
- So what crosses the API boundary depends on the channel. G7 measures the ladder on fresh contexts:
  - yes/no answers in side branches;
  - the next-token law at the first content position, reduced to its top 1, 5 or 20 tokens plus "other" (APIs
    return up to 20);
  - the full next-token law (Fisher–Rao);
  - the centred log-probability vectors (cosine), the linear image of the final state that Theorem 6 uses.

Known before locking:
- every earlier result (G1–G6 pass);
- an exploratory look at L2's stored top-20 log-probabilities, on the G1 confirm contexts;
- two dry runs on 600 L2 development contexts (100 per kind; none in G1's layer-selection set):
  results/t1/g7_channels_DRYRUN_dev_v1.json (without the log-probability channel) and
  results/t1/g7_channels_DRYRUN_dev.json (with it). Own-model excess against the last layer:
  - answers −0.011 [−0.014, −0.007]; top-1 +0.043; top-5 +0.059; top-20 +0.075 [0.069, 0.082];
  - full (Fisher–Rao) +0.070; log-probability +0.207 [0.198, 0.216];
  - own model best in its row: answers 0/8, top-20 2/8, log-probability 8/8.
  - Against the chosen layer: answers +0.006 [0.003, 0.009], top-20 +0.037, log-probability +0.071 (own best 5/8).
- In the first dry run, the full next-token law under Fisher–Rao carried no more own-model geometry than top-20. That
  led to adding the log-probability channel, which is what Theorem 6 uses.
- gpt-oss: at the plain prompt its next token is a fixed format token, so the channel is read after its answer
  prefill, at the first content position. Hidden states are read at that same position for every model.
- The G7 contexts exist (1,000; 0 overlap with L2 or G6). No model has seen them.

## Design
- **Contexts:** 1,000 fresh contexts of G6's ten kinds, 100 per kind, numpy seed 77.
- **Models:** the 8 open models, bf16, no system message.
- **Channel data:**
  - one forward pass per context gives the next-token law, its top 20, and the hidden states at L2's 6 layers, at the
    same position;
  - answers come from scripts/132's vLLM pass.
- **Layers:** G1's dev-chosen layer, and the last layer (where the next-token law is read out).
- **Statistic:** the 8×8 matrix R[A, B] = Spearman(channel geometry of A, hidden geometry of B) over all context
  pairs. The own-model excess is the mean diagonal of the two-way residual (R − row means − column means + grand mean).
  Bootstrap over contexts: 500 resamples, seed 1390, the same resamples for every channel (so differences are paired).

## Endpoints
- **K1 (answers reveal no model-specific geometry).** The answers' excess against the chosen layer lies inside
  [−0.02, +0.02] (95% interval).
- **K2 (the next-token channel leaks it).** The top-20 excess against the last layer has a lower 95% bound > 0.02.
- **K3 (the ladder).** Against the last layer, all three hold: top-20 − answers > 0; top-5 − top-1 > 0;
  top-20 − top-5 > 0 (lower 95% bounds, paired).
- **K4 (the linear channel leaks most).** Against the last layer, log-probability − top-20 > 0 (lower 95% bound).
- **K5 (identification).** With the log-probability channel, against the last layer, the model's own hidden geometry
  is the best match in its row for at least 7 of the 8 models. With answers, for at most 2 of 8.
- **Reported:** everything at both layers; the own model's rank in each row; full (Fisher–Rao) vs top-20.

## Readings, fixed in advance
- **K1–K4 pass:** what an API reveals of a model's own state depends on its channel. Yes/no answers reveal only what
  models share. Next-token probabilities leak the model's own geometry, more with each additional token returned, and
  the log-probability vectors leak most. That is consistent with the exact recovery of Theorem 6.
- **K2 or K3 fail:** top-k log-probabilities do not leak measurably more than answers; the paper says so.
- **K4 fails:** the linear channel adds nothing beyond top-20 at the level of geometry.
- **K5 fails:** log-probabilities do not single out a model's own internals among eight.

## Limits stated in advance
- Eight open models. A closed API is simulated by the channel it would return, not measured.
- The top-k statistic is the Bhattacharyya coefficient computable from what an API returns. When two contexts' top-k
  sets differ, it is not an exact Fisher–Rao distance.
- Own-model excess is a small part of each geometry. Most of what every channel reveals is shared.
