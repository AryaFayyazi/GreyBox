# Questions as a common drive: why answers reveal only the shared geometry (hypothesis — REFUTED 2026-10-19)

**Status: refuted by its own tests (docs/research_ledger.md).** P3 failed: the instruction preamble alone, with no
question, erases the model-specific signal in the readout. P1 failed: cross-model agreement of hidden geometries FALLS
after a question (0.73 → 0.56 at 75% depth), it does not rise. The washout is a property of the constrained readout, not
an internal synchronisation. Kept for the record.

## Observations so far
- Answers reveal no model-specific geometry. G6: own-model excess +0.002 on 1,000 fresh contexts. G7: +0.001 against
  the chosen layer.
- The next-token law at the end of the context does reveal it: G7 top-20 +0.076.
- The same top-20 channel read after a question is appended does not: +0.011 against +0.075 on development contexts
  (scripts/141). What erases the model's identity is the question and the computation it triggers, not the yes/no
  projection.

## The hypothesis (from physics: generalised synchronisation under a common drive)
- Different dynamical systems driven by the same strong input can lock into a state that is a function of the input
  alone, whatever their internal differences.
- Here the "systems" are language models, and the drive is the token sequence. Write h_M(x) for model M's state after
  context x.
- After a bare context, h_M(x) still carries model-specific structure: its own continuation statistics, tokenizer and
  style.
- Appending a question q is a strong common drive. Every model must compute the same thing, the answer to q about x.
  If h_M(x ⊕ q) is dominated by a component s_q(x) that every model computes alike (up to its own gauge), then:

## Predictions (testable on open models)
- **P1 (internal convergence).** Cross-model agreement of hidden geometries is higher after a question than after
  the bare context: the mean over model pairs of Spearman(D_A, D_B), with D_M the cosine geometry of h_M.
- **P2 (washout).** The model-specific part visible in output channels shrinks after a question. Seen for top-20
  (+0.075 → +0.011).
- **P3 (content, not length).** Appending neutral text, the preamble without a question, gives less convergence and
  less washout than a question (scripts/143 tests this).
- **P4 (dose).** Questions that constrain the answer more, those the models answer most deterministically, produce
  stronger convergence. Exploratory; to be stated before any confirmatory test.

## Consequences if confirmed
- Behavioural evaluation by questions measures what models share. Model-specific internal structure needs
  direct-readout channels (log-probabilities).
- An open model is a principled stand-in for a closed one exactly in the question-driven regime, the regime in which
  monitors ask "would this action violate the policy?". That links the geometry results to the monitor.
- What it is not: a claim that models' weights or full states become identical. Only geometry over contexts is compared.

## What would refute it
- P1 fails: cross-model hidden agreement does not rise after a question. The washout would then be a property of the
  readout, not of the state.
- P3 fails: neutral text washes out as much as a question. Then any appended computation, not the question's
  content, would be the cause.
