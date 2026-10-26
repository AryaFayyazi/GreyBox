# G78: a closed model's geometry on new kinds of context, with the full battery (fixes G73's design)

Written and locked 2026-10-26, before any G78 answer on the test contexts exists.

## Why (diagnosis of G73's failure; exploratory, on G73's existing data)
G73 failed: Claude Haiku 5.5's answers to 24 battery questions, on 400 fresh contexts including new kinds, agreed with the
eight open models' hidden geometries at 0.10-0.48, and its embedded replies scored higher on 7 of 8 models. G73 asked
Claude only 24 of the battery's 88 questions, while the open models' answer geometry it was set against uses all 88.
On the same 400 contexts, open models restricted to the same 24 questions agree with their own hidden geometry at only
0.10-0.38 (0.11-0.36 with 8 simulated samples), against 0.23-0.68 with all 88: Claude scored like an open model given the
same questions. Claude's 24-question geometry agrees with the open models' 24-question answer geometry at 0.45-0.63. So
the failure points to the battery's size, not to the closed model.

## Design (fixed)
- Contexts: 200 of G6's 1,000 fresh contexts that G73 did not use (numpy seed 7800); Claude has never seen them.
- Claude Haiku 5.5 answers ALL 88 BATTERY_V2 questions (the open models' items), 4 samples each, through G73's code
  unchanged (graybox.adapters.fork; scripts/32's frame and parser; graybox.budget.Ledger, cap $150.68 = the approved cap);
  8 real replies to each unmodified context.
- Geometry: Theorem 5's debiased Fisher-Rao distances (G73's code). Hidden geometry of each of G6's 8 open models: G6's
  stored states, G1's development-chosen layer, cosine (G73's code). Text baselines (G73's five): the input under MiniLM,
  Qwen3-Embedding-0.6B, MPNet; Claude's own replies (mean of 8) under Qwen3-Embedding and MPNet.
- Bootstrap over contexts (1,000, seed 7801); per model, the strongest baseline is chosen on the full sample (as G73).

## Endpoints
- **P1 (primary).** Mean over the 8 models of (Claude's answer geometry - the strongest text baseline), Spearman with that
  model's hidden geometry: lower 95% bound > 0.
- **P2 (G73's endpoint, kept).** Claude beats the strongest text baseline (lower bound > 0) on at least 6 of 8 models.

## Power (simulated before the lock, open models as stand-ins for Claude)
Each open model in turn plays Claude (its exact P(yes) on the 88 questions, sampled 4 times), against the other 7 models'
hidden geometry and the same baselines (its own replies). At 200 contexts: P1 passed for every stand-in (16 of 16 runs;
at 300 contexts the mean margin was +0.12 to +0.22); P2's count passed in 56% (G73's bar has modest power; kept unchanged).

## Cost check
Smoke run on 3 contexts G73 already used (not test contexts): $0.031, i.e. $0.0105 per context, about $2.10 for 200.

## Readings, fixed in advance
- P1 passes: through its text API, with the full battery, a closed model's answers reveal the hidden geometry that open
  models share on new kinds of context better than any text baseline, including its own replies; G73's failure was the
  battery's size.
- P1 fails: the closed model's geometry does not extend to new kinds of context through this battery; reported as such.
