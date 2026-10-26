# Preregistration G73: a closed model's answer geometry on fresh contexts, against every text a monitor could embed

Locked, and published, before any G73 context is collected. Code: scripts/265_g73_claude_fresh_geometry.py. Paid (Anthropic).

## Why
G2 showed Claude Haiku 5.5's answer geometry is stable (split-half 0.95) and agrees with open models' hidden geometries at
0.45-0.51, but on held-out L2 contexts with only an input embedding as baseline (roughly matched), and Claude's own replies
were never tested as a baseline. Reviewers asked whether a closed model's answers reveal the shared hidden geometry better
than any text available to a monitor. G6's 1,000 fresh contexts have 8 open models' hidden states and input embeddings.

## Design (fixed)
400 of G6's fresh contexts (numpy seed 7300). Claude Haiku 5.5 (reasoning off): G2's 24 battery items, 8 samples each
(graybox.adapters.fork, scripts/32's parser), and 8 real replies to each unmodified context (max 400 tokens).
Claude's answer geometry: Theorem 5's debiased Fisher-Rao distances. Text baselines on the same 400 contexts: the input
under MiniLM, Qwen3-Embedding-0.6B and MPNet (stored, scripts/147), and Claude's own replies under Qwen3-Embedding and
MPNet (mean of 8, as for the open models' replies in scripts/147). Truth: each of G6's 8 open models' hidden geometry
(stored states, G1's development-chosen layer, cosine). Spearman over all context pairs; bootstrap over contexts, 500,
seed 7301. A smoke run (2 contexts, separate file, never analysed) checked collection (192/192 parsed); the analysis code
ran end to end on synthetic data (output discarded). Collection starts after G72's episodes end (one paid process).
Cost: about $2.

## Endpoint
- **E1 (primary).** For each open model, Claude's answers − the strongest of the 5 text baselines (chosen per model):
  lower 95% bound > 0 on at least 6 of the 8 models. Reported: every value, and the open models' own answers.

## Readings, fixed in advance
- E1 passes: through its API alone, a closed model's answers reveal the hidden geometry that open models share better
  than any text a monitor could embed, its own replies included.
- E1 fails: on fresh contexts, text embeddings capture as much of the shared hidden geometry; reported with the margins.
