# G83: questions written for new kinds of context

Locked 2026-10-27 before any answer to the new questions exists. Code: scripts/282_g83_newkind_battery.py; questions:
src/graybox/battery_newkinds.py (24, written before any answer). GPU only.

## Why
Exploratory splits of G78 (scripts/280): on fresh contexts of the six kinds the 88-question battery was written for, answer
geometry beats every text baseline (Claude +0.36, open models' own exact answers +0.32); on new kinds (news, code,
jailbreak_bench, injection) it barely does (Claude +0.02, open models +0.08). Reading: the battery's coverage, not the API,
limits new kinds. G83 tests that reading: questions written for those kinds should restore the geometry there.

## Design (fixed)
- Contexts: G6's 400 fresh contexts of the four new kinds; hidden states: G6's stored states at G1's development-chosen
  layer, cosine geometry (G6/G73 code). Note: the 88 questions' answers on these contexts exist (G6); the 24 new ones do not.
- Answers: each of G6's eight open models, exact P(yes) on the 24 new questions (scripts/132's generation code unchanged:
  G6's frame and preamble, no system message, logprob tier). Fisher-Rao angles; Euclidean geometry.
- Text baselines (G73/G78's five, with each model's own replies): input under MiniLM, Qwen3-Embedding-0.6B, MPNet; the
  model's own replies under Qwen3-Embedding-0.6B and MPNet.

## Endpoints (bootstrap over contexts, 1,000, seed 8300; means over the eight models)
- **E1 (primary).** Spearman(answers on 88+24 questions, hidden) - Spearman(answers on 88, hidden): lower 95% bound > 0.
- **E2.** Spearman(answers on 88+24, hidden) - Spearman(strongest text baseline, hidden): lower 95% bound > 0.

## Readings, fixed in advance
- E1 and E2 pass: questions written for a kind of context restore the answer geometry there; the new-kind limit is the
  battery's coverage, which a user can extend.
- E1 fails: adding questions for new kinds does not help; the limit is not only coverage.
