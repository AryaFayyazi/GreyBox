# Preregistration G43: does a broader battery of the agent's own answers reconstruct its decision state?

Locked, and published, before any answer of this battery exists. Code: scripts/219_g43_situation_battery.py. GPU.

## Why
G36 found that, at a proposed call, an agent's 8 policy answers agree only weakly with its own decision-state geometry
(0.01–0.16 over all pairs of calls). Eight questions about one policy measure a narrow slice of a rich state; on generic
contexts, 24–88 broad questions reconstruct the hidden geometry well (G1: 0.41–0.74; G34). The chain's first link at the
decision needs a broader measurement of the agent's situation.

## Design (fixed)
- Questions: 32 fixed yes/no questions about the agent's situation at the call (what it has read, done and been told;
  who must approve; what the call shown does), written before any answer, naming no specific policy rule (listed in the
  code).
- Calls: 200 of G15's held-out calls per open agent (Gemma-4, gpt-oss-20b, Qwen3.8, Muse-Glimmer; numpy seed 4300;
  the gpt-oss call without a valid state dropped, as G15).
- Answers: each agent's own API, G15's frame (the call shown, on hold), 4 answers each, G15's settings and parser.
- Geometry: Theorem 5's debiased Fisher–Rao distances of the 32 angles. Truth: the stored decision state, layer index 3
  (cosine), as G36. Comparison: G15's 8 policy answers on the same calls.

## Endpoints (pooled: mean over agents; bootstrap over calls, 500, seed 4301)
- **S1 (primary).** Spearman(32-question geometry, decision-state geometry) − Spearman(8-question geometry, the same):
  lower 95% bound > 0.
- **S2.** Spearman(32-question geometry, decision-state geometry): lower 95% bound > 0.2.
- **Reported.** Per agent; the 40-question combination.

## Readings, fixed in advance
- S1 passes: a broader battery of the agent's own answers reconstructs more of its decision state than the policy
  questions do; S2 passes: it reconstructs a substantial part of it.
- S1 fails: at the decision, broader situation questions do not reconstruct the decision state better at this size.

## Known before locking
G36's results; G15's answers. A dry run on fake answers (removed; no result file) checked the code (about 0).
