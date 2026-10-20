# Preregistration G11: does an agent's offline self-audit predict how much it can police itself live?

Locked, and published, before any G11 episode exists. Theory: docs/theory_selfaudit.md (the prevention law). Code:
scripts/151_live_selfaudit.py and scripts/152_live_selfaudit_analysis.py (both unchanged since the G9 lock),
scripts/156_g11_analysis.py. No other model; nothing trained.

## Why
G9 confirmed that Gemma-4 polices itself through its own API. Development audits show that agents differ: the share of
their own violating calls that they flag (all 4 answers "no", cut 1.0) is Gemma-4 0.883, Muse-Glimmer 0.742, Qwen3.8
0.308, gpt-oss 0.220. For a gate that blocks every flagged violation, the prevention law makes the relative reduction
in violations about the flag rate tau. Gemma-4 (G9): tau 0.883 → observed 0.82 [0.76, 0.88] (known; the calibration
point, not part of this test). G11 tests the prediction on two more agents, with nothing tuned on them.

## Known before locking
- The development audits above (results/t1/self_audit_DEV_*_sample.json); no live episode of Muse-Glimmer or Qwen3.8
  exists; neither agent has run CAPE's v4fresh split.
- G9's protocol and its result.

## Design
- **Agents:** Muse-Glimmer-30B (vLLM 0.30 server) and Qwen3.8-27B, CAPE's server settings for each.
- **Scenarios:** v4fresh, every unsafe non-nuisance instance (333), seeds 0 and 1: 666 episodes per arm per agent.
- **Arms:** none; gate@1.0 (G9's: a call with all 4 audit answers "no" goes to review; CAPE's checker reviews).
- **Prediction (fixed):** relative reduction = development tau at cut 1.0: Muse-Glimmer 0.742, Qwen3.8 0.308.
- **Bootstrap:** over scenario instances (both seeds together), 2,000 resamples, seed 1560.

## Endpoints
- **P1 (the law predicts each agent).** For each agent, |observed relative reduction − predicted| ≤ 0.15.
- **P2 (the ranking).** Muse-Glimmer's reduction exceeds Qwen3.8's: lower 95% bound of the difference > 0.
- **P3 (no loss of task success).** For both agents, success gate − none: lower 95% bound > −0.03.

## Readings, fixed in advance
- **P1–P3 pass:** how much an agent can police itself is predicted, before any live run, by a cheap offline audit
  through its own API.
- **P1 fails:** the law's prediction misses for that agent (e.g. blocked violations are retried, or flags are
  correlated), and we report by how much.
- **P2 fails:** the offline ranking does not carry to live prevention.

## Limits stated in advance
- Two agents; one benchmark; the reviewer is CAPE's checker.
- Episode-level reduction is compared with a call-level flag rate; a blocked violation may be retried.
