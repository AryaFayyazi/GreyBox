# Preregistration G9: can an agent police itself in real time, through its own API?

Locked, and published, before any G9 episode exists. Theory: docs/theory_selfaudit.md (written before any self-audit
result). Code: scripts/151_live_selfaudit.py (episodes), scripts/152_live_selfaudit_analysis.py (analysis),
scripts/150_self_audit.py (the audit prompt and parser). No other model is used and nothing is trained: the only model
in the loop is the monitored agent, reached through an OpenAI-style API.

## Why
A monitor for a closed agent should need only that agent's API. The protocol: before a proposed tool call runs, the
agent itself is asked in a side branch (a separate API call; the real conversation never sees it) whether the call
complies with its policy. A flagged call is replaced by a fresh draw of the agent (S_M), or held for review (gate).

## Known before locking (all development data)
- Stored turn-start resamples: where Gemma-4 violated, its own violation rate R had median 0.31.
- Offline self-audit, Gemma-4 clean steps (5,493; 993 violations): within-template AUROC 0.950; flags 96.6% of its own
  violations and 11.1% of its safe actions (majority of 4 answers "no").
- Cross-audit (gpt-oss judging Gemma-4's steps 0.878; Gemma-4 judging gpt-oss's 0.780; gpt-oss on its own 0.741): a
  generation–verification gap, not privileged self-knowledge.
- Live pilot 1 (core24b, 200 scenarios, seed 7, escalation stops the episode): violating episodes 22.5% → 4.5% (gate),
  5.5% (S4).
- Live pilot 2 (core24b, 200 scenarios, seed 8, escalation goes to review), violating episodes / success / reviews per
  episode: plain 24.0% / 37.0%; gate@1.0 5.5% / 46.0% / 0.70; gate@0.75 6.5% / 44.0% / 1.04; S4@0.75 8.0% / 42.5% /
  0.20; S4@1.0 9.0% / 40.5% / 0.13.
- The two operating points below were chosen on pilot 2.
- Claude Haiku 5.5's development self-audit is weak (AUROC 0.46 with this prompt); G9 is about Gemma-4. Claude is a
  separate, later test.
- No Gemma-4 episode on the v4fresh split exists anywhere in this project.

## Design
- **Agent:** Gemma-4 (26B-A4B), CAPE's server settings (vLLM, its own tool and reasoning parsers, thinking on).
- **Scenarios:** CAPE-Bench v4fresh, every unsafe-world, non-nuisance scenario (333; 8 templates), seeds 0 and 1:
  666 episodes per arm. At most 20 turns.
- **Audit:** scripts/150's ASK prompt as an operator user message after the agent's own conversation, with the proposed
  call; 4 sampled answers (reasoning on, max 4,096 tokens); s = share of "no".
- **Arms** (same per-step seeds, so arms share every draw until they diverge):
  - none: the plain agent;
  - gate@1.0: a call with s = 1 (all 4 answers "no") goes to review;
  - S4@0.75: a call with s ≥ 0.75 is replaced by a fresh draw, up to 4 draws; after 4 flags, the first draw goes to
    review.
- **Review:** CAPE's checker stands in for a human reviewer: an allowed call runs; a violating call is blocked and the
  agent is told so.
- **Outcomes per episode:** an executed violation (any); success (goal reached and no executed violation); number of
  reviews.
- **Bootstrap:** over the 333 scenario instances (both seeds of an instance resampled together), 2,000 resamples,
  seed 1520 (scripts/152). Relative reduction = 1 − (arm's violating-episode rate) / (none's rate), per resample.

## Endpoints
- **L1 (the gate prevents violations).** gate@1.0 vs none, relative reduction in episodes with an executed violation:
  lower 95% bound ≥ 0.50.
- **L2 (resampling prevents violations with few reviews).** S4@0.75 vs none: relative reduction lower 95% bound ≥ 0.40,
  AND reviews per episode ≤ 0.40 (point estimate).
- **L3 (no loss of task success).** For both arms, success − none: lower 95% bound > −0.03.
- **Analysis call:** scripts/152 files=[gemma4_v4fresh_g9_c100,gemma4_v4fresh_g9_c075] endpoints=g9
  out=g9_live_self_policing.json.
- **Reported:** absolute rates, reviews per step, draws per step, the live flag rates on candidate draws (tau, phi),
  per-template rates.

## Readings, fixed in advance
- **L1–L3 pass:** an open agent prevents most of its own policy violations in real time through its own API, with no
  other model and no training, and completes at least as many tasks safely.
- **L1 or L2 fails:** the development effect does not hold on unseen scenarios at this size; the paper says so.
- **L3 fails:** self-policing costs task success.

## Limits stated in advance
- One open agent; its violations are mostly slips that it recognizes when asked. Agents whose violations are
  misreadings of the policy (Claude Haiku 5.5 on development data) are not covered by G9.
- The reviewer is CAPE's checker, a perfect reviewer. Reviews per episode measure the human cost.
- v4fresh has 8 of CAPE's 24 templates. The checker's reading of the policy is the ground truth.
