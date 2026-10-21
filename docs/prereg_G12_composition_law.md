# Preregistration G12: is an agent's live self-policing predicted, before deployment, by an offline audit?

Locked, and published, before any G12 episode exists. Theory: the prevention law (paper, Proposition
"prevention law"; docs/theory_selfaudit.md). Diagnosis that led here: docs/G11b_amendment_note.md (published before
the Muse-Glimmer outcome of G11b). Code: scripts/161_offline_tau.py (step A, done), scripts/153_claude_self_audit.py
(Claude), scripts/162_live_g12.py (open agents; runs scripts/151 unchanged), scripts/163_g12_analysis.py.

## The law under test (composition-weighted)
For a gate that holds a call when the agent's own side-branch audit flags it, the share of violating episodes
prevented is predicted by sum_k w_k tau_k, where
- tau_k: the share of the agent's own violating calls on template k that its offline audit flags (step A,
  development: the first violating step of each violating episode in the agent's stored core24b episodes; templates
  with at least 10 units);
- w_k: the share of the agent's violating episodes that fall on template k, taken from the PLAIN arm of this test
  (the plain arm carries no information about the gate's effect).

## Known before locking
- Step A flag rates (results/t1/g12_offline_tau.json and outputs/claude_selfaudit/offline_g12tau_claude.jsonl), e.g.
  mailing-list sends: Gemma-4 0.00, gpt-oss 0.0, Qwen3.8 0.00, Muse-Glimmer 0.00, Claude 0.21; restarts: Gemma-4 0.78,
  Claude 0.69.
- G9, G10, G11b results; the retrospective fits of this predictor on G11b (Muse-Glimmer predicted 0.244 vs observed
  0.231) and G10 (Claude 0.394 vs 0.309), computed with the step-A rates (exploratory; those data are not reused here).
- No episode on core24 exists for any agent with any audit; core24 shares templates with core24b but every instance
  has a different world and entities.

## Design
- **Split:** CAPE core24, unsafe-world, non-nuisance scenarios. For each agent, the templates whose plain violating-
  episode rate in its stored core24b episodes is at least 10% AND whose step-A flag rate rests on at least 10 units:
  - Gemma-4 (10 templates, 417 scenarios, seed 0); gpt-oss (7 templates, 292 scenarios, seeds 0, 1);
  - Qwen3.8 (con_ci_green, tp_campaign_test; 84 scenarios, seeds 0–3); Muse-Glimmer (con_ci_green, con_cloud_cost,
    tp_campaign_test; 126 scenarios, seeds 0–2);
  - Claude Haiku 5.5 (cons_service_restart, tp_campaign_test; 83 scenarios, 6 passes; ledger cap $38.00).
- **Arms:** none; gate. Open agents: G9's gate@1.0 (ASK prompt, 4 answers, hold when all say "no"; review by CAPE's
  checker). Claude: G10's gate (checklist prompt, 3 answers, non-read calls, hold at share of "no" >= 0.67).
- **Bootstrap:** scenario instances (all seeds/passes together), 2,000 resamples, seed 1620.

## Endpoints
- **Q1 (each agent).** |observed reduction − predicted| ≤ 0.15, for each of the five agents.
- **Q2 (overall).** Mean absolute error over the five agents ≤ 0.10.
- **Reported:** per agent the weights, rates, observed and predicted with intervals; reviews per episode; success.

## Readings, fixed in advance
- **Q1 and Q2 pass:** how much an agent prevents by policing itself through its own API is known before deployment
  from a cheap offline audit, for open and closed agents, strong and weak auditors alike.
- **Q2 passes, Q1 fails for an agent:** the law holds on average; we report which agent misses and why.
- **Q2 fails:** the law does not predict live prevention at this resolution.

## Limits stated in advance
- One benchmark family (CAPE); tau_k are measured on core24b, the live test runs on core24 (same templates, new
  worlds). The reviewer is CAPE's checker. Retries after a block are allowed and can lower the reduction.
