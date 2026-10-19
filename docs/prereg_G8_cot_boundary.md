# Preregistration G8: does reading an agent's reasoning help a monitor?

Locked, and published, before any reader feature of a held-out scenario WITH the agent's reasoning exists.
Code: scripts/149_cot_boundary.py (stage=features split=held, then stage=analyze). It reuses scripts/96 (T17's agents,
units, scenario split and strata) and T4's reader prompt (scripts/69). No model is trained or fine-tuned; the probes are
the shipped recipe (standardise, PCA 256, logistic C 0.05).

## Why
- Corollary 1 (paper, Theory): when a label is decided by the action, the visible tokens and facts of the world, the
  agent's hidden tokens add nothing at the Bayes level. The reasoning is such a hidden token stream.
- The corollary is silent on a label that the current action does NOT decide, such as a violation later in the
  episode. There the reasoning may add information (it can state a plan).
- Published work reports that chain-of-thought monitors beat action-only monitors (e.g. CoT Red-Handed, 2505.23575;
  Monitoring Monitorability, ICML 2026) and that access to the reasoning can also mislead a monitor (persuasion,
  2607.08066). Those use zero-shot LLM monitors on sabotage side tasks. G8 asks the question for trained readers on
  policy labels, across three open reasoning agents, with labels the theory sorts into two kinds.

## Known before locking
- T4 (published): on Gemma-4's fresh units the reader with the reasoning scores 0.9927 vs 0.9939 without
  (−0.0014 [−0.0046, +0.0014]). Gemma-4 is therefore not part of the confirmatory set.
- Dry run on Gemma-4's clean units (grouped 5-fold CV): action −0.006, later violations −0.007
  (results/t1/cot_boundary_DRYRUN_dev.json).
- Dry run on the three agents' DEVELOPMENT scenarios (results/t1/cot_boundary_DRYRUN_agents_dev.json), cot − act:
  - action, in-agent grouped CV: gpt-oss −0.040, Qwen3.8 +0.011, Muse −0.030;
  - action, transfer from Gemma-4: −0.067, −0.067, −0.033;
  - later violations, in-agent CV: +0.019, +0.010, −0.005 (AUROC 0.53–0.67; 29–90 positives per agent);
  - later violations, transfer: −0.079, +0.120, +0.058.
- A smoke test of stage=analyze ran with development features standing in for held-out ones (in-sample; not a result).
- Held-out label counts are known (safe steps followed by a later violation: 45, 22, 108). No held-out reader feature
  with reasoning exists. The held-out features WITHOUT reasoning exist (T17) and were scored in T17 (action label only).

## Design
- **Agents:** T17's three open reasoning agents: gpt-oss-20b, Qwen3.8-27B, Muse-Glimmer-30B. Units, labels and the
  scenario hash split are T17's. Held-out = scenarios with an odd hash.
- **Monitors:** the shipped reader (Qwen3-8B, layer 32) on
  - "act": the transcript + the proposed action (T17's stored features);
  - "cot": the transcript + the agent's reasoning for that turn (last 4,000 characters, special tokens stripped) + the
    action, in T4's rendering.
- **Labels:**
  - action: the proposed action violates the policy (all decision steps);
  - future: a later tool call in the same episode violates (only steps whose own action is safe).
- **Probes:**
  - in-agent: trained on the agent's development units, scored on its held-out units;
  - transfer: trained on Gemma-4's clean units only (no label from the agent: the deployable case).
- **Metric:** within-template AUROC (T17's strata). **Bootstrap:** held-out scenarios resampled within each agent,
  1,000 times, seed 1490. Pooled = mean over the three agents of cot − act, with independent resamples per agent.

## Endpoints
- **C1 (no gain on action labels, in-agent).** Pooled in-agent cot − act, action label: upper 95% bound < +0.01.
- **C2 (the reasoning hurts transfer on action labels).** Pooled transfer cot − act, action label: upper 95% bound < 0.
- **C3 (consistency).** Transfer cot − act < 0 (point estimate) for each of the three agents, action label.
- **C4 (later violations; two-sided reading).** Pooled in-agent cot − act, future label:
  - "reasoning helps" if the lower 95% bound > 0;
  - "equivalent" if the interval lies inside (−0.05, +0.05);
  - otherwise "inconclusive".
- **Reported:** every agent and both probes for both labels; the AUROC levels; future transfer.

## Readings, fixed in advance
- **C1–C3 pass:** for labels that the action decides, reading the reasoning gives a trained monitor no gain, and it
  makes a monitor trained on another agent worse. This is what Corollary 1 predicts at the Bayes level, and more: the
  reasoning is a nuisance input that shifts across agents.
- **C1 fails:** the reasoning carries action-label information that the reader cannot get from the action at this
  sample size; the paper says so.
- **C2 or C3 fail:** the transfer cost seen in development does not replicate.
- **C4 "reasoning helps":** the theory's boundary is real: the reasoning forecasts later violations that the current
  action does not show. **"equivalent":** no forecasting value at this resolution. **"inconclusive":** too few
  positives to say.

## Limits stated in advance
- Three open agents, one reader, one policy suite (CAPE's agentic scenarios). Not adversarial agents; not sabotage.
- The reasoning is cut to its last 4,000 characters, and the reader's input to its last 12,000 tokens.
- Later-violation positives are few (175 pooled), so C4 has low power.
- In-agent probes see only the agent's development units; the probes are not re-tuned for the reasoning input.
