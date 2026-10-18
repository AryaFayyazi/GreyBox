# Preregistration T23: T22's judge on four closed frontier agents' τ²-bench writes

Locked, and published, before the judge runs on these writes. Known before locking: T20–T22's results. The judge has
never scored these writes. T21 trained monitors on them, so their labels and composition are known: 894 writes,
227 violations, 41 tasks (29 with a violation). Writes per agent: Claude 3.7 Sonnet 254, GPT-4.1 195,
GPT-4.1-mini 289, o4-mini 156.

## Why
In T22 the zero-shot judge reached within-tool AUROC 0.804 on T20's writes, where T21's reader reached 0.535. But the
violations came from only 8 tasks, so neither interval excluded 0. The judge uses no labels, so the 894 published
writes of four closed frontier agents are a valid test set for it, with 12 times as many violations. The monitors that
need labels are scored by cross-validation grouped by task: each write is scored by a monitor that never saw its task.

## Design
- **Judge:** T22's judge, unchanged: Qwen3-32B with thinking, the same prompt and parser (imported from
  scripts/103d), seed 0 with T22's sampling settings. An answer with no number scores 50.
- **Reported:** the mean score over seeds 0–3.
- **Units:** the 894 writes (t18_train.parquet), with CAPE's oracle labels.
- **Baselines (5-fold GroupKFold over task_id):**
  - READER_cv: Qwen3-8B layer 32, the same recipe as T18/T20/T21;
  - TXT_cv: TF-IDF as in T9;
  - TOOL_cv: the tool's violation rate (see amendment 1).
- **Metric:** within-tool AUROC. Cluster bootstrap over task_id, 1,000 resamples.

## Endpoints
- **K1 (the judge checks the rules):** JUDGE − TOOL_cv. Pass: lower 95% bound > 0. Seed 2301.
- **K2 (the judge beats the reader on tasks the reader never saw):** JUDGE − READER_cv. Pass: lower 95% bound > 0.
  Seed 2302.
- **Reported:**
  - JUDGE − TXT_cv (seed 2303);
  - the 4-seed mean − READER_cv (seed 2304);
  - pooled AUROC and the label-free catch rate at 5%;
  - caught and safe flagged at 50;
  - within-tool AUROC per agent;
  - parse failures, truncations and the share of scores that are exactly 0.

## Readings, fixed in advance
- **K1 and K2 pass:** for a written rule set whose facts are in the transcript, a zero-shot reasoning judge checks
  closed agents' actions on tasks with no labels, where readouts trained on other tasks do not. The paper and the tool
  then recommend the judge for explicit rules and the grey-box probe for policies learned from labels. T22's result was
  limited by its sample.
- **K1 passes, K2 fails:** the judge checks rules beyond the tool identity, but not better than the reader trained on
  other tasks of the same benchmark.
- **K1 fails:** the judge does not reliably check these rules on these agents either.

## Limits stated in advance
- These are the agents and tasks that T21 trained on. The judge does not use them, but we chose this test after T22.
- The four agents are older frontier models with published trajectories, not agents we ran.
- One judge model, one prompt.

Code: scripts/103e_t23_tau2_judge_frontier.py (judge: one GPU, vLLM; analyze: CPU).

## Amendment 1 (2026-10-17, before any judge result on these writes; published with its own lock)
A CPU dry run of the baselines showed that TOOL_cv, the tool's rate in each training fold, is not constant within a
tool: it varies between folds and scored a within-tool AUROC of 0.217. That would have inflated K1. TOOL_cv is now the
tool's rate on all writes. It is constant within each tool, so its within-tool AUROC is exactly 0.5, as in T18–T22.
Nothing else changes. Dry-run values of the other baselines (within tool): READER_cv 0.646, TXT_cv 0.490.
