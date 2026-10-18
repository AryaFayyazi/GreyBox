# Preregistration T21: on τ²-bench, do the monitors learn the policy, or the tasks?

Locked, and published, before any T21 number exists. Known before locking: T20's results. T20's 13 test tasks with
writes all occur in its training trajectories (checked with labels only). Removing them leaves 554 training writes with
144 violations from 28 tasks.

## Why
In T20, word counts (TF-IDF) beat the reader on τ²-bench: within-tool AUROC 0.987 against 0.738. But every test task
also occurs in training, with other agents. Word counts can then score well by memorising task-specific tokens
(reservation ids, names, dates) instead of learning the policy. A user test of the SDK on a real support agent found
this pattern: the heaviest words were specific order ids and amounts, and the monitor failed on new customers. If this
explains T20, the paper's reading "the labels transfer through word counts, not through the reader" is wrong for new
tasks, and so is the tool's advice. T21 tests it.

## Design
- **TRAIN:** the published frontier trajectories of T18/T20 (four agents, CAPE's oracle labels), with every write of a
  task that occurs in the test removed. 554 writes, 144 violations, 28 tasks.
- **TEST:** T20's test set, unchanged: Claude Haiku 5.5 and GPT-6-luna with a persistent customer on the 20
  development tasks. 262 writes, 19 violations.
- **Monitors:** T18/T20's recipes and features, unchanged.
  - READER: Qwen3-8B layer 32, conversation-aware context, standardize → PCA 256 → logistic (C = 0.05).
  - TXT: TF-IDF, C by GroupKFold over task_id.
  - TOOL: the tool's violation rate in TRAIN.
- **Metric:** within-tool AUROC (the tool alone scores 0.5), with a cluster bootstrap over the test task_id (1,000
  resamples).

## Endpoints
- **H1 (the reader generalizes to unseen tasks better than word counts):** READER − TXT. Pass: lower 95% bound > 0.
  Seed 2101.
- **H2 (the reader beats the tool identity on unseen tasks):** READER − TOOL. Pass: lower 95% bound > 0. Seed 2102.
- **Reported:**
  - TXT − TOOL (seed 2103);
  - pooled AUROC;
  - the label-free catch rate at 5%;
  - T20's same-task values for comparison.
- **Exploratory:** TF-IDF on text whose entities (emails, 6-character codes, dates, user ids, numbers) are replaced by
  their type, minus TOOL (seed 2104).

## Readings, fixed in advance
- **H1 passes:** T20's word-count advantage came from the overlap of tasks. On unseen tasks the reader uses labels of
  the policy better. The paper then reports T20 and T21 together, and the tool's docs say that word counts need labels
  of the same tasks.
- **H1 fails, H2 passes:** the reader learns something that transfers to unseen tasks, but not more than word counts.
- **H1 and H2 fail:** on τ²-bench neither monitor transfers labels to unseen tasks for these closed agents. The paper
  says so.

## Limits stated in advance
- T21 is designed after T20's results; the test set is the same.
- Violations come from few tasks: the intervals are wide.
- The TRAIN tasks differ from the test tasks, but the airline policy and the tools are the same.

Code: scripts/103c_t21_tau2_unseen_tasks.py (CPU; reader features from T18/T20).
