# Preregistration T17: does the grey box replicate the hidden state of other open agents?

Locked before any T17 number exists. No map from reader features to these agents' states has been fit. The GPU smoke
test (8 steps per agent) checked only the decision token, the model load and the speed.

## Question
T15 showed replication on one agent (Gemma-4): a linear map from the reader's latent state recovered R² 0.664 of
Gemma's decision state, and white-box probes worked on the replica. One agent can be a special case. T17 asks
whether the same reader (Qwen3-8B, layer 32) replicates the hidden state of three more open agents with different
architectures, and with a stricter split than T15.

## Agents (CAPE-Bench core24b and core24b_reasoning)
| Agent | Architecture | Blocks | Units |
|---|---|---|---|
| gpt-oss-20b | mixture of experts | 24 | T6's gptoss20b_clean: 5,229 (729 violations) |
| Qwen3.8-27B | hybrid linear attention | 64 | built here: 4,664 (164 violations) |
| Muse-Glimmer-30B | dense, sliding window | 52 | built here: 5,184 (684 violations) |

- **Units** follow T1 (scripts/15 `units_of`). They are every proposed non-read action after the reveal, up to and
  including the first violation, in informed episodes. All violations are kept. Safe actions are subsampled with
  seed 0: Qwen3.8 3,000 safe and 1,500 safe-world twins; Muse Glimmer 4,500 safe (CAPE has no Muse safe-world set).
- **Sets:** every core24b and core24b_reasoning violation-world set that CAPE has for the agent, because violations
  are scarce. They are listed in `SETS` in the script. The benchmark of each set comes from its name. (The pattern in
  scripts/15 needed a prefix, which Qwen3.8's sets do not have; T17 uses a fixed parser that asserts the spec file
  exists.)

## Split (stricter than T15)
Development and fresh steps come from **disjoint scenarios**: a scenario is in development if sha256(scenario_id) is
even. (In T15, fresh steps came from new seeds of mostly the same scenarios.)

## Target
The agent's hidden state at its decision point: the end-of-reasoning token, after this turn's reasoning and before
the tool call.
- **Context:** the agent's own chat template, applied to the turn-start context plus this turn's reasoning and tool
  call, and cut after the end-of-reasoning token.
  - gpt-oss: `<|end|>` of the analysis message;
  - Qwen3.8: `</think>`;
  - Muse Glimmer: `<|eom|>` of its `to=self` reasoning message.
- **Checks:** the script asserts that the last token is that token. A step is excluded (`ok` = false) when it has no
  reasoning, when the turn prompt is not a prefix of the rendering, or when the token is missing. The excluded counts
  are reported.
- **Layers:** 5 depths, block round(n·f) − 1 for f ∈ {0.5, 0.62, 0.72, 0.82, 0.9}. This is the grid that gave
  Gemma's layers 14–26 (T15).
- **Processing:** the states are standardized on development and reduced to the top 128 principal components (fit on
  development). The share of variance kept is reported.
- **Engine:** Hugging Face, bf16, a forward hook on each block output.
- **Fidelity check (reported):** the first 16 tokens of the agent's actual tool call are appended, and the agent's
  model predicts them teacher-forced. The mean accuracy and mean probability tell whether the rebuilt context is the
  one the agent saw. The state is read at the end-of-reasoning token. (Smoke test, 6 steps per agent: accuracy
  0.86–1.00. The state with or without the appended tokens had cosine ≥ 0.993.)

## Replica sources (each from the transcript and the proposed action only)
- **READER:** Qwen3-8B layer-32 features on the shipped monitor prompt. For gpt-oss these are T6's features
  (outputs/t6/gptoss_act.npz); for the two new agents they are computed by the same code.
- **TXT:** TF-IDF of context and action (T9's features).

## Map and endpoints (T15's, unchanged)
- **Map:** ridge regression to the 128 components. Alpha ∈ {1, 10, 100, 1000, 10000} is chosen by 5-fold GroupKFold
  over scenario_id on development. The map is fit on all development steps.
- Bootstrap over fresh scenario_id, 1,000 resamples. Seeds: gpt-oss 171/172, Qwen3.8 181/182, Muse Glimmer 191/192.
- **E1, replication (per agent):** variance-weighted R² over the 128 components on fresh steps, R²(READER) − R²(TXT).
  Pass: lower 95% bound > 0.
- **E2, readouts work on the replica (per agent):** a probe trained on TRUE development states (standardize →
  logistic, C = 0.05) predicts the violation label. It is applied to the true fresh states and to the
  READER-reconstructed fresh states. Within-template AUROC (replica − true), where the template is the scenario
  template and tool, with gpt-oss's harmony residue removed from tool names. Pass: lower 95% bound > −0.02
  (non-inferiority).

## Readings, fixed in advance
- **Replication generalizes:** E1 and E2 pass on all three agents.
- **Partial:** E1 passes on all three, and E2 fails on one or more.
- **Not general:** E1 fails on one or more agents. The paper then states that replication was shown for Gemma-4 only
  and names the agents where text features replicate the state as well as the reader.
- Each agent is a separate replication. The headline is the conjunction, so no multiplicity correction is applied.

## Reported, not tested
- R² values per agent and per layer; the variance kept; E2 with TXT reconstructions.
- **Power note:** Qwen3.8 has only 164 violations, about 80 in the fresh half. E2's interval may be wide.
- **Exploratory:** Gemma-4's T15 development units under the same scenario-disjoint split (seeds 201/202). This puts
  all four agents on one footing.

Code: scripts/96_t17_replication_agents.py. Stages: units (CPU), hidden (GPU; Muse Glimmer runs in a venv with
transformers 5.20.1, because the shared venv's 5.14.1 does not know its architecture), reader (GPU), analyze (CPU).
