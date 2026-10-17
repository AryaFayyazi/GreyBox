# Preregistration T8: monitoring a closed frontier agent with a reader trained only on an open agent

Locked before the Claude agent run (scripts/74 stage=run). Two pilots (10 and 40 episodes, seeds 49 and 48) checked
the harness, the cost and the violation rate (4/40). Their episodes are never analysed.

## Why
The practical claim of T4–T7 is that closed agents can be monitored from their transcripts. The test that matters is
a real closed frontier agent, monitored by a reader that never saw that agent.

## Data
- **Agent:** Claude Haiku 5.5 on CAPE-Bench core24b x− scenarios, through Anthropic's tool-use API (scripts/74).
  CAPE's semantics: MockEnv labels, partner-world reveal, one executed call per turn, max 20 turns.
- **Episodes:** 1,000, shuffled scenarios (rng 0), seed 50. Stopping rule: the $9.90 ledger cap.
- **Units** (the clean-unit rule): in informed episodes, executed non-read tool calls after the reveal, up to and
  including the first unsafe action. y = MockEnv unsafe_action, used only for scoring.

## Monitors (frozen; nothing is fit on Claude)
- **READER:** models/monitor/reader_probe.joblib (Qwen3-8B probe trained on Gemma-4 CAPE-Bench traces only).
- **J_act:** T3's zero-shot judge (Llama-3.3-70B, same prompt).

## Endpoint (within-template AUROC; scenario-cluster bootstrap, 1,000 resamples)
- **S1.** AUROC(READER) − AUROC(J_act) on Claude's units: lower 95% bound > 0. Tested only with at least 10 unsafe
  units.
- **Reported:** both AUROCs; violations caught at 5% of safe actions escalated; units, episodes, spend.

Code: scripts/74_claude_agent.py (data), scripts/75_t8_claude_monitor.py (stages hidden, judge, analyze).
