# Preregistration T6: train the transcript reader on one agent, monitor another

Locked before any T6 hidden state or judge output exists, and before T5's result.

## Why
T4 found that a transcript reader trained on labels beats the agent's own internals (same agent, same scenarios).
A practical monitor for closed agents must also transfer to agents it was not trained on.

## Design
- **Reader probe:** exactly T4's (Qwen3-8B; prompt; layer 32, chosen by T4 on Gemma development units). Trained only
  on Gemma-4 development units (5,493), applied unchanged to gpt-oss-20b's clean units (5,229; 729 unsafe), a
  different agent family.
- **Baseline:** T3's zero-shot judge (Llama-3.3-70B, same prompt) on gpt-oss's actual actions.
- Code: scripts/71_t6_cross_agent.py (stages hidden, judge, analyze).

## Endpoint (within-template AUROC on gpt-oss units, scenario-cluster bootstrap, 1,000 resamples)
- **Q1.** AUROC(Gemma-trained reader on gpt-oss) − AUROC(zero-shot judge on gpt-oss) > 0: lower 95% bound > 0.

## Reported, not tested
- **Label efficiency:** reader probes trained on development units from 10, 25, 50, 100, 200 or all scenarios (5
  random draws each), tested on Gemma's fresh units.
- **Latency:** median reader forward time per action (one H100, no batching).
