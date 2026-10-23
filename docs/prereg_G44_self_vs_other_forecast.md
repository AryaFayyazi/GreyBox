# Preregistration G44: does an agent's own state forecast its next violation better than another model reading the same conversation?

Locked, and published, before the other model has answered anything. Code: scripts/220_g44_self_vs_other_forecast.py.
GPU only (no API cost): the agents' own answers already exist (G40, G27b).

## Why
The thesis is that a closed agent's API reveals its own hidden state. Every own-vs-other test so far (G16, G19, G21)
asked about a call already in the transcript: there any capable reader can judge the call from the facts, and other
models did as well or better. Before the call exists, what the agent will do next depends on its own policy, which the
transcript does not show. If the agent's answers about its next action forecast its next violation better than another
model's answers to the same questions in the same seat, the API reads something of the agent's own state beyond the
shared transcript. Binder et al. (2024) found privileged self-prediction only after fine-tuning; this test is
training-free, through the API, on agents' policy violations.

## Design (fixed)
- Other model: Gemma-4 (open), a control, never part of the method. It answers G27's 8 forward-looking questions in
  G27's pre-call frame, 4 answers each, in the agent's seat: the agent's conversation before the call, replayed in
  Gemma-4's chat format (as G16 did for calls).
- Closed agent (primary): GPT-6-luna's G40 units (252 fresh episodes, seed 52) in template|tool strata holding both
  classes: 402 calls, 27 violations, 3 strata. Own answers: G40's, asked before every turn.
- Open agents (secondary): G31's steps of gpt-oss-20b (136, 82 violations), Qwen3.8 (41, 23), Muse-Glimmer (226, 93),
  in template x tool x step strata holding both classes. Own answers: G27b's.
- Score: G27's raw read-out of the 8 answers; nothing fit; the same for own and other.

## Endpoints (cluster bootstrap over scenarios, 2000, seed 4400)
- **F1 (primary).** GPT-6-luna: within-template|tool AUROC of its own read-out minus the other model's: lower 95% > 0.
- **F2.** Open agents, pooled (mean over agents): within-template x tool x step AUROC own minus other: lower 95% > 0.
- **Reported.** Own and other AUROCs; GPT-6-luna within template x tool x position.

## Readings, fixed in advance
- F1 passes: before its call exists, a closed agent's own answers forecast its next violation better than another
  model reading the same conversation: the API reads the agent's own state, not only the shared transcript.
- F2 passes: the same holds for open agents.
- F1 fails: at this power, another model in the agent's seat forecasts as well; the forecast reads the transcript.

## Known before locking
The agents' own pre-call AUROCs (G30, G38, G40). No answer of the other model exists; no own-vs-other comparison has been
computed.
