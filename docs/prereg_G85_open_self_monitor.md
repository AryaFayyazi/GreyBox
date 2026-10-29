# G85: G84 with the agent itself as the reasoning text monitor

Locked 2026-10-28 before any G85 monitor answer on a unit exists. Code: scripts/287_g85_open_self_monitor.py (G84's
transcripts and analysis, unchanged). GPU only (no API cost).

## Why
G84 passed: before the call, on four open agents, the geometry (0.77) and the signed forecast (0.82) beat a reasoning
text monitor (Qwen3-32B, 0.65). The hostile reading: the monitor was one external model, while G52 (Claude) and G54
(GPT-6-luna) used the agent itself, which may know its own coming action better (on GPT-6-luna it matched its own
sampled policy). G85 repeats G84 with each open agent as its own reasoning monitor.

## Design (fixed)
- Units, transcripts, prompt, scores and analysis: G84's, unchanged (1,189 units of G82; G52's monitor prompt as the
  system message; G52's transcript layout from the exact turn-start context; next action not shown).
- Monitor: the agent itself, served as in its rollouts (CAPE's server settings: its sampling, its reasoning on, at
  medium effort where the model has a setting; Muse-Glimmer under vLLM 0.30 with its system suffix), no tools, 3
  samples (seeds 8501 + unit index), at most 6,000 new tokens. Score: mean parsed probability; unparsed = 0.5; failed
  requests counted and reported.

## Endpoints (G84's, unchanged)
- **P1 (primary).** Mean over the four agents of (geometric - self-monitor): lower 95% bound > 0.
- **P2.** Mean over the four agents of (signed - self-monitor): lower 95% bound > 0.
- Holm over P1 and P2 at 0.05 (G68's rule). Cluster bootstrap over scenarios within agent, 2,000 resamples, seed 8401.

## Expectation, stated before the run
Uncertain. G68's single self-report answers (no reasoning) scored 0.47-0.86 before the call, below the agents' own
sampled actions by +0.095 on average; reasoning may close that gap.

## Readings, fixed in advance
- P1 passes: G84's result is not an artefact of the external monitor; the geometry beats the agent reasoning about
  itself too.
- P1 fails: an open agent reasoning over its own transcript forecasts its violations about as well as the geometry; the
  paper reports G84 together with this and limits the claim to monitors other than the agent itself.

## Smoke test (before the lock)
`stage=smoke`: each agent's serving path on one synthetic transcript that is not a unit (parsed probabilities printed
in the slurm log g85smoke). No unit was sent to any monitor before the lock.
