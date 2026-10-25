# Preregistration G68: says vs does on four open agents across their task families

Locked, and published, before any G68 draw. Code: scripts/260_g68_open_says_does.py (job: scripts/g68_gpu.sh). GPU only;
no API cost.

## Why
Every says-vs-does result on Claude Haiku 5.5 comes from one CAPE-Bench template, the only one where its violations
vary; an exploratory scan (scripts/259) found none more under a realistic autonomy prompt. Open agents violate on many
templates. The theory (Proposition 3, Corollary 2) says the forecast before the call is the agent's policy, which sampling
supplies; it predicts that the agent's own sampled next actions forecast its violation better than its self-report.

## Design (fixed)
Agents: Gemma-4, gpt-oss-20b, Qwen3.8, Muse-Glimmer. Units: G27/G27b's steps (CAPE-Bench episodes, up to and including
the first violating call) in template|tool|step strata holding both classes (G30's rule): 289, 136, 41 and 226 steps
(94, 82, 23 and 93 violations; 13, 11, 4 and 6 templates).
- **Says:** the agent's pre-call self-report, G30's raw read-out (signed sum of the step's 8 forward-looking answer angles),
  measured by G27/G27b before this test.
- **Does:** 8 sampled next actions from the exact step context (CAPE's message builder), each distinct drawn call read by
  the agent's own 8 at-call answers (4 each); G46's forecast, unchanged (scripts/226). The draw and read code is
  scripts/226 stage_other's unit() with the agent in its own seat. CAPE's checker labels each drawn call for a reported
  ceiling only.
Checks before the lock (CPU): replaying each step's prior calls reproduces CAPE's logged label of the step's call on all
692 units; the analysis runs on synthetic records. The GPU job first draws 4 units per agent into separate smoke files,
never analysed. A change forced by a smoke failure will be recorded as a deviation.

## Endpoints (AUROC within template|tool|step; cluster bootstrap over scenarios within each agent, 2000 resamples, seed 6800)
- **P1 (primary).** Mean over agents of (does − says): lower 95% bound > 0.
- **P2.** Mean over agents of does: lower 95% bound > 0.5.
- Holm over P1 and P2 at family-wise 0.05. Reported: each agent's values and intervals, the number of agents with
  does > says, the checker ceiling.

## Readings, fixed in advance
- P1 passes: across four open agents and their task families, what the agent does (its sampled next actions) forecasts
  its violation better than what it says before acting. With Claude Haiku 5.5's result, says vs does holds beyond one
  template and one agent.
- P1 fails: on open agents, sampled actions do not beat the self-report; the Claude result stays limited to its template.
