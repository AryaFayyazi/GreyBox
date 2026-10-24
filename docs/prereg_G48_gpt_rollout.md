# Preregistration G48: the policy forecast on a second closed agent, across six kinds of task

Locked, and published, before any G48 draw exists. Code: scripts/229_g48_gpt_rollout.py (uses scripts/226's read-out and
forecast functions, hashed in the G46 lock). Paid (OpenAI) plus GPU. Ledger cap 26.80 (account balance $11.08 on 2026-10-24 over a ledger of $16.26, less a margin).

## Why
G46 tests, on Claude Haiku 5.5 and one kind of task, whether the agent's own sampled next actions, read by its own
answers, forecast its coming violation (corollary "reading the policy by sampling"). G48 asks the same of GPT-6-luna, on
G45's 504 episodes over six CAPE-Bench templates, where its self-report already forecasts (G40, G45).

## Design (fixed)
- Units: G45's GPT-6-luna units (non-read calls up to and including the first unsafe one, in template|tool strata
  holding both classes): 1,134 calls, 131 violations, 6 strata. Position = the unit's ordinal in its episode.
- own (the method): before the unit's turn, K = 8 next actions drawn from the agent through its API (one request, n = 8,
  its loop's settings, no seed); each distinct drawn call read by the agent's own 8 at-call answers (STATE_QUESTIONS,
  STATE_FRAME; one request of n = 4 per question); forecast = mean read-out over the 8 draws, a draw that ends the
  episode scoring the read-out's minimum. No label, no training, no other model.
- report: the agent's self-report (G45's pre-call battery, G27's raw read-out).
- ceiling (reported): share of the 8 draws that CAPE's checker calls unsafe.
- other (a control, GPU): Gemma-4 (thinking on) in GPT-6-luna's seat, its own 8 draws from the same replayed context,
  read by its own 8 answers x 4 (scripts/226's other stage on these units).
- Why the identity can be tested here: GPT-6-luna's loop calls the API with parallel_tool_calls = False (one tool call
  per turn), so the replayed history is the exact one; on Claude Haiku 5.5 (G46b) the replay dropped refused extra calls.

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 4800)
- **Q1 (primary).** own, AUROC within template|tool: lower 95% bound > 0.50.
- **Q2.** own, AUROC within template|tool|position: lower 95% bound > 0.50.
- **Q3 (reported, two-sided).** own − report within template|tool.
- **C0.** Calibration: mean Y − mean f_own (f_own = unsafe share among the agent's non-read draws, units with >= 2):
  95% interval inside [−0.05, +0.05].
- **B3.** The policy-gap identity (scripts/230's formulas with Gemma-4's draws): observed − predicted Brier excess of the
  other model, 95% interval inside [−0.02, +0.02].
- **Reported.** B1 observed excess, predicted excess, B2 policy gap (both noises removed); other's AUROCs.

## Readings, fixed in advance
- Q1 and Q2 pass: on a second closed agent and six kinds of task, the agent's own policy, read by sampling through its
  API, forecasts its coming violation, beyond position.
- Q1 fails: the policy forecast does not carry to GPT-6-luna.
- C0 and B3 pass: on a second closed agent, with a faithful replay, the proposition predicts in size how much another
  model loses before the call (near zero if the two policies agree, as G45 suggests).

## Cost and conduct
About $0.0065 per unit (about $7.4); OpenAI ledger cap set to the account balance. Checkpoint per wave; resumes after an
API error; if the budget stops it, we analyse the complete units and record this.
