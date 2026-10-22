# Preregistration G16: does the monitor need the agent's OWN answers, or does another model's answers to the same questions do as well?

Locked, and published, before any G16 answer exists. Code: scripts/176_other_answerer.py (answers), scripts/177_g16_analysis.py
(analysis), web/build_observable_ref.py (questions), scripts/171_decision_greybox_LOCKED.py (states, files, pilot).

## Why
G15 showed that a fixed read-out of an agent's own answers to 8 questions about its proposed call ranks its violations
as well as a probe trained on its hidden states. A reviewer asked whether the agent's OWN answers matter, or whether any
model answering the same questionnaire does as well (earlier development data: "verification, not self-knowledge").

## Design
- Same held-out calls as G15 (G14's battery files; the gpt-oss call without a stored state dropped; Claude's 5 pilot
  actions dropped), same 8 questions and frame, 4 answers each, same fixed read-out (mean violation-direction angle).
- The other answerer reads the agent's conversation up to the call in its own chat format, then the question:
  Gemma-4 answers about gpt-oss's, Qwen3.8's, Muse-Glimmer's and Claude's calls; Muse-Glimmer answers about Gemma-4's.
- Within-template AUROC; scenarios resampled within each agent, 2,000 resamples, seed 1780.

## Endpoints (two-sided; the direction is not predicted)
- **O1.** Mean over the four open agents of [own − other], with 95% interval. Reading, fixed now: lower bound > 0, "the
  agent's own answers matter"; upper bound < 0, "another model's answers do better"; otherwise, "no detectable difference".
- **O2 (reported).** Claude's restart template: own − other (Gemma-4), with 95% interval.

## Readings
- **Own answers matter:** the title's "own answers" is earned: the agent's answers carry what another model's do not.
- **No detectable difference:** the monitor is a fixed questionnaire that any capable model can answer about the call;
  the paper says so and the title drops "own".
- **Another model does better:** the questionnaire, not self-knowledge, carries the signal; the paper says so.

## Known before locking
G15's own-answer results (all calls); the exploratory read-out variants (results/t1/g15_readout_variants_exploratory.json).
No other-answerer answer exists. Development cross-audit (G9 period, one question): another model judged Gemma-4's calls
at least as well as gpt-oss judged its own.
