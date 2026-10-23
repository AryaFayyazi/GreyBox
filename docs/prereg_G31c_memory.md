# Preregistration G31c: how long the state remembers, and the noise that the filter assumed

Locked, and published, before any analysis of G27b's answers: gpt-oss-20b's battery exists on disk and has not been
read (beyond counting parsed answers and server errors), and Qwen3.8's and Muse-Glimmer's batteries do not exist yet.
Code: scripts/204_g31c_memory.py. No label is used anywhere in this test.

## Why
G27's Kalman forecast failed and its model checks failed (unstable identified dynamics; innovations far from their
nominal size). An exploratory, label-free look at Gemma-4's answers (after G27's locked analysis) found two causes.
(1) The noise: within a template and step, the variance of the answer angles is about 0.2 of the noise 1/(m + 1/2)
that the filter assumed. For nearly deterministic answers the exact variance of the Anscombe angle is far below that
value (at p = 0.01 and 4 answers, 0.013 against 0.222; Theorem 5′). The filter then took signal for noise.
(2) The memory: the answers' within-episode autocorrelation is 0.18 at one step, 0.11 at two, and about 0 from three
on. The state carries information for one or two steps; the deciding information is renewed at each step. Under such a
state, a lag-2 instrument (Proposition IV) is weak, and the identified F is unstable. This test checks both causes on
three more agents.

## Data and quantities (fixed; no label)
G27b's step batteries of gpt-oss-20b, Qwen3.8, Muse-Glimmer (pooled); Gemma-4 reported (seen). Each answer angle
minus its template x step mean (removes position and task). rho_h: within-episode autocorrelation at lag h = 1..5,
mean over the 8 questions. Ratio: unbiased within-cell variance of the angles (cells with at least 2 steps) over the
mean nominal noise 1/(m + 1/2), mean over the 8 questions.

## Endpoints (pooled: mean over the three agents; cluster bootstrap over scenarios within each agent, 2000, seed 3150)
- **A1 (primary).** rho_1: lower 95% bound > 0 (the answers measure a state that persists across steps).
- **A2.** rho_1 − rho_3: lower 95% bound > 0 (the memory decays within three steps).
- **A3.** Ratio: upper 95% bound < 1 (the nominal noise overstates the real sampling noise).
- **Reported.** rho_h for h = 1..5 per agent and pooled.

## Readings, fixed in advance
- A1 and A2 pass: the state read through the API has a short memory, so answers more than a step or two before a call
  cannot forecast it; this, not a failure to read the state, explains why G27's forecast failed.
- A3 passes: a filter must not use the nominal noise for nearly deterministic answers; it needs the exact or an
  empirically estimated noise (G31's split halves give it).
- A1 fails: the answers at successive steps do not share a persistent state beyond position and task.
- A3 fails: the nominal noise is right for these agents, and G27's misspecification on Gemma-4 does not generalize.
