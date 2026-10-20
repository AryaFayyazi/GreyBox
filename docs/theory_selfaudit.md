# Self-audit and resampling: an exact law for API-only prevention

Written 2026-10-19, before any self-audit result on any agent existed (the Gemma-4 development run was in progress;
only two 40- and 6-unit smoke tests had been seen, with an earlier, discarded prompt and with the current one).

## Setting
An agent at state x (the visible conversation before a step) proposes an action A ~ π(· | x). V(A, x) ∈ {0, 1} marks a
policy violation (CAPE's checker). R(x) = P(V = 1 | x) is the agent's own violation probability at x; it is a
property of the agent's sampling, measurable by resampling the agent at x through its API.

Self-audit: in a side branch (a separate API call; the real conversation never sees it), the agent is shown its own
proposed call and asked whether the call complies with its policy. F ∈ {0, 1} is the flag (the majority of n sampled
answers is "no"). Write τ(x) = P(F = 1 | V = 1, x) and φ(x) = P(F = 1 | V = 0, x).

Protocol S_M (audit and resample, at most M draws): draw A_1; audit it; if not flagged, execute it; if flagged, draw
again; after M flagged draws, escalate (hold the step for a human). M = 1 is the plain gate.

## Proposition (prevention law)
Assume the draws A_1, …, A_M are i.i.d. from π(· | x) and the audit flags are independent across draws given (x, V).
With f(x) = R τ + (1 − R) φ (a draw is flagged) and q = 1 − f:
  P(a violation executes at x)  = R (1 − τ) (1 − f^M) / q,
  P(the step is escalated)      = f^M,
  P(a safe action executes)     = (1 − R)(1 − φ)(1 − f^M) / q.
Proof: the first unflagged draw is executed; draw j is the first unflagged one with probability f^{j−1} q, and given
that it is unflagged it violates with probability R (1 − τ) / q. Sum over j ≤ M. ∎

Consequences:
- Without the protocol a violation executes with probability R. The plain gate (M = 1) gives R (1 − τ) and escalates
  f. More draws turn escalations into executed actions: escalation falls as f^M, and the executed-violation rate rises
  towards R (1 − τ) / q, the share of violations among unflagged draws.
- Everything on the right is measurable through the agent's API alone: R by resampling, τ and φ by auditing. So the
  safety and the cost of S_M are predicted BEFORE the protocol is run; a live run tests the prediction.
- The only model in the loop is the monitored agent itself. Nothing is trained.

## What the data must show for this to matter
1. A knowing–doing gap: τ well above φ. The agent flags its own violating actions, when asked, far more often than its
   safe ones, although it proposed them.
2. Stochastic violations: R(x) well below 1 at most states where violations occur (development rollouts: Gemma-4
   median 0.31, gpt-oss median 0.75 at turn starts with an observed violation), so resampling yields safe actions.
3. The law's predictions match a live run of S_M within sampling error.

## Limits
- The independence of audit flags across draws is an assumption; a live run checks the law it implies.
- τ, φ and R vary with x; the law is applied state by state and averaged, not with pooled rates, when R(x) is known.
- An agent that knows it is audited could answer strategically. These agents are not adversarial; an adversarial
  agent is out of scope.
