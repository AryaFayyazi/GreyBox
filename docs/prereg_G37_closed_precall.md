# Preregistration G37: the closed agent's state before its call exists

Locked, and published, before any answer of this test exists and before the results of G30 (the same question on
open agents) and G31 (dither on open agents). Code: scripts/210_g37_closed_precall.py. Paid API; runs after credit.

## Why
The chain of the paper: reconstruct the agent's state through its own API, then forecast and prevent its violations.
On open agents, the answers asked before a call rank that call among calls at the same step (Gemma-4, exploratory;
G30 tests three more agents). This test asks the same of the closed agent, before its call exists, on the benchmark
where its violations vary within a kind of action (tau2-bench airline; on CAPE-Bench its violations fall in one kind of
task).

## Design (fixed)
G18's 360 writes (50 violations, 27 tasks; G18's sample). At each write, before it exists: G18's prefix (the
conversation up to the agent's turn), G18's operator note, G27's pre-call frame ("[Operator check before your next
step. You have not acted yet.] Question about your next action: ..."), the call NOT shown. 8 forward-looking
questions: G18's policy questions about "your next action" (wording 0), and G33's 3 other wordings made forward-looking
in the same way (fixed before locking). Closed-A's own API, G18's model and settings.
- R4: wording 0, 4 answers (G27's design). D4: wording 0's first answer plus 1 answer of each other wording (equal
  budget; the reversed wording flipped back). Read-out: G18's fixed signed sum of the 8 answer angles; nothing fit.

## Endpoints (within the coming write's tool, as G18; cluster bootstrap over tasks, 2000, seed 3700)
- **P1 (primary).** AUROC of R4: lower 95% bound > 0.50. The state read before the call ranks the coming violation.
- **P2 (secondary).** AUROC(D4) − AUROC(R4): lower 95% bound > 0.
- **Reported.** D4's AUROC; pre-call R4 minus G18's at-call read-out; unanimity.

## Readings, fixed in advance
- P1 passes: through its own API, before the call exists, the closed agent's answers about its next action rank the
  coming violation: forecasting from the reconstructed state carries to a closed agent.
- P2 passes: dither helps the closed agent at equal budget before the call.
- P1 fails: at this power the closed agent's pre-call answers do not rank its coming violation; the closed-agent claim
  stays at the call (G18).

## Cost and conduct
56 calls per write (about $2.7 at G18's measured rate). One process; checkpoint per wave; explicit ledger cap; a pilot of
at most 8 writes measures the cost first (its answers are kept and counted).
