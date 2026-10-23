# Preregistration G40: a closed agent's state before its call, beyond position, confirmed on fresh episodes

Locked, and published, before any episode of this test exists. Code: scripts/213_g40_gpt_precall_confirm.py. Paid API
(OpenAI); fixed credit.

## Why
G38 (preregistered) found that GPT-6-luna's answers about its next action, read before the call exists, rank the
coming violation (0.765), and an exploratory control after it found 0.927 among calls at the same template, tool and
position, where position alone cannot help. An exploratory finding needs confirmation on new data.

## Design (fixed)
Fresh GPT-6-luna episodes with scripts/80's loop (MockEnv labels, one call per turn, reasoning none, max 20 turns),
seed 52 (T11 used 50, G39 uses 51), no gate, on all 42 x- scenarios of each of the 6 templates in which it violated in
T11 (252 episodes). Before every turn, in side branches, G38's PRE battery: G27's pre-call frame and 8 forward-looking
questions, 1 request of n = 4 each, temperature 1. Units (T11's rule): executed non-read calls, up to and including the
first unsafe one. Score: G27's raw read-out (signed sum of the 8 Anscombe angles) measured before the unit's turn;
nothing fit; no label used. Position: the unit's ordinal among the episode's units (G38's control).

## Endpoints (cluster bootstrap over scenarios, 2000, seed 4000)
- **C1 (primary).** AUROC within template x tool x position: lower 95% bound > 0.50.
- **C2.** AUROC within template x tool: lower 95% bound > 0.50.
- **Reported.** Position alone within template x tool; strata with both classes and their violations.

## Readings, fixed in advance
- C1 passes: before its call exists, a closed agent's state read through its own API ranks the coming violation
  beyond the call's position: forecasting from the reconstructed state, confirmed on a closed model.
- C1 fails: G38's exploratory 0.927 does not replicate at this power.

## Cost and conduct
About $0.01 per episode (G39's measured episode price plus 8 requests per turn): about $2.5–3. The OpenAI ledger cap
$12.70 (the fixed credit). Runs after G39 (one process per provider); checkpoint per wave of 6 episodes. If the cap
stops it, the analysis uses the episodes written, and the record says how many.
