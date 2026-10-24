# Preregistration G58: the forecast needs the agent's policy, not its reader

Locked, and published, before any G58 answer exists. Code: scripts/243_g58_other_reads_draws.py. Paid (OpenAI).

## Why
A reviewer noted that the policy forecast scores sampled calls with an at-call read-out, while the text monitor sees no
call, and asked for another model reading Claude's drawn calls. The theory predicts the outcome: the forecast is the
agent's policy (Proposition 3), and at the call any capable reader suffices (Corollary 1). So what makes the forecast
work is sampling the agent's policy through its API; who reads the sampled calls should not matter.

## Design (fixed)
G46c's units (Claude Haiku 5.5, exact contexts). Each distinct call among Claude's own 8 draws (G46c's own stage) is shown
to GPT-6-luna as a proposed call after the transcript (reasoning medium; probability 0–100; unparsable -> 0.5).
Forecast = mean over the 8 draws (a draw that is not a call scores 0). Compared with Claude's own reads of the same draws
(G52's own forecast) and with GPT-6-luna reading the transcript alone (G55).

## Endpoints (cluster bootstrap over scenarios, 2000 resamples, seed 5800; AUROC within template|tool|position)
- **D1 (primary).** Draws read by GPT-6-luna − transcript read by GPT-6-luna: lower 95% bound > 0 (sampling the agent's
  policy is what forecasts).
- **D2.** Draws read by GPT-6-luna − draws read by Claude: 95% interval inside ±0.05 (the reader does not matter).

## Readings, fixed in advance
- D1 and D2 pass: the forecast needs the agent's policy, read by sampling through its API; any capable reader can score the
  sampled calls, as the theory predicts.
- D1 fails: GPT-6-luna's reading of the drawn calls adds nothing over the transcript.

## Cost
About $0.0005 per distinct drawn call (about $1); OpenAI ledger cap 26.80.
