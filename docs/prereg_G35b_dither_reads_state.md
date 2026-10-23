# Preregistration G35b: with a better measurement, does the read-out follow the hidden state beyond the label?

Locked, and published, before any answer of G31 exists (G31's jobs have not started). Code:
scripts/208_g35b_dither_reads_state.py. CPU only.

## Why
G35 found that 20% of the read-out's monitoring power runs through the hidden state's violation coordinate (L3
pass), but that within a template and label the 4-answer read-out does not follow that coordinate (L1 fail). Within
such a cell the answers are nearly unanimous (G32), so the read-out barely varies: a measurement limit. G31 re-measures
steps with 4 wordings x 4 answers (dither). If the limit is the measurement, the dithered read-out should follow the
hidden coordinate within a cell.

## Data (fixed)
G31's batteries (gpt-oss-20b, Qwen3.8, Muse-Glimmer, Gemma-4), arms R4 (G27's design) and D16. P: G15's probe on the
stored decision state (layer index 3, PCA 32, C = 0.1, fit on development calls), applied to steps with a valid
stored state that are not development calls; an instrument for the hidden state, never part of the method. Cells:
template x tool x step x label with >= 3 steps.

## Endpoints (pooled: mean over agents; cluster bootstrap over scenarios within each agent, 2000, seed 3550)
- **K1 (primary).** L1 of D16 (Spearman between the read-out and P within cells, ranks within each cell, pooled):
  lower 95% bound > 0.
- **K2.** L1(D16) − L1(R4): lower 95% bound > 0.
- **Reported.** L3 per arm and its difference; per agent; steps with a valid state.

## Readings, fixed in advance
- K1 passes: with a dithered measurement the agent's own answers follow its hidden state's violation coordinate beyond
  the label: the monitor reads the reconstructed state, not only the outcome.
- K2 passes: the 4-answer design was the limit.
- K1 fails: even dithered, the read-out and the hidden coordinate agree within a cell no better than chance at this
  power; the claim stays at G35's level (a fifth of the monitoring power runs through the hidden coordinate).

## Known before locking
G35's results. No G31 answer exists. A synthetic check (removed) verified the code: a coarse noisy arm gives a lower
L1 than a graded one.
