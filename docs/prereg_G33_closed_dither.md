# Preregistration G33: the closed agent's own answers, dithered

Locked, and published, before any answer of this test exists and before G31's result (G31 runs the same idea on open
agents). Code: scripts/205_g33_closed_dither.py. Paid API; runs only after the user adds credit.

## Why
On tau2-bench airline, Closed-A's own 8 answers rank its policy violations at 0.704 within tool (G18). G32 found why
this setting is the weakest: Closed-A's 4 answers agree on 96–97% of questions, so each question gives about one bit,
and the read-out's label-free SNR is the lowest of all eight settings (−0.48). The theory of G31: a deterministic
yes/no answer only signs a hidden quantity; meaning-preserving variations of the question act as dither and recover
its size; repeating one wording does not.

## Design (fixed)
G18's 360 writes (50 violations, 27 tasks; G18's sample), prefixes, frame, operator note, model and settings
(reasoning off, temperature 1, max 16 tokens), through Closed-A's own API. Per question:
- D16: 4 fixed wordings written before any answer (wording 0 is G18's; 2 paraphrases; 1 with reversed polarity,
  flipped back), 4 answers each, pooled.
- R16: wording 0, D's 4 answers plus 12 more.
- R4: wording 0, D's 4 answers (a replicate of G18's design). G18_R4: G18's stored answers (2026-10-21).
Read-out: G18's fixed signed sum of the 8 answer angles; nothing fit; no label used.

## Endpoints (within-tool AUROC; cluster bootstrap over tasks, 2000, seed 3330)
- **Q1 (primary).** AUROC(D16) − AUROC(G18_R4): lower 95% bound > 0.
- **Q2 (secondary).** AUROC(D16) − AUROC(R16): lower 95% bound > 0 (dither beats repeats at equal budget).
- **Reported.** AUROC of every arm with D16's interval; R16 − R4; D16 − R4; label-free SNR and unanimity per arm.

## Readings, fixed in advance
- Q1 passes: the closed agent's weakest setting was limited by the measurement; dithered questions through its own API
  read more of its state.
- Q2 passes: the gain comes from dither, not from more answers.
- Q1 fails: at this power, dither does not improve the closed agent's read-out on tau2-bench.

## Cost and conduct
About 224 calls per write (about $11 at G18's measured rate with prompt caching). One process; checkpoint per wave; the
shared ledger with an explicit cap. A pilot of at most 6 writes measures the cost first; the pilot's answers are kept
and counted (they are part of the 360).
