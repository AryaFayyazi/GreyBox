# Preregistration G31: are the within-step effects limited by the measurement? Repeats against dither

Locked, and published, before any answer of this test exists. Code: scripts/203_g31_dither.py.

## Why
Within a step, the answers before a call rank its violation at about 0.67 (Gemma-4; G30 tests three more agents) and
track the true hidden state at a same-step Spearman of about 0.16. These numbers are measured with 4 answers per
question. G32 found that the answerers are nearly deterministic: the 4 answers agree on 62–97% of questions. A
deterministic yes/no answer is a quantizer: it gives the sign of a hidden quantity x, not its size. Repeating the same
question then adds little; more answers cannot recover x.

## Theory being tested (stated before the data)
Signal processing has a standard remedy: dither. Let the answer to wording v be 1{x + e_v > 0}, where e_v is the
shift that the wording brings, symmetric with distribution G. The mean answer over wordings is G(x / tau), smooth and
increasing in x. So averaging over meaning-preserving perturbations of the question recovers the size of x that a
fixed question hides. A wording with reversed polarity (answer flipped) also cancels a bias to say yes. Predictions:
(i) dithered answers agree less (lower unanimity); (ii) at an equal answer budget, the dithered measure ranks
violations and tracks the true state better than repeats of one wording; (iii) both 16-answer measures do at least
as well as G27's 4-answer design.

## Design (fixed)
- Steps: every step of G27's (Gemma-4) and G27b's (gpt-oss-20b, Qwen3.8, Muse-Glimmer) batteries in a template x tool
  x step stratum that holds both classes (chosen from labels and strata only): Gemma-4 289 steps (94 violations),
  gpt-oss-20b 136 (82); Qwen3.8 and Muse-Glimmer: their batteries do not exist yet.
- Each step: G27's prefix and frame, the agent's own API, same sampling. For each of G27's 8 questions:
  R (repeats): G27's wording, 16 answers. D (dither): 4 fixed wordings written before any answer (wording 0 is G27's;
  2 paraphrases; 1 with reversed polarity, flipped back), 4 answers each, pooled. Every answer is stored.
- Read-out (nothing fit): G27's signed sum of the 8 answer angles; arms R4 (first 4 of R: G27's design), R16, D16.
- Truth: the stored decision states, layer index 3 (as G28 and G30).

## Endpoints (pooled: mean over agents; cluster bootstrap over scenarios within each agent, 2000, seed 3300)
Within template x tool x step:
- **M1 (primary).** AUROC(D16) − AUROC(R4): lower 95% bound > 0.
- **M2 (secondary).** AUROC(D16) − AUROC(R16): lower 95% bound > 0 (dither beats repeats at equal budget).
- **M3 (secondary).** Same-step Spearman with the true decision-state geometry, D16 − R4: lower 95% bound > 0.
- **Reported.** Per agent; AUROC(R16) − AUROC(R4); AUROC of D4 (wording 0 of D, a replicate of R4); unanimity, SNR
  and split-half reliability per arm (R: answers 1–8 against 9–16; D: wordings 1–2 against 3–4) with Spearman–Brown;
  the RSA of D16 divided by the square root of its reliability.

## Readings, fixed in advance
- M1 passes: the within-step numbers of G27/G30 are limited by the measurement, not by what the agent's state holds;
  a better measurement through the same API reads more of the state.
- M2 passes: the gain comes from dither: varying the question, not repeating it, reads the size of the hidden quantity.
  This is the design for closed models, whose answers are the most deterministic (G32).
- M3 passes: the dithered measure tracks the true hidden state better within a step.
- M1 fails: at this power, 16 answers do not rank violations better than 4; the within-step numbers are not
  measurement-limited at this scale.

## Known before locking
G27's 4-answer read-outs on Gemma-4 and their within-step AUROC (0.674); gpt-oss-20b's G27b battery exists and has
not been read beyond parse counts. A dry run on fake answers with real step keys (removed; no result file) checked the
code: with a deterministic quantizer and a planted signal, M1 and M2 detect the dither gain and the RSA is near 0.
Deviation from the order of G27b: Gemma-4's re-measure runs last because it is the slowest server.
