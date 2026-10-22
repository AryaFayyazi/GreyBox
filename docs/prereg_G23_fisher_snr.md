# Preregistration G23: does the theory predict, without labels, which questions monitor?

Locked, and published, before any universal-question answer (G22) is analysed: the answer files of G22 exist on disk
(its GPU job is running) but have not been read; no AUROC of any universal question has been computed. Code:
scripts/190_g23_fisher_snr.py.

## Why
Three independent reviews (twice) said the geometry theory does no work in the monitor: a mean of raw P(yes) ranks
calls as well as the arcsine read-out. The theory does make a prediction that raw probabilities cannot: in arcsine
(Fisher-Rao) coordinates the sampling noise of an answer angle is known and constant, Var ~= 1/(m + 1/2) for m parsed
answers (Anscombe), whatever the answer law. So each question's signal-to-noise ratio across UNLABELLED calls,
SNR_j = (pooled within-stratum variance of theta_j) / mean 1/(m + 1/2) − 1, is computable with no label, and it says
how much of the question's spread is not sampling noise. On raw P(yes) the noise p(1−p)/m depends on p and no such
correction exists.

## Prediction (fixed; no label is used to compute SNR or the weights)
Questions with higher SNR rank violations better; weighting the questions by max(SNR, 0) gives a better read-out than
weighting them equally.

## Data
Gemma-4's answers (thinking on, 4 per question) to 16 questions about the same calls: the 8 tailored questions (G16 on
CAPE-Bench; G21 on tau2-bench) and the 8 universal questions (G22). Five targets: CAPE-Bench held-out calls of
gpt-oss-20b (599), Qwen3.8 (373), Muse-Glimmer (600) and Claude Haiku 5.5's actions (495) (strata: template; clusters:
scenario; units as in G16), and Claude's 360 tau2-bench writes (strata: tool; clusters: task).

## Endpoints (mean over the five targets; cluster bootstrap within each target, 2000, seed 2300)
- **F1 (primary).** Spearman correlation, over the 16 questions, between SNR_j and the question's within-stratum AUROC
  (oriented by the question's sign): lower 95% bound > 0.
- **F2 (two-sided).** AUROC of the SNR-weighted read-out of the 16 questions minus the equal-weighted read-out.
- **F3 (two-sided).** SNR-weighted minus weighted by the raw-P(yes) within-stratum variance (no noise model).
- **Reported.** Spearman for raw-P(yes) variance; per target; the per-question SNR and AUROC; tailored-8 and universal-8.

## Readings, fixed in advance
- F1 passes: the theory's label-free noise model tells which questions carry the state that monitoring needs; the
  coordinates do work that raw probabilities cannot.
- F1 fails: across questions, label-free signal does not predict monitoring value; questions can vary across calls
  for reasons unrelated to the policy. The paper says so.
- F2 above 0: a label-free, theory-derived weighting improves the monitor; below 0: it hurts (spread unrelated to the
  policy is up-weighted); the paper reports either.
- F3 says whether the noise model, not just variance, matters.

## Known before locking
The tailored answers of G16/G21 and their aggregate AUROCs; per-question AUROCs of Gemma-4's tailored answers on tau2
(G21 controls). G22's universal answers have not been read. A dry run on random universal answers (removed) checked
that the code runs; on random counts that are not binomial, SNR up-weights the noise, as the prediction's failure mode.
