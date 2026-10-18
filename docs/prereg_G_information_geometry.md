# Preregistration G: the observable latent geometry of a model, training-free

Locked, and published, before the confirmatory analyses run. Theory: docs/theory_infogeom.md (Theorems 1–6).
Code: scripts/129_infogeom.py. **No model is trained or fine-tuned, and no second model stands in for the one under
study.** Each estimate uses only the outputs of the model it describes. Where the hidden states are visible (open
models), they serve only as the ground truth for the check.

Known before locking:
- every earlier result of this project;
- an exploratory pilot on DEVELOPMENT contexts (docs/research_ledger.md) and two dry runs of G1 on 800 dev contexts
  (layer chosen on 1,000 other dev contexts; results/t1/g1_open_geometry_DRYRUN_dev_v1_absF2.json and
  results/t1/g1_open_geometry_DRYRUN_dev.json):
  - ρ(answers, hidden) 0.40–0.79; ρ(text, hidden) 0.25–0.34; F1 and F3 pass for 8 of 8 models;
  - F2 in its first draft (exact − sampled, upper bound < 0.02 absolute) passed for only 5 of 8 (gpt-oss 0.025,
    OLMo 0.025, Phi-4 0.021). The absolute margin ignored that the models' ρ differ by a factor of two, so F2 was
    changed to the scale-free ratio below, AFTER the first dry run and BEFORE the lock. The second dry run gives the
    ratio's dev values: lower bounds 0.958–0.996 for 7 models, 0.938 for gpt-oss (fails);
  - the dev layer profile peaks at a middle or late stored layer for 7 of 8 models; for Mistral it peaks at the first
    stored layer (layer 8);
- a pilot of G3 on Qwen3-0.6B with generation seed 100 (results/t1/g3_exact_recovery_PILOT_seed100.json): Pearson
  0.00–0.36 below d(d+1)/2; 0.9915 at 1.1×; 0.99999997 at 1.5×; 0.99999999 at 2.0× (median cosine error 1.3e-5);
  baselines 0.40 (raw) and 0.11 (whitened). E1–E3 were written before the pilot and are not changed. The confirmatory
  run uses generation seed 0: fresh sequences;
- a calibration of G2's C1 bound on OPEN models only (192 random dev contexts, the same 24 questions, 4 + 4 simulated
  samples): split-half ρ 0.76–0.96, vs text 0.21–0.30. C1's bound was raised from 0.5 (draft) to 0.7, below the
  lowest open model;
- of Claude's data, only its structure: 192 contexts, 24 questions, exactly 8 samples per cell. No distance,
  correlation or coordinate from Claude's answers was computed before the lock.

## G1. Open models: the answer geometry is the hidden geometry (Theorems 1, 4, 5)
- **Models:** the 8 open chat models of L2 (gemma3_4b, gptoss20b, llama31_8b, mistral7b, olmo3_7b, phi4mini, qwen25_7b,
  qwen3_8b); their stored P(yes) on the 88-question battery and their stored hidden states (6 layers each).
- **Contexts:** the 1,440 CONFIRM contexts of L2 (never used to choose anything).
- **Layer:** for each model, the stored layer whose cosine geometry has the highest Spearman correlation with the exact
  answer geometry on 1,000 DEV contexts (numpy seed 0). Chosen on dev only.
- **Answer geometry:** d_Q(x, x') = ‖ϑ(x) − ϑ(x')‖, ϑ = 2 arcsin √P(yes) over the 88 questions (Theorem 1).
  Sampled version: n = 8 Bernoulli draws per cell (seed 129 + model index), Anscombe angles, debiased by 2m/(n + 1/2)
  (Theorem 5).
- **Hidden geometry:** 1 − cosine of the hidden states at the chosen layer (invariant under the orthogonal gauge).
- **Baseline:** 1 − cosine of the MiniLM text embeddings of the same contexts (surface text, not the model's state).
- **Statistic:** Spearman correlation over all pairs of contexts (RSA). No fitted parameter.
- **Uncertainty:** bootstrap over contexts (500 resamples, seeds 1291 + i, 1301 + i, 1311 + i).
- **F1 (answers beat surface text):** ρ(answers, hidden) − ρ(text, hidden). Pass: lower 95% bound > 0 for at least 7
  of the 8 models.
- **F2 (eight samples are enough):** ρ(sampled, n = 8) / ρ(exact). Pass: lower 95% bound > 0.95 for at least 7 of 8.
- **F3 (more questions, more state):** ρ(88 questions) − mean ρ(8 random questions; 5 draws). Pass: lower 95% bound > 0
  for at least 7 of 8.
- **Reported:** the dev layer profile (Theorem 4 predicts a peak in the middle-to-late layers, not the first stored
  layer); the convergence curve over 4, 8, 16, 32, 64, 88 questions; raw-P(yes) geometry.

## G2. Claude Haiku 5.5: its own observable geometry (Theorem 5)
- **Data:** the 192 CONFIRM contexts of L3 and Claude's 8 sampled answers to each of 24 battery questions (the
  existing L3 run; no new API calls).
- **Estimator:** debiased Fisher–Rao distances from the Anscombe angles; an answer that is neither yes nor no is not
  counted (the cell's n drops).
- **Split half:** in each cell, the 8 samples are permuted (seed 2290); samples 1–4 form half A, 5–8 half B. Each half
  gives a geometry D_A, D_B on the same contexts.
- **C1 (the geometry is a property of Claude, not of sampling noise):** Spearman ρ(D_A, D_B), bootstrap over contexts
  (500, seed 2291). Pass: lower 95% bound > 0.7.
- **C2 (it is not the surface text):** ρ(D_A, D_B) − mean of ρ(D_A, text) and ρ(D_B, text) (seed 2292). Pass: lower
  95% bound > 0.
- **Reported only (not an estimate of Claude's state):** the classical-MDS coordinates of Claude's geometry, its
  eigenvalues and the share of negative eigenvalue mass (0 for an exactly Euclidean geometry); the Spearman
  correlation between Claude's answer geometry and each open model's hidden geometry on the same contexts.

## G3. Exact recovery of the final hidden geometry from log-probabilities (Theorem 6)
- **Model:** Qwen/Qwen3-0.6B, float32, TF32 off (d = 1,024; vocabulary 151,936; tied embeddings, no output bias,
  RMSNorm ε = 1e-6). The model is the subject; it is not a surrogate for another model.
- **Contexts:** 9,000 sequences of 128 tokens, sampled from the model itself (temperature 1, no top-k or top-p, torch
  seed 0), every position a context: 1,152,000 contexts. 2,000 validation contexts (numpy seed 1) are held out; only
  for them is the true state read, for the check.
- **Observed:** the centered log-probabilities c(x) (all 151,936 tokens), projected on their rank-d span (top d right
  singular vectors of the first 3,000 contexts).
- **Solve:** whiten the projected B on the training contexts, then least squares for symmetric S in b_iᵀ S b_i = d by
  conjugate gradient (from S = 0; relative residual 1e-12 or 5,000 iterations).
- **Truth:** u = x / rms(x), x the input of the final RMSNorm (its gain is folded into the output map).
- **Statistic:** Pearson correlation between recovered and true cosines over all pairs of validation contexts.
- **N:** 0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2.0 × d(d+1)/2 = 524,800 training contexts (seed 3).
- **E1 (exact recovery):** at N = 2.0 × d(d+1)/2, Pearson > 0.99.
- **E2 (the counting threshold of Theorem 6):** median absolute cosine error at 2.0× is below one tenth of the
  smallest median error at any N < d(d+1)/2.
- **E3 (the sphere constraint adds information):** the recovered cosines beat both baselines without the constraint
  (cosines of the raw projected log-probabilities, and of the whitened ones).

## Readings, fixed in advance
- **G1 passes:** for open models, a model's latent geometry is read from its yes/no answers alone, in exact flat
  coordinates, with no training, far better than from the text; eight samples per question suffice; adding questions
  reveals more of the state.
- **G2 passes:** Claude has a stable observable latent geometry, measured from 8 sampled answers per question, which is
  not the surface geometry of the text. (Its relation to Claude's hidden states cannot be checked; the paper says so.)
- **G3 passes:** with full log-probabilities, the final hidden geometry is not estimated but recovered exactly, with
  the phase transition at d(d+1)/2 contexts predicted by counting.
- **Failures** are reported as such, with the same prominence.

## Limits stated in advance
- G1's ground truth is the hidden state at one stored layer; the theory relates answers to the state through the
  pulled-back Fisher metric (Theorem 4), not identity, so ρ < 1 is expected.
- G2 tests stability and distinctness, not agreement with Claude's (unavailable) hidden states.
- G3 needs full log-probabilities, which closed APIs do not expose (OpenAI gives at most 20); for d = 4,096 it needs
  about 1.7 × 10⁷ contexts.
