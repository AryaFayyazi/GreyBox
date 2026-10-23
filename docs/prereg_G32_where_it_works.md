# Preregistration G32: does a label-free quantity from the agent's own answers say where the read-out works?

Locked, and published, before any cell-level read-out AUROC or SNR below is computed. Code:
scripts/202_g32_where_it_works.py.

## Why
The read-out's strength depends on the setting: within-stratum AUROC near 0.99 for Gemma-4 on CAPE-Bench, 0.70 for
Closed-A on tau2-bench. A monitor that works in some settings is usable only if one can tell, without labels, where it
works. The theory gives such a quantity. In arcsine coordinates the sampling noise of each answer angle is known,
1/(m + 1/2), so the variance of the read-out beyond sampling noise (SNR) can be computed from the answers alone. If
the read-out does not vary beyond noise, it cannot rank anything. The theory also gives a ceiling (Gaussian, equal
variance; within-class variance >= noise): AUROC <= Phi(sqrt(SNR+ / (2 p (1-p)))), with p the violation rate.

## Data (fixed)
G25's targets: each agent's OWN answers (8 questions x 4 answers) about its own calls. CAPE-Bench: Gemma-4,
gpt-oss-20b, Qwen3.8, Muse-Glimmer (strata: template|tool), Closed-A (495 actions); AgentDojo: Gemma-4, gpt-oss-20b
(strata: suite); tau2-bench: Closed-A's 360 writes (strata: tool). Cells: (target, stratum) with >= 5 violations and
>= 5 non-violations (29 cells; counted from labels and strata only, before locking).

## Quantities (per cell; the score and SNR use no label)
s = sum_j sigma_j theta_j / K (the fixed read-out); N = sum_j [m_j > 0] / (m_j + 1/2) / K^2;
SNR = Var_cell(s) / mean_cell(N) - 1; AUROC_cell of s.

## Endpoints (bootstrap over cells, 2000, seed 3200)
- **C1 (primary).** Spearman, over cells, between SNR and AUROC: lower 95% bound > 0.
- **C2 (secondary).** Mean AUROC of cells with SNR > 0 minus that of cells with SNR <= 0: lower 95% bound > 0.
- **Reported.** Spearman of raw variance (no noise model), of cell size, and of base rate with AUROC; partial
  Spearman of SNR given size and base rate; share of cells with AUROC <= ceiling + 0.05; the analysis at >= 3 of each
  class (36 cells); per target: the read-out's pooled within-stratum SNR and within-stratum AUROC; every cell.

## Readings, fixed in advance
- C1 passes: from the agent's answers alone, without labels, one can tell where the read-out will rank violations
  well. The setting-dependence of the monitor is measurable in advance, not a hidden failure.
- C2 passes: SNR <= 0 flags settings where the read-out should not be trusted.
- C1 fails: at this power the label-free quantity does not track where the read-out works.
- The ceiling check says whether the observed AUROCs are near the limit that the measurement noise allows.

## Known before locking
Per-target AUROCs of the read-out (G15, G17, G18) and G25's per-question SNRs and AUROCs per target. No per-cell
read-out AUROC or SNR has been computed. A synthetic check (random binomial answers; removed) checked the code: SNR is
slightly below 0 under no signal (the Anscombe noise is slightly above the exact variance away from p = 1/2), and the
ceiling is close to the observed AUROC under a planted signal.
