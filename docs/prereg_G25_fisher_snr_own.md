# Preregistration G25: does the theory predict which questions monitor, on the method's own answers?

Locked, and published, before G23's result exists and before any per-question AUROC of these answers is computed,
except tau2-bench (G18 reported its per-question AUROCs). Code: scripts/192_g25_fisher_snr_own.py (calls G23's locked
scripts/190 functions unchanged).

## Why
The method uses only the agent's own API and its tokens. G23 tests the theory's label-free prediction on another
model's answers (Gemma-4, a comparison setup). G25 runs the same test on each agent's OWN answers.

## Data (each agent's own 8 answers x 4 about its own calls; nothing is fit; no label is used for SNR or weights)
CAPE-Bench (G15): Gemma-4 (600), gpt-oss-20b (599), Qwen3.8 (373), Muse-Glimmer (600), Claude Haiku 5.5 (495;
strata template, clusters scenario). AgentDojo (G17): Gemma-4 (816), gpt-oss-20b (821; strata suite, clusters
episode). tau2-bench (G18): Claude Haiku 5.5 (360; strata tool, clusters task). Eight targets.

## Endpoints (G23's, mean over the eight targets; cluster bootstrap within each target, 2000, seed 2500)
- **F1 (primary).** Spearman correlation, over the 8 questions, between the label-free Fisher-Rao SNR and the
  question's within-stratum AUROC: lower 95% bound > 0.
- **F2 (two-sided).** SNR-weighted read-out − equal-weighted read-out (here over the 8 own questions; G23's code names
  it "16" because G23 pools two sets of 8).
- **F3 (two-sided).** SNR-weighted − raw-P(yes)-variance-weighted.

## Readings, fixed in advance
As G23: F1 passes, the theory's noise model tells, with no label, which of the agent's own questions carry the
monitoring signal; F2 above 0, a label-free theory-derived weighting improves the method's own read-out.

## Known before locking
Aggregate AUROCs of these read-outs (G15, G17, G18); per-question AUROCs on tau2 (G18). G23's result does not exist.
