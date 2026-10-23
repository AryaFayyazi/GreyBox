# Preregistration G35: does the monitor come from the reconstructed state?

Locked, and published, before any quantity below is computed. Code: scripts/207_g35_readout_reads_state.py. CPU only;
existing data (G15's held-out batteries and the stored decision states).

## Why
The chain of the paper: (1) the agent's own answers reconstruct its hidden geometry (G1; G34 for a closed model);
(2) the monitor reads the violation from that reconstruction. G15 showed that the read-out ranks violations nearly as
well as a probe on the hidden state. It did not show that the two read the same variable: both could merely
correlate with the label. This test asks whether the read-out follows the hidden state's violation coordinate beyond
the label, and whether that shared variable carries the read-out's monitoring power.

## Data (fixed)
G15's held-out calls of Gemma-4 (600), gpt-oss-20b (599), Qwen3.8 (373), Muse-Glimmer (600). s: G15's fixed read-out
of the agent's own 8 answers (4 each). H: the agent's stored decision state, layer index 3. P: G15's probe on H
(32 PCs, C = 0.1, fit on development calls): an instrument that measures the hidden state's violation coordinate; it is
never part of the method.

## Endpoints (pooled: mean over the 4 agents; cluster bootstrap over scenarios within each agent, 2000, seed 3500)
- **L1 (primary).** Spearman between s and P within template and within class (cells with >= 5 calls; ranks within
  each cell, pooled): lower 95% bound > 0. Among calls with the same label, the read-out follows the hidden state's
  violation coordinate.
- **L2.** Spearman between the answer geometry (Euclidean, 8 angles) and the decision-state geometry (cosine) over
  pairs of calls with the same template and label (cells capped at 200 calls, seed 0): lower 95% bound > 0.
- **L3.** The share of the read-out's within-template AUROC above 0.5 that is lost when the part of s that P predicts
  (rank regression within template) is removed: lower 95% bound > 0.
- **Reported.** Per agent; the read-out's and the probe's AUROCs; the number of cells.

## Readings, fixed in advance
- L1 passes: the reconstruction carries the hidden state's own violation variable, not only the label.
- L3 passes: the read-out's monitoring power runs through that hidden variable: the monitor comes from the
  reconstructed state.
- L2 passes: within a template and label, the answer geometry tracks the decision-state geometry.
- L1 fails: the read-out and the probe agree only through the label; the paper does not claim that the monitor comes
  from the reconstructed state.

## Known before locking
The read-out's and the probe's within-template AUROCs (G15). No correlation between s and P, and no geometry at the
decision, has been computed. A synthetic check (removed) verified the code: about 0 under no link, L1 0.51 and L3 0.70
with a planted shared variable.
