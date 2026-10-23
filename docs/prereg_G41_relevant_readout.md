# Preregistration G41: read the state along the directions the call can violate

Locked, and published, before any per-tool or relevant-question read-out below is computed. Code:
scripts/216_g41_relevant_readout.py. CPU only; existing data (no credit remains for new closed-model answers).

## Why
On tau2-bench, Claude Haiku 5.5's equal-weight read-out of 8 policy questions ranks its violations at 0.70 (G18), and
dither did not raise it (G33). An exploratory per-question look (after G33's and G37's locked analyses) found that the
signal sits in questions that apply only to some kinds of call (cancellation conditions 0.63; payment rules 0.61 at the
call) while others never apply to a given call and add only noise. At a proposed call the tool is known, and the policy
says which rules can apply to it. Measuring along the relevant directions is the same principle as choosing a
measurement basis in tomography.

## Read-out (fixed from the policy text; nothing fit; no label)
For a call with tool T: the mean signed angle of the general questions (0: complies, sign −1; 7: forbidden) and of the
rule questions the policy ties to T: cancel_reservation {1, 3, 5}; update_reservation_flights {2, 3, 4};
book_reservation {4, 5}; update_reservation_baggages {3, 4, 6}; update_reservation_passengers {3, 6}.

## Data (existing)
Claude Haiku 5.5's 360 writes (50 violations, 27 tasks): G33's dithered answers (D16) and G18's 4 answers (G18_R4).

## Endpoints (within tool; cluster bootstrap over tasks, 2000, seed 4100)
- **R1 (primary).** AUROC(relevant, D16) − AUROC(full, D16): lower 95% bound > 0.
- **R2.** The same on G18's 4 answers: lower 95% bound > 0.
- **Reported.** Every read-out's AUROC with its interval.

## Readings, fixed in advance
- R1 passes: reading the state along the directions the call can violate improves the closed agent's monitor; the
  fixed equal-weight read-out was diluted by irrelevant directions.
- R1 fails: relevance gating does not help at this power.

## Known before locking (disclosed)
Pooled per-question within-tool AUROCs at the call (G18: 0.555, 0.632, 0.571, 0.535, 0.613, 0.516, 0.503, 0.588) and
with dither (D16: 0.547, 0.659, 0.638, 0.637, 0.689, 0.566, 0.552, 0.556), and before the call (G37). No read-out over a
per-tool subset of questions has been computed. Because these data were seen in aggregate, a pass here is a
confirmatory test on seen data; a fresh-data replication needs new answers from the closed model.
