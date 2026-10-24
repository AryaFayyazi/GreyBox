# Incident: an early look at G45's Claude labels (2026-10-23, about 13:50 PDT)

While G45's Claude run was in progress (53 of 328 episodes written), we wrote a draft live-gate test (G46, never locked,
never run; kept in docs/drafts_not_run/). A dry run of its code on the partial G45 file printed a number that uses
G45's labels: the share of violating episodes whose first violating call a pre-call gate at the 80th-percentile cut
would hold (0.0). We then looked on purpose, exploratory, at the per-turn AUROC of the pre-call read-out on those 53
episodes (turn 2: 0.44, turn 3: 0.35; almost all violations at turn 2; read-out SD 0.20 at turn 2) and at seed-to-seed
agreement of the violation (7 of 12 scenarios agree).

What this does and does not change:
- G45's analysis code and endpoints were locked and published before any episode (results/prereg_G45_lock.json). The
  look cannot change them, and we run them unchanged on the complete data.
- We dropped G46 as drafted (before any lock or episode), because on these development data its gate holds nothing
  that a position gate does not hold.
- From here, G45's Claude episodes serve as development data for any later test on Claude Haiku 5.5; that test runs on
  fresh seeds and is locked before its data exist.

## Second early look: partial G46 other-model data (2026-10-23, about 20:40 PDT)
While testing the figure script scripts/228_fig_says_does.py, it read the partly written Gemma-4 draw file
(outputs/g45/g46_other_gemma4.jsonl, 320 of 1,589 units) and printed within-position AUROCs on those units: self-report
0.432, own draws 0.696, Gemma-4 draws 0.509, ceiling 0.809. G46 and G46b were locked and published before (ed80043,
777dc6f); their analysis code is unchanged and runs on the complete data. G46b's Brier quantities were not computed.
The figure script now waits for the complete file.
