# Preregistration G14: can a black-box agent's decision state be recreated through its API well enough to monitor it with a white-box probe?

Locked, and published, before any G14 held-out battery or the Claude battery exists (except the 5-action Claude pilot
below). Code: scripts/171_decision_greybox.py (batteries, analysis stage=g14), scripts/172_claude_decision_greybox.py
(Claude), web/build_observable_ref.py (the 8 questions and their frame, unchanged).

## The method under test (the grey box)
A closed agent's hidden coordinates are not identifiable from its outputs (gauge symmetry), but the part of its state
that the outputs determine can be estimated. At each proposed call:
- an open reader (Qwen3-8B) reads the same transcript and call; a linear map, fit on an open agent whose states are
  known, recreates the agent's decision state (32 principal components) from the reader's state;
- the agent itself answers 8 yes/no questions about the call in side branches (4 answers each, arcsine angles); a
  second linear map corrects the reader's recreation with these answers (fit on the open agent's development battery).
- A white-box probe trained only on TRUE decision states of the open agent is applied to the recreated state.

## Known before locking (development data only; scenario-disjoint 5-fold cross-fit, scripts/171 stage=analyze mode=dev)
Within-template AUROC of a probe trained on true decision states, on 391–600 development calls per agent:

| agent | white box (true state) | grey box (reader + answers) | grey − white [95%] | reader only | answers only | ensemble of the OTHER agents' grey boxes |
|---|---|---|---|---|---|---|
| Gemma-4 | 0.956 | 0.977 | +0.021 [+0.003, +0.045] | 0.978 | 0.995 | 0.961 |
| gpt-oss-20b | 0.798 | 0.859 | +0.061 [−0.009, +0.134] | 0.856 | 0.699 | 0.724 |
| Qwen3.8 | 0.854 | 0.864 | +0.010 [−0.081, +0.122] | 0.831 | 0.862 | 0.819 |
| Muse-Glimmer | 0.856 | 0.902 | +0.047 [+0.003, +0.109] | 0.898 | 0.862 | 0.938 |

- A single source's coordinates can fail (Gemma-4's grey box on Qwen3.8: 0.58); the ensemble of the other agents does
  not (mean difference to the own white box about −0.005). Hence P2 and the Claude score use the ensemble.
- Hostile controls (development): answers paired with other calls' states: 0.46 (Gemma-4), 0.67, 0.38; a probe fit
  WITH labels on the answers: 0.980 (Gemma-4), below the label-free recreation 0.995.
- Recreation R² of the top 32 components: reader 0.30–0.66, answers 0.05–0.15.
- Code dry run on the development batteries: results/t1/g14_DRYRUN_dev.json (in-sample white box; not results).
- Claude: reader features of all 3,128 T8b actions were computed (inputs only; never scored against labels). A 5-action
  battery pilot (answers seen) is excluded from the analysis (scripts/171 PILOT).
- No held-out battery exists for any agent.

## Design
- **Open agents (truth known):** Gemma-4 (its fresh units: unseen task families), gpt-oss-20b, Qwen3.8-27B and
  Muse-Glimmer-30B (T17's held-out scenarios). Battery on up to 300 violating and 300 allowed held-out calls per agent
  (seed 1710), each agent through its own vLLM server and parsers. Maps, probes and PCA fit on development data only.
  Layer: the decision state's 4th stored layer (index 3), fixed.
- **Closed agent:** Claude Haiku 5.5 on its T8b actions (500: all restart actions, all violations, other allowed actions;
  seed 1720), 8 questions x 4 answers through its API (thinking disabled, effort low); ledger cap fixed at the lock.
  Its state is recreated in each open agent's coordinates and read by that agent's white-box probe; the four scores, each standardized on Claude's own sample (no labels), are averaged. Ledger cap: $39.10 (ledger $34.07 at the lock).
- **Bootstrap:** scenarios resampled within each agent, 2,000 resamples, seed 1750.

## Endpoints
- **P1 (the grey box loses little against the white box, own agent).** Mean over the four open agents of [within-template
  AUROC of the white-box probe on the recreated state (reader + answers) − on the true state]: lower 95% bound > −0.03.
- **P2 (other agents' coordinates suffice).** Mean over the four open agents of [the average of the OTHER three agents'
  grey boxes (standardized scores) − the agent's own white box]: lower 95% bound > −0.07.
- **P3 (closed agent).** Claude, the four agents' grey boxes averaged: AUROC on the restart template (where its violations
  are decided by its transcript; 25 violations after the pilot exclusion) at least 0.80, with a lower 95% bound at least 0.65.
- **Reported, not endpoints:** whether the answers add to the reader (P3_answers_add_to_reader_mean), per-agent within and
  pooled AUROC of every score, Claude's pooled AUROC and scores on mailing-list sends.
- **Run calls:** slurm/g14.sh (GPU batteries, one job); scripts/172 stage=battery cap=39.10 (Claude).
- **Analysis call:** scripts/171 stage=g14 (writes results/t1/g14_decision_grey_box.json).

## Readings, fixed in advance
- **P1–P3 pass:** a black-box agent's decision state can be recreated through its API (an open reader plus its own
  answers) well enough that a probe trained on true hidden states monitors it about as well as with the weights.
- **P1 passes, P2 fails:** recreation works within an agent family's own coordinates but not across agents.
- **P1 fails:** the recreation loses the white-box probe's power on held-out data.

## Limits stated in advance
- One benchmark (CAPE-Bench); 32 principal components of one layer; the "truth" for the closed agent is its label,
  not its state.

## Deviation 1 (2026-10-21 13:22, before the Claude battery finished; no G14 data examined)
The Claude ledger cap rises from $39.10 to $39.40. Reason: after 40 actions the cost per action was $0.0103, not the
pilot's $0.0076 (later conversations are longer), so 495 actions need about $5.07 and would stop near $39.15. The cap
is a spending guard, not an analysis parameter; the sample, prompts, settings and analysis are unchanged. If the run
stops at $39.10 it resumes from its checkpoint with cap=39.40.
