# Paper outline (working draft; numbers marked ⟨…⟩ are pending or development-only)

**Working title.** *Grey Boxes: What Black-Box Access Can Recover of a Model's Hidden State, and When It Is Enough
for Safety*

## Claim structure (each claim lists its evidence and its status)

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Raw hidden coordinates are not identifiable from outputs, but the behavioural state is: exactly for finite-state models, up to battery depth (Myhill–Nerode) | T2b, T3; L0 (tracr): 21/21 states from 21 black-box-designed questions; rotation leaves outputs identical and breaks the white-box readout (r 0.9995 → −0.46) | done (L0, exact) |
| C2 | On a model with a known world model, black-box queries recover the hidden vector almost completely, and depth governs the rest | L1 Othello-GPT: hidden R² 0.96–0.98 from 61–317 queries; probe-subspace R² 0.65 → 0.72 with depth; board 76–80% vs 98% white-box | done (dev); confirmation on fresh games ⟨todo⟩ |
| C3 | On real LLMs, Ψ recovers model-specific state beyond the input text and beyond another model's reading of the same text | L2: ΔR² = R²(text+Ψ_self) − R²(text+Ψ_other), per layer and per neuron; 8 models, 7 families | running |
| C4 | Recovery scales with battery size, and black-box-designed batteries beat random ones; designs transfer across models | v3 512-item pool; QR design; random vs designed vs designed-on-other | queued |
| C5 | For behaviour-defined safety labels, no monitor beats the rollout forecast; on correctly selected units, white-box probes have no information advantage | T1, T1b; CAPE clean units (Gemma): R_K16 vs WB vs Ψ vs text, AUROC and proper scores | queued |
| C6 | Outcome-selected evaluation units make supervised monitors look better than the Bayes forecaster | T1b; CAPE v57 units: TXT 0.919 > R 0.904; clean units restore the ordering ⟨pending⟩ | partly done |
| C7 | Concept axes learned in Ψ space transfer to unseen model families (shared coordinates), including model-specific behaviour (refusal) | L2 leave-one-family-out; scripts/18 | queued |
| C8 | A live tool: any API model, its trajectory in a shared space, validated readouts | Grey Box API + Live viewer; adapter check vs offline Ψ | built; validation queued |

## Figures

1. The ladder: L0 / L1 / L2 / L3, what is known at each rung, and the headline number per rung.
2. L0: battery depth vs identified states; the gauge test (outputs identical, white-box readout broken).
3. L1: 3D trajectories, estimated vs true (from the viewer); recovery vs depth curve; board panels.
4. L2: R² by layer for self / other / text; the ΔR² bars per model; the visible-neuron map.
5. Scaling: R²(m) random vs designed vs designed-on-other.
6. Safety: the T1 ladder on clean units (R, WB, Ψ, text) with AUROC, Brier and log loss; the selection artefact on
   outcome-selected units.
7. Live: a conversation's path in the shared space with readouts.

## Positioning (docs/prior_art_2026-10-15.md)

- Question batteries: Pacchiardi 2309.15840 and QueRE 2501.01558. AISI's Cooney et al. 2606.12618 find the battery
  fails on belief-verified organisms. Our T1 explains why: those labels are not behaviour-defined.
- Reader models: Model Hermeneutics (LessWrong, 2026-09-30) is our Ψ_other / reader control.
- Logit-based recovery: Carlini 2403.06634 and Finlayson 2403.09539 (the logprob tier, final layer only).
- Neurons: Cammarata, "Neural networks are already clean" (Goodfire) motivates neuron-level visibility.
- We never claim to be first to ask questions. We claim the theory, the validation against ground truth across a
  ladder, the scaling and design results, and the safety-sufficiency test.

## Discipline

- Development results guide the design. Every headline number in the paper comes from a preregistered confirmation:
  the L2 confirmation split (30% of contexts, untouched), fresh Othello games, and fresh CAPE rollouts with unused
  seeds.
- Failures go to docs/PREREGISTRATION_RECORD.md with their cause and fix.
