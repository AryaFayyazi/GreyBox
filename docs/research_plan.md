# Grey-box safety: research plan (draft 2026-10-15)

Status: DRAFT. The framing is provisional until the prior-art check (docs/prior_art_2026-10-15.md) is read.
Nothing in this file is a result.

## 1. The problem

CAPE ("The Agent Knew", arXiv 2610.17301) shows that a linear readout of an open agent's residual stream forecasts
unauthorized actions and the lies that follow, and that gating on it prevents violations. That needs the weights.
Frontier agents (Claude, GPT, Gemini) expose only text, sometimes top-k logprobs, and they hide or summarize their
chain of thought. CAPE's own attempt to transfer to frontier τ² trajectories, an open model *reading* the frontier
transcript (the reader gate), did not pass (AUROC 0.587).

**Question.** What part of the white-box safety signal can be recovered from black-box access alone, at what query
cost, and can it forecast and prevent violations of agents whose weights we do not have?

## 2. The object: the interrogation state (the "grey box")

The literature report (Latent recovery black box LLMs.md) shows that raw hidden coordinates are not identifiable
(GL(d) gauge; intermediate layers leave no output footprint). We therefore estimate a different object, which is
identifiable by construction.

For a black box B, a context x and a fixed battery Q = {q_1..q_m} of short suffix questions with finite answer sets:

    Ψ_B(x) = ( P_B(a_j | x ⊕ q_j) )_{j=1..m}    ∈ [0,1]^m

- Estimable from any API: one answer-token logprob per question (logprob tier), or n sampled answers
  (text-only tier, coordinate error ≤ 1/(2√n)).
- Coordinates are **gauge-free and shared across models**: coordinate j means the same question for every model.
  A forecaster fitted in Ψ-space on open models applies to any black box without an alignment step.
- Geometry: the Fisher-Rao map θ_j = 2·arcsin√p_j makes the product-Bernoulli Fisher metric Euclidean and
  stabilizes the variance (Var θ̂_j ≈ 1/n). Visualization and uncertainty use θ.

## 3. Theory (to be written as propositions with proofs)

**T1 Behavioural sufficiency ("probes cannot beat rollouts").** Let Y = g(τ) be any label on the future trajectory
τ of the agent-environment system started at context x. Let h(x) be any internal representation (any function of x
and the weights). Then π*(x) = P(Y=1 | x) has pointwise-maximal ROC and minimal risk under every proper scoring rule
among all forecasters f(h(x)). π* is estimable by K black-box rollouts with error ≤ sqrt(log(2/δ)/2K).
Consequence: for behaviour-defined safety labels, a white-box readout has no information advantage over a black box.
Its advantages are cost, latency, and not needing an environment simulator. A white box can also have an advantage for
labels that the model's behaviour does not define (latent-knowledge labels).
This organizes the white-box vs black-box comparison into a measurable cost-accuracy frontier.

**T2 Interrogation sufficiency (finite linear dimension).** If B's future-event probabilities have finite linear
(Hankel) rank r, then r core tests determine all future-event probabilities linearly (classical PSR result, to be
cited after verification). Then π* is a linear function of Ψ_{Q*}. For real models the rank is measured empirically
from the Ψ spectrum.

**T3 Gauge.** Hidden states are identified only up to GL(d) even with full logprobs (Carlini Lemma 4.2). Ψ is
identified exactly. Hence cross-model transfer is possible in Ψ-space and not in hidden-state space without paired
data.

**T4 Estimation.** Finite-sample bounds for Ψ̂ (Hoeffding per coordinate; the arcsin map gives isotropic noise) and
their propagation to a linear forecaster's AUROC.

## 4. Validation design: open models as black boxes

Ground truth exists only for open models. Wrap each CAPE actor behind a simulated API that exposes chosen observables
(answer-token logprobs, samples only, top-k). Run the estimators blind. Compare with the true hidden states and the
frozen CAPE readouts.

- **Actors.** CAPE's four families: Gemma-4-26B-A4B, gpt-oss-20b, Muse-Glimmer-30B, Qwen3.8-27B.
- **Unit: turn start.** The context ends with the last tool result, before the agent's turn begins. This is the one
  point every black box exposes, because frontier APIs hide the chain of thought. Label: the action of this turn is
  the first unauthorized action (VIOL). Negatives: other turns of informed non-violators (X−OTHER), and the same
  critical turn in the authorized twin (X+CRIT).
  - Secondary unit: the decision point (after reasoning), only for models whose reasoning is visible.
- **Forecasters at the same unit.**
  - R (rollout ceiling, T1): K sampled turns, scored by CAPE's label checker.
  - I (white box): CAPE-style linear readout of the hidden state at turn start. Fit on development data and frozen.
  - Ψ (grey box): logistic regression on θ(Ψ̂).
  - Baselines: T_sup (TF-IDF text), M (zero-shot LLM monitor), O (observer: an open model's activations on the
    transcript, the reader-gate idea), E (a sentence-embedding of the transcript, input-only).
- **Recovery.** Cross-validated R² of the hidden state from θ(Ψ̂): its top principal components, the white-box
  readout score I, and CCA spectra. Compare with null calibrations: permutation nulls, an input-only text embedding,
  and an untrained battery (random-answer questions).
- **Transfer.** Leave-one-family-out: fit Ψ-forecaster on three families, freeze it, test on the fourth as an unseen
  black box.
- **Prevention.** Closed-loop gate on the Ψ forecast with the actor as a black box (block or escalate; no steering),
  over whole episodes against the authorized twin, as in CAPE.
- **Cost.** Queries, tokens and latency per decision for R, Ψ, M and I.

## 4b. Positioning after the prior-art check (docs/prior_art_2026-10-15.md)

- Suffix-question batteries are established: Pacchiardi 2309.15840 and QueRE 2501.01558.
- UK AISI (Cooney, Africa & Irving, 2606.12618) found that the battery matches activation probes on prompted lying
  and drops to chance on belief-verified model organisms.
- Open reader models probing closed transcripts already work (Model Hermeneutics, LessWrong, 2026-09-30).
- We therefore do not claim the battery or the reader as new. Our contributions:
  1. **Theory that predicts when black-box access suffices.** By T1, for behaviour-defined labels no white-box
     monitor can beat the rollout forecast. White-box access can win only on labels that behaviour does not define,
     such as belief-relative deception. This predicts the AISI null as a label-type effect, which we test directly
     (pillar 5 below).
  2. **The first matched-compute comparison of the three access levels on agents:** white-box probe, rollouts and the
     interrogation state, against the measured irreducible ceiling.
  3. **Validation of the grey box as a state estimate** against real residual streams, with **actor-vs-reader
     specificity**: does Ψ from the actor beat the same battery and probes run on a reader model, the Model
     Hermeneutics setting?
  4. **Gauge-free cross-family transfer and a black-box prevention gate.**

## 5. Candidate pillars (headline to be fixed after the pilot)

0. Label-type test: one behaviour-defined label (the violation) and one belief-relative label on the same agents.
   The prediction is black-box ≈ white-box on the first and a white-box advantage on the second.

1. Theory: T1 to T4.
2. Recovery: Ψ recovers the white-box safety readout (R² and AUROC non-inferiority to I), and beats text and observer.
3. Transfer: the forecaster transfers to a held-out family as a black box.
4. Prevention: a black-box gate prevents violations while keeping authorized work.
5. Frontier: the frozen forecaster applied to frontier agents on CAPE-Bench mock tools (needs API keys and budget,
   which are the user's decision).
6. Tool: an API plus a 3D viewer of the θ(Ψ) trajectory with a forecast axis and uncertainty ellipsoids.

## 6. Discipline (carried over from CAPE)

- Development and confirmation data split by scenario seed. Every confirmatory test gets a preregistration with a
  hash lock before its data exist.
- Within-(template, tool) AUROC with a scenario-cluster bootstrap. Holm correction within each family of endpoints.
- Every failure is recorded with cause and fix in docs/PREREGISTRATION_RECORD.md.
- One GPU at a time. Outputs on /data/models/graybox_ws.

## 7. Product: Grey Box Live (user request 2026-10-15; build after L2 confirms)

- A user chats with any model or agent through its API (bring your own key). After every turn the server forks the
  conversation and runs a small designed battery. Forks never alter the real reply, and providers cache the shared
  prefix.
- The view shows:
  - **3D path:** the turn-by-turn trajectory in the shared Ψ space, with several models in one space.
  - **Concept graph:** nodes for harm, refusal intent, deception, uncertainty, sycophancy and topic, lit by the
    estimated activation of concept axes. The axes are learned from open models' internals and shown only if they
    pass leave-one-family-out validation.
  - **Live safety panel:** calibrated forecasts with thresholds and uncertainty. In agent mode this includes the
    CAPE-style violation and misreport forecasts.
- Honesty rules: every node and flag shows its held-out accuracy. Nothing is presented as a real neuron. For open
  models the true activations are overlaid.
- Cost rule: live mode uses a designed battery of 16–24 items (greedy information design, as in L0). On text-only
  APIs it uses adaptive sampling.

## 8. Live v2: fix multi-turn carry-over (planned 2026-10-15 after the first real session)
- Failure: readouts were trained on single messages, so in multi-turn chats an alarm carries over: the "17 × 23"
  turn after a poison request was flagged at 0.86.
- Redesign:
  1. Battery v2m: the same items, worded about "the user's latest message" (locked before data).
  2. Multi-turn L2 contexts: 2–3-turn conversations from the dev pools, with switches harmful → benign,
     benign → harmful and random → random. Labels come from the latest message only.
  3. Ψ for 4+ open models.
  4. Retrain the readouts on single plus multi-turn data. Validate leave-one-family-out, with a separate false-alarm
     rate on harmful → benign switches.
  5. Republish the live session.
- GPU: about 4 models × 15 minutes, after the confirmations.
