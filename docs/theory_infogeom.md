# The observable latent geometry of a closed model (training-free)

Goal: estimate the latent space of a closed language model rigorously, from its outputs alone, with no trained
surrogate. Answer: hidden COORDINATES are not identifiable (gauge), but the latent GEOMETRY is — the Fisher–Rao
geometry of the model's answer distributions — and it is measured exactly, in flat coordinates, from yes/no answers.

## Setting
A model reads a context x and reaches an internal state s(x) (all keys and values; a deterministic function of the
tokens). A battery Q = (q_1, ..., q_m) of yes/no questions is asked in side branches: each question is appended to x
separately, so the answers are independent given x. The answer to q is Bernoulli with
    Ψ_q(x) = P(yes | x ⊕ q) = f_q(s(x)),
where f_q is the network run on the question tokens from the cached state. The model of the answers is the product
family P_x = ⊗_q Bernoulli(Ψ_q(x)).

## Definition (observable geometry)
d_Q(x, x') = the Fisher–Rao geodesic distance between P_x and P_{x'} within the family of independent Bernoulli answers.

## Theorem 1 (exact flat coordinates)
With ϑ_q = 2 arcsin √Ψ_q,  d_Q(x, x')² = Σ_q (ϑ_q(x) − ϑ_q(x'))².
Proof. The Fisher information of Bernoulli(p) is 1/(p(1−p)); dϑ/dp = 1/√(p(1−p)), so the Fisher metric is dϑ²: flat.
The Fisher metric of a product family is the sum of the factors' metrics, so in ϑ the family is the box [0, π]^m with
the Euclidean metric. The box is convex, so geodesics are straight segments and the geodesic distance is Euclidean.
(Check for one factor: the Bhattacharyya coefficient of Bernoulli(sin² a), Bernoulli(sin² b) is cos(a − b), and
2 arccos cos(a − b) = 2|a − b| = |ϑ − ϑ'|.) ∎
So ϑ(Ψ_Q) is an ISOMETRIC embedding of the observable geometry into Euclidean space: no approximation, no fitting.

## Theorem 2 (invariance and uniqueness)
(a) d_Q depends only on the model's output distributions, so it is invariant under every reparametrization of the
internals that leaves outputs unchanged (the gauge group of Proposition 1 in the paper).
(b) By Čencov's theorem (1982; Campbell 1986 for finite outcome spaces), the Fisher metric is, up to a constant factor,
the only Riemannian metric on families of probability distributions that is invariant under Markov morphisms
(congruent embeddings / sufficient statistics). Hence d_Q is the canonical geometry of a model's state that can be
defined from its outputs: no other output-based Riemannian geometry is invariant in this sense.

## Theorem 3 (what d_Q identifies)
d_Q is a pseudo-metric on contexts; d_Q(x, x') = 0 iff the model answers every question in Q identically after x and x'.
For a model whose answers come from a finite automaton, d_Q is a metric on its minimal (Myhill–Nerode) states iff Q holds
a distinguishing suffix for every pair of states (Proposition "battery design" in the paper). In general, as Q grows,
d_Q converges (monotonically, for nested batteries) to a metric on behavioural classes: contexts after which the model
answers every question alike. This is the gauge-invariant content of the hidden state.

## Theorem 4 (local relation to the hidden geometry)
Let Ψ_q = σ(a_q(s)) with a_q smooth. For nearby states s, s' = s + Δ,
    d_Q² = Δᵀ G_Q(s) Δ + O(|Δ|³),   G_Q(s) = Σ_q Ψ_q(1 − Ψ_q) ∇a_q ∇a_qᵀ,
the Fisher information of the answers pulled back to the state: the observable distance is the hidden distance in the
pulled-back Fisher metric. Proof: dϑ_q = Ψ'_q / √(Ψ_q(1−Ψ_q)) and Ψ'_q = Ψ_q(1−Ψ_q) ∇a_qᵀΔ. ∎
Consequences that the experiments test:
- Euclidean (or cosine) hidden geometry and d_Q agree when G_Q is close to isotropic on the directions along which
  contexts vary; G_Q gains rank and conditioning as questions are added, so agreement should rise with |Q| and saturate.
- Agreement should be highest at the layers from which the answers are read most linearly (middle-to-late layers),
  not at the earliest layers, which still encode surface form.

## Theorem 5 (estimation from samples: any closed API)
With n sampled answers per question, the Anscombe-corrected angle ϑ̂ = 2 arcsin √((k + 3/8)/(n + 3/4)) has variance
1/n + O(n⁻²) for every Ψ bounded away from 0 and 1, independently across questions and contexts. Hence
    E[‖ϑ̂(x) − ϑ̂(x')‖²] = d_Q(x, x')² + 2m/n + O(m n⁻²),
so d̂² = ‖ϑ̂(x) − ϑ̂(x')‖² − 2m/n is unbiased to first order, and its standard deviation is O(√m / n + d_Q / √n).
Because the coordinates are Euclidean, classical multidimensional scaling recovers ϑ(Ψ_Q) up to a rigid motion (exact
for exact distances; stable under perturbation: Arias-Castro et al., 2020). The cost per context is m·n API calls.

## Theorem 6 (exact recovery of the final hidden geometry from log-probabilities)
For a pre-norm model whose output is logits z = W u with u = γ ⊙ h / rms(h) and no output bias: fold γ into W, so the
normalized state u lies on the sphere ‖u‖² = d (up to the normalization's ε). Centered log-probabilities are
c(x) = P W u(x) with P = I − 11ᵀ/V, a rank-d linear image of u. Write the observed matrix C = Q_d B (rank-d SVD); then
U = K B for an unknown invertible K, and the sphere gives ‖K b_i‖² = d, i.e. b_iᵀ S b_i = d with S = KᵀK, linear in S.
If the matrices b_i b_iᵀ span the symmetric matrices (generically, n ≥ d(d+1)/2 contexts), S is unique, and the Gram
matrix of the hidden states, UᵀU = Bᵀ S B, is recovered exactly: every distance and angle between the final normalized
hidden states, from outputs alone; the states themselves up to an orthogonal transformation (the gauge group).
Limit, stated plainly: it needs full log-probabilities and d(d+1)/2 contexts (5×10⁵ for d = 1,024; 8×10⁶ for 4,096).

## Why this is the right object
- It is what is identifiable (Theorems 2–3) and the only canonical choice (Čencov).
- It is measured exactly in flat coordinates (Theorem 1) and estimated from text-only APIs with known error
  (Theorem 5).
- It connects to the hidden state through the pulled-back Fisher metric (Theorem 4), which is testable where the
  hidden state is known.

## Correction 1 (2026-10-18, after the lock of G; the text above is the locked version)
Theorem 5 above states its error terms wrongly. A numerical check after the lock (exact binomial sums) shows:
- The Anscombe angle is biased at order 1/n, not n⁻²: for fixed p in (0, 1),
  E ϑ̂ = ϑ + cot(ϑ)/(4n) + O(n⁻²) (n = 32: bias 0.0102 vs 0.0104 at p = 0.1; 0.0033 vs 0.0034 at p = 0.3).
  So E‖ϑ̂(x) − ϑ̂(x')‖² − 2m/n differs from d_Q² by a term of order d_Q²/n, not only O(m n⁻²).
- The variance 1/(n + ½) holds only where n p(1 − p) is large. Near p = 0 or 1 it is smaller (0.68/(n + ½) at
  p = 0.1, n = 8), so subtracting 2m/(n + ½) over-corrects for saturated answers. Saturated answers are common for
  language models.
- With n samples the estimated angles cannot leave [2 arcsin √(3/8/(n + 3/4)), π − the same], so large distances
  between saturated answers are contracted (in a test with saturated P: true d² 48.4, debiased estimate 30.1). This
  is a limit of what n samples identify, not of the formula.

### Theorem 5′ (exact, replaces the bias claim of Theorem 5)
Fix n and let μ_n(p) = E ϑ̂ for k ~ Binomial(n, p). μ_n is a known, strictly increasing function of p (ϑ̂ is
increasing in k, and k is stochastically increasing in p), and μ_n → ϑ pointwise as n → ∞. Split each cell's 2n
samples into two independent halves A and B of n. Then the cross-fit estimator
    D̂²(x, x') = Σ_q (ϑ̂ᴬ_q(x) − ϑ̂ᴬ_q(x'))(ϑ̂ᴮ_q(x) − ϑ̂ᴮ_q(x'))
satisfies E D̂² = Σ_q (μ_n(Ψ_q(x)) − μ_n(Ψ_q(x')))² EXACTLY, for every Ψ in [0, 1]: no variance model, no
asymptotics. Proof: the halves are independent, so E[(a_A − a'_A)(a_B − a'_B)] = (E a_A − E a'_A)(E a_B − E a'_B),
and both factors equal μ_n(Ψ_q(x)) − μ_n(Ψ_q(x')). ∎ (Monte Carlo, 20,000 draws, saturated P: mean 25.76 vs target
25.79.)
So with n samples per half, the observable geometry is the Euclidean geometry of μ_n(Ψ_Q), and it is estimated without
bias. As n grows it tends to the Fisher–Rao geometry of Theorem 1.

Effect on the locked analyses: none on their definitions. F2 measures the real loss of the locked estimator (as a rank
correlation), and C1/C2 use the locked estimator as specified. Analyses with the cross-fit estimator are exploratory and
are labelled so.
