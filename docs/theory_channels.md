# What each output channel reveals of a model's state

Setting as in docs/theory_infogeom.md. A **channel** is what an API returns for a context x: a probability law
O_κ(x) on a set of observations. Examples, from coarse to fine:
- **answers:** yes/no answers to a battery, in side branches;
- **top-k:** the next-token law at the first content position, reduced to the k most likely tokens plus "other";
- **full:** the whole next-token law (full log-probabilities).

The **observable geometry** of a channel is d_κ(x, x') = Fisher–Rao distance between O_κ(x) and O_κ(x'), that is
2 arccos BC(O_κ(x), O_κ(x')) with BC the Bhattacharyya coefficient.

## Proposition C1 (monotonicity; a coarser view never shows more)
If channel κ' is a garbling of κ (O_κ'(x) = T O_κ(x) for one Markov kernel T, the same for all x), then
d_κ'(x, x') ≤ d_κ(x, x') for all x, x'.
Proof. For any kernel T, BC(Tp, Tq) ≥ BC(p, q). This is the data-processing inequality for the Bhattacharyya coefficient,
an f-divergence affinity, and equivalently Čencov's monotonicity of the Fisher metric. The map
BC ↦ 2 arccos BC is decreasing. ∎
- Example: top-k with a FIXED set of k tokens is a garbling of full. So is any fixed partition of the vocabulary.
- An API's top-k is not a fixed partition: the k tokens depend on the context. Its observable similarity is the
  Bhattacharyya coefficient computable from the returned values (shared top-k tokens, plus sqrt(other · other)). It is
  a statistic of what the API returns, not an exact Fisher–Rao distance.
- Answers come from different branches of the computation (the question tokens are appended), so they are not a
  garbling of the next-token law. Where they fall on the ladder is an empirical question.

## Proposition C2 (the full channel fixes the model's own final geometry)
With full log-probabilities, the Gram matrix of the final normalised states is recovered exactly from d(d+1)/2
contexts (Theorem 6 of docs/theory_infogeom.md; the construction of Carlini et al. 2024, App. H). So the full channel
carries the model-specific final geometry completely.

## Hypothesis (tested in G7)
- Decompose each model's hidden geometry into a part shared across models and a model-specific remainder.
- G6 showed that the answers carry the shared part only: the own-model excess was +0.002.
- Hypothesis: the model-specific remainder becomes visible as the channel becomes finer. The own-model excess should
  be near 0 for answers, positive for top-k, and larger for the full law.
- Consequences if it holds:
  - For monitoring: yes/no access reveals what models share, so open models can stand in for a closed one.
  - For security and auditing: top-k log-probabilities, as some providers return, leak a model's own internal
    geometry, and full log-probabilities give it exactly.
