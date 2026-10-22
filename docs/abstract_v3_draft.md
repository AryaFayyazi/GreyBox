Draft abstract for the unified grey-box paper (insert after G14; G14 numbers via macros from paper/g14_numbers.tex).

Probes on hidden states can stop an agent's unsafe action before it runs, but they need the weights, and the most
capable agents are closed. We make a closed agent grey: we recreate its hidden decision state from what its API
returns and read the recreation with probes trained on the true states of open models. Hidden coordinates cannot be
recovered: a gauge symmetry leaves every output unchanged. Geometry can. Yes/no answers asked in side branches give
answer laws whose Fisher–Rao distances are Euclidean in arcsine coordinates, with an exactly unbiased cross-fit
estimator from sampled text. On eight open models this observable geometry tracks hidden geometry beyond the kind of
context (partial Spearman 0.46; input text 0.09), and it is shared: a model's answers match other models' hidden
geometries as well as its own, Claude's included, so an open model can stand in. A probe on the recreated state is the
conditional expectation of the same probe on the true state, and can beat it. On four open agents whose decision states
we know, the recreation from an open reader and the agent's own answers [G14 sentence: e.g. "monitors violations as
well as the weights (within-template AUROC \GFtGrey vs \GFtWhite), also in another agent's coordinates"]; on the closed
Claude Haiku 5.5 it ranks unapproved restarts at \GFtClaude (preregistered). Policing closed agents with it: a reader
gate cut Claude's and GPT's episodes with an executed violation from about 7% to 0.2% live (at 1.8 and 5.8 points of
task success); the agent's own audit, one question of the grey box, prevented 69% of the violating episodes its
transcript decides; a law predicted that effect for five agents within 0.15 (mean error 0.035), and on a second
benchmark the audit cut successful prompt injections by 84% and 50% (all preregistered). We release the tool.

Title candidates (decide after G14, keep "Grey Box"):
- Grey Box: Recreating a Closed Agent's Hidden State to Police It
- Grey Box: Reading Closed Agents' Hidden States Through Their APIs
- (current) Grey Box: What Crosses the API, and How to Police Closed Agents
