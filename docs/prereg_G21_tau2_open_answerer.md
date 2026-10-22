# Preregistration G21: can an open model answer the questionnaire about a closed agent's calls on the second benchmark?

Locked, and published, before any Gemma-4 answer about these writes exists. Code: scripts/188_g21_tau2_open_answerer.py.

## Why
G16 found no detectable difference between an agent's own answers and another model's on CAPE-Bench. If that holds on
the second benchmark, a closed agent can be monitored with no API call to it: an open model answers the questionnaire
about the closed agent's calls from the transcript. G21 tests this on tau2-bench, where the questions and the agent
(Claude Haiku 5.5) differ from CAPE-Bench.

## Design (fixed; nothing is fit, no label is used, no API is called)
- Units: G18's 360 writes (50 violations), G18's labels.
- Answerer: Gemma-4 (26B-A4B), thinking on, its standard sampling, served by vLLM. It reads Claude's conversation up
  to the write as Claude saw it (tau2's agent system prompt with G18's SYSTEM_NOTE, the airline tools, every earlier
  turn), then G18's frame with each of G18's 8 questions; 4 answers each; G18's parser and read-out.
- Statistic: within-tool AUROC; cluster bootstrap over task_id (2000).

## Endpoints
- **A1 (primary).** Gemma-4's read-out: lower 95% bound > 0.50.
- **A2 (reported, two-sided).** Claude's own read-out (G18) − Gemma-4's, 95% interval.
- Validity: if fewer than half of Gemma-4's answers parse, the run is invalid; the cause is fixed, recorded as a
  deviation, and the run repeated before any analysis.

## Readings, fixed in advance
- A1 passes: an open model, with no training and no call to the closed agent, ranks the closed agent's violations above
  chance on a second benchmark.
- A1 fails: on tau2-bench the open answerer does not rank them above chance at this power.
- A2 says whether the closed agent's own answers add anything here (interval above 0), or not.

## Known before locking
G16, G18, G19 results; G18's own answers. No Gemma-4 answer about these writes exists.
