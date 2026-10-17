# Abstract v1 (superseded 2026-10-17; kept for reference)

## Abstract

Safety monitors that read a model's internals need its weights; the most capable agents expose only text. We ask
what outputs reveal about a model's hidden state, and what monitoring closed agents actually requires.

**Identifiability.** Hidden coordinates are not identifiable from outputs (any rotation of the hidden space leaves
every output unchanged), but the behavioural state is. A model's answers to a fixed battery of side-branch questions
recover it on a ladder of models with known internals:
- a compiled program: 21/21 states;
- Othello-GPT: R² 0.967, preregistered;
- eight open chat models: answers beat text by +0.120 R², and a model's own answers beat another model's at equal
  budget, preregistered;
- readouts trained only on open models transfer to Claude through text sampling alone.

**Resampling versus probes.** For safety labels defined by behaviour, we prove that resampling the model and scoring
each sample with the label function dominates any readout of its hidden state, at every moment. Five preregistered
tests on 12,128 fresh steps of a 26B tool-using agent show what this does and does not buy:
1. With the environment's exact rule checker as the label function, resampling beats a white-box probe at the turn
   start (0.904 vs 0.841). After the agent's reasoning, both approach ceiling (0.9997 vs 0.979). A white-box
   probe's well-known advantage is a timing advantage.
2. With a deployable LLM judge as the label function, the order reverses: judge-scored resampling falls to 0.782,
   below the probe. The bottleneck is the label function, not the access.
3. Amortizing the label function closes the gap without any access. An open reader model with a linear probe trained
   on labelled traces reads only the transcript and the proposed action:
   - it beats the agent's own internals (0.994 vs 0.979);
   - on scenario families unseen in training it stays within 0.007 of them (0.969 vs 0.976);
   - every trained monitor beats a zero-shot 70B judge by 0.05–0.08.
   - trained on one agent and applied to another family, it still beats the zero-shot judge (0.915 vs 0.833);
   - 74 labelled examples already beat the judge, and the reader runs in 41 ms per action;
   - on an external prompt-injection benchmark it beats both the zero-shot judge (0.78 vs 0.54) and a white-box
     gate (0.82 vs 0.60);
   - a bag-of-words classifier trained on the same labels matches it in distribution, but not on unseen task
     families (0.950 vs 0.969) or on a new benchmark (0.50 vs 0.78): the reader is what makes labels transfer;
   - on a closed frontier agent (Claude Haiku 5.5 acting through its API), trained only on an open agent's traces,
     it catches 96% of violations at 5% escalation (pooled AUROC 0.989; zero-shot judge 0.570). The cut needs no labels from
     Claude: a gate frozen on its own unlabelled traffic catches 95% online, and holds through bursts of violations;
   - its limit is the policy, not the agent. For a policy family removed from training it still outranks the judge
     (0.912 vs 0.570), but it escalates none of that family's violations until about 20 labelled examples are added
     (85% with 20, 94% with 40).

**Chat.** On Claude Haiku 5.5, the refusal forecast ties a strong graded moderator (prospective, 923 fresh
conversations). Combined with it, the forecast doubles the precision of a compliance alarm.

We release the preregistration record with every failed test, an SDK, a monitoring API, and an in-browser viewer that
places any model, open or closed, in a shared space.

