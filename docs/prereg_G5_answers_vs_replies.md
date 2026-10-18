# Preregistration G5: which outputs carry the state? Yes/no answers vs the model's own replies

Locked, and published, before the confirmatory data exist. Code: scripts/133_g5_answers_vs_replies.py (it reuses the
generation code of scripts/132 and the geometry code of scripts/129). No model is trained or fine-tuned.

## Why
G1 showed that a model's yes/no answer geometry matches its own hidden geometry far better than the geometry of the
input text. An API also gives a second output: the model's replies. Model-equality tests and output-based audits use
replies. G5 asks which output geometry matches the hidden geometry, at stated costs.

Known before locking:
- every earlier result, including G1–G3 (all pass);
- **G4, a dropped design (dev only, scripts/132, results/t1/g4_shift_DRYRUN_dev.json).** It asked whether answers
  detect a hidden system-prompt change better than replies, at equal calls. On 480 dev contexts the answer was no.
  - 8 one-sample answers had less power than 8 sampled replies for every model and condition. For example, DAN at
    N = 20 contexts: answers 0.00–0.24, replies 0.17–0.87.
  - Exact log-probabilities detected every change, including the near-null "helpful" prompt, with power ≥ 0.94 at
    N = 20. That is expected of any deterministic output and is not new.
  - The shift test with 8 binary answers was conservative (empirical level 0.00–0.05).
  - G4 is not preregistered and will be reported as a negative development result: for detecting THAT a state moved,
    sampled replies are the more sensitive output.
- the dry run of G5 on the same 480 dev contexts (results/t1/g5_answers_vs_replies_DRYRUN_dev.json):
  - ρ(answers exact, hidden) 0.38–0.75;
  - ρ(8 replies, hidden) 0.23–0.46, close to ρ(context text, hidden) 0.23–0.30;
  - H1 8/8, H2 7/8 (gpt-oss −0.05), H3 8/8;
  - 143 of these 480 dev contexts were also in G1's layer-selection set; the confirm contexts are not;
- on dev, the 8-call design (8 questions × 1 sample vs 8 replies) LOSES to replies (8 questions are too few; G1 F3).
  The equal-cost comparison below is therefore at equal OUTPUT tokens (88 answers of 1 token vs 1 reply of 128),
  not at equal calls; the answers use about 88 times more INPUT tokens (the context is a shared, cacheable prefix);
- G4's "none" condition reproduces L2's answers and states for 7 models, but not for gpt-oss: its chat template
  inserts the current date. G5 therefore generates answers, replies and hidden states in one fresh pass.

## Design
- **Models:** the 8 open chat models of L2 and G1.
- **Contexts:** all 1,440 L2 CONFIRM contexts, no system message (L2's rendering).
- **One fresh pass per model** (scripts/132's code; vLLM seeds 0 for answers, 1 for replies):
  - answers: top-20 log-probabilities of the first answer token after each of the 88 battery-v2 questions;
  - replies: 8 replies at temperature 1, at most 128 tokens, embedded with all-MiniLM-L6-v2 (masked mean, L2 norm);
  - hidden: the residual stream at the last prompt token (HF, bf16), at G1's dev-chosen layer.
- **Geometries** (all fit-free):
  - answers exact: Fisher–Rao distances of P(yes | yes or no) over 88 questions (logprob tier);
  - answers sampled: 1 answer per question, drawn as yes / no / other from the raw top-20 probabilities (numpy seed
    1330), Anscombe angles with n = 1 (or n = 0 for other); 88 calls of 1 output token;
  - replies: 1 − cosine of the mean embedding of the 8 replies (8 calls), or of the first reply (1 call; 128 tokens);
  - hidden: 1 − cosine.
- **Statistic:** Spearman over all pairs of contexts; bootstrap over contexts (500; seeds 1331 + i, 1341 + i, 1351 + i).

## Endpoints
- **H1 (logprob tier):** ρ(answers exact) − ρ(8 replies). Pass: lower 95% bound > 0 for at least 7 of 8 models.
- **H2 (sample tier, equal output tokens):** ρ(88 one-sample answers) − ρ(1 reply). Pass: lower 95% bound > 0 for at
  least 6 of 8 models.
- **H3 (replies saturate):** ρ(8 replies) − ρ(1 reply). Pass: upper 95% bound < 0.05 for at least 7 of 8 models.
- **Reported:** ρ of each geometry; the context-text geometry; input and output tokens per context for each design.

## Readings, fixed in advance
- **H1–H3 pass:** a model's replies carry its state's geometry only about as well as the input text does, and more
  replies do not help. Its yes/no answers carry it, and more questions help (G1 F3). The answer geometry is the right
  observable for WHERE the state is. G4 says replies are the more sensitive observable for WHETHER it moved; the
  paper reports both.
- **H1 passes, H2 fails:** the advantage needs exact log-probabilities, or more than one sample per question.
- **H1 fails:** the claim that answers beat replies is withdrawn.

## Limits stated in advance
- One reply embedder (MiniLM, the same as the text baseline). A stronger embedder could raise the reply geometry.
- Replies of at most 128 tokens at temperature 1, no system message.
- The hidden ground truth is one layer, at the last prompt token: the state from which the reply begins.
