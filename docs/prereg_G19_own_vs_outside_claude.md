# Preregistration G19: does a closed agent's answer need its own conversation?

Locked, and published, before any outside-reviewer answer exists. Code: scripts/185_tau2_claude_outside.py.

## Why
The method reads an agent through its own answers, asked in side branches of its own conversation. A reviewer asked
whether "own" matters: would the same model, reading the same conversation as an outside reviewer, answer as well?
G16 asks this for open agents with another model as the answerer. G19 asks it for the closed agent with the SAME model,
so that only the position changes: inside its own conversation (as the agent) or outside it (as a reviewer).

## Design (fixed; nothing is fit, no label is used)
- Units: G18's confirmation writes (Claude Haiku 5.5, tau2-bench airline): every violation (50) and 130 allowed writes
  drawn uniformly (seed 1900), 180 in all.
- Own: G18's stored answers (the agent in its own conversation, with G18's system note).
- Outside: the same model, settings, 8 questions, 4 answers each and read-out; the agent's operating instructions and
  the conversation are given as a transcript (graybox.chat.chat_ctx, the format of T18-T23's monitors) in one user
  message, with a system prompt that names the reader as a reviewer for the operations team. No tools are given.
- Statistic: within-tool AUROC (pair-weighted), cluster bootstrap over task_id (2000).

## Endpoint
- **O1 (two-sided).** Own − outside, read-out of 8 answers, 95% interval.
- **Reported.** Own − outside for the single overall question; the share of parsed answers.

## Readings, fixed in advance
- O1 interval above 0: the answers carry more when asked inside the agent's own conversation; "own" matters.
- O1 interval below 0: an outside reader of the same transcript does better; the method's value is the questions,
  not the position, and the paper must say so.
- Interval includes 0: no detectable difference at this power; the paper must not claim that "own" matters.

## Budget
Anthropic ledger cap $44.70 (shared; the battery stops at the cap). Estimate about $0.9.
