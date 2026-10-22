# Changelog

The package is `greybox-monitor` on PyPI; you import `graybox`.

## 0.11.0 (2026-10-22)
**Added**
- `observable_state(..., operator_note=True)`: adds `OPERATOR_NOTE` to the system prompt, saying that a message marked
  [Operator check] comes from the operations team and that the customer does not see it. When the user turn belongs to
  a third party (a customer), Claude declined checks placed there, or treated them as an injection; with the note it
  answered (G18 smoke runs). `with_operator_note(messages)` does the same for your agent's own conversation.
- `parse=parse_first_line`: reads an answer on the first line ("No.", "[Operator check answer: no]") before any text the
  agent goes on to write to the customer.
- `TAU2_AIRLINE_QUESTIONS`: the 8 questions of the preregistered G18 test (tau2-bench airline). There the read-out ranked
  Claude Haiku 5.5's policy violations at within-tool AUROC 0.704 [0.536, 0.822] (single answer 0.555); within one task
  it was weaker (0.61, exploratory). Write your own questions from your policy before you look at answers.

**Changed**
- The 0.10.0 note said the read-out ranks violations "as well as" a probe on hidden states. Against a probe tuned
  afterwards it is 0.015 lower on average (exploratory); "nearly as well" is the accurate wording.
- An outside reviewer (the same model reading the transcript) did about as well as the agent's own answers (G19:
  +0.043 [-0.136, 0.226]); the value is in the questions and the read-out.

## 0.10.0 (2026-10-21)
**Added**
- `graybox.selfaudit.observable_state(ask, messages, name, args, n=4)`: the agent's observable state at a proposed call,
  training-free and API-only. The agent answers 8 yes/no questions about the call in side branches (`STATE_QUESTIONS`,
  the battery of the G15 test, verbatim); the answers' arcsine angles are exact Fisher-Rao coordinates; `score` is the
  fixed read-out (mean angle in each question's violation direction; no labels, nothing fit). In the preregistered G15
  test it ranked held-out violations as well as a probe trained on the agent's hidden states (within-template AUROC
  0.82-0.99 on four open agents; mean difference +0.045 [-0.017, +0.135]) and Claude Haiku 5.5's unapproved restarts at
  0.909. Limits: the questions were written for CAPE-Bench's kinds of violation; a violation decided by a fact the agent
  never checked is not flagged. Set the hold cut on your own unlabelled traffic.

## 0.9.2 (2026-10-21)
**Changed**
- `openai_asker` and `anthropic_asker` default to `max_tokens=2048`. In the G10 test the "checklist" audit had 700
  tokens and 9.3% of audits were cut off before the verdict (disclosed in the paper). The docstrings give the G10
  settings for Claude (thinking disabled, low effort).
**Docs**
- `flag_rates`: τ predicts the share prevented at the states you audited; over another mix of states, weight τ by
  where the violations occur (a preregistered test, G11b, failed when τ came from other states).

## 0.9.1 (2026-10-20)
**Fixed** (from a user test of 0.9.0)
- `graybox.selfaudit` fails closed: when no answer can be read, the call is held (`unparsable="hold"`, with a warning);
  `unparsable="run"` reproduces the tests. `ask` returning a string, or `n < 1`, raises.
- `n` is set once, on `SelfAudit`, and passed to the asker (`ask(messages, n)`); one-argument askers still work.
- `anthropic_asker` moves a "system" message to Anthropic's `system` field.
- `parse_answer` also reads "Answer: no" and "Final answer: no".
- `prevention_law` rejects `M < 1` and always returns floats; its docstring explains M > 1.
- Custom prompts (with `{name}` and `{args}`); arguments that are not JSON-serializable no longer crash.
**Added**
- `flag_rates(results, violates)`: τ and φ of the audit on your own labelled calls.
**Docs**
- The format of `messages`, what "flagged" means for each prompt, `reads`, cut and n, cost and latency.

## 0.9.0 (2026-10-20)
**Added**
- `graybox.selfaudit`: self-policing through the agent's own API. Before a proposed tool call runs, the agent itself
  is asked, in a side branch, whether the call complies with its policy; a flagged call is held for review. No other
  model, no labels. Standard library only.
  - `SelfAudit(ask, prompt="checklist" | "ask", n=3, cut=1.0).check(messages, tool, args)`: the two prompts used in
    the preregistered tests, verbatim, and the same answer parser. Read-only tools are not audited.
  - `openai_asker(client, model, tools)` and `anthropic_asker(client, model, system, tools)`: the agent's own answers.
  - `prevention_law(R, tau, phi, M)`: the expected share of violations that run and of steps that go to review.
  - Preregistered results: Gemma-4, violating episodes 27.9% → 5.0% (G9); Claude Haiku 5.5, violations that its
    transcript decides −69% [51, 88] at no cost in task success, and no effect on violations decided by a fact it never
    checked, as predicted (G10). It needs an agent that judges its policy well: measure that on labelled steps first.

## 0.8.1 (2026-10-19)
**Fixed**
- `graybox-geometry` crashed on a clean install: the API client (`httpx`) is not a core dependency. Install it with
  `pip install 'greybox-monitor[geometry]'`; without it, the command now says so. A test from a fresh install of 0.8.0
  found this.

## 0.8.0 (2026-10-18)
**Added**
- `graybox.geometry`: the observable latent geometry of any model, from its outputs alone. Nothing is trained.
  - `interrogate`, `fisher_angles`, `fisher_angles_from_counts`, `observable_distances`: Fisher-Rao distances
    between contexts from yes/no answers. They are exact in the angles 2 arcsin √P(yes). From sampled answers, the
    known sampling bias is removed.
  - `mds`, `rsa`: coordinates up to a rigid motion; comparison of two geometries with no fit.
  - `logprob_basis`, `recover_gram`: with full log-probabilities, the exact Gram matrix of the final normalized hidden
    states, by one linear solve (the RMSNorm sphere). It needs d(d+1)/2 contexts and warns with fewer.
  - Numpy only. Tests check each theorem on cases with a known answer (tests/test_geometry.py).
  - `graybox-geometry contexts.txt --provider openai|anthropic --model M --out g.json`: one command gives a chat
    model's observable distances and 2-D coordinates over your contexts. It reads P(yes) exactly with logprobs, or
    uses two sampled halves and the cross-fit. It prints the number of calls and a cost estimate before it runs.

## 0.7.0 (2026-10-18)
**Added**
- `graybox.recalibrate` (`graybox-recalibrate`): set the review cut again on new, unlabelled traffic after a change of
  agent, prompt, tools or tasks. It reports how far the old cut drifted. This is the recipe that a preregistered live
  test confirmed: the re-set cut escalated 4.7% of actions (budget 5%; the old cut 16.6%), and the gate still blocked
  every violation.
- MODEL_CARD.md for the released probes.

**Fixed**
- Tied scores at the cut. Identical actions get identical text-monitor scores, and a quantile cut then escalated the
  whole tied group (37% instead of 5% in a test). `graybox.stats.budget_cut` keeps the budget. Without ties it equals
  the quantile, so existing continuous-score cuts are unchanged. QuantileGate, TextMonitor.fit and the browser trainer
  use it.

## 0.6.0 (2026-10-18)
**Added**
- `graybox.integrations`: the Guard in three agent frameworks, tested offline against the real SDKs:
  - `claude_agent_hook(guard)`: a PreToolUse hook for the Claude Agent SDK. The history comes from the session
    transcript; an escalated call is denied, or sent to the user's permission prompt with `on_escalate="ask"`.
  - `openai_agents_guardrails(guard)`: tool input and output guardrails for the OpenAI Agents SDK. The SDK gives a tool
    guardrail only the run's original input, so the output guardrail records each call and its result, and the
    monitor sees the whole run (a test shows that without it the earlier call is missing and a violation runs).
  - `langgraph_tool_wrapper(guard)`: a `wrap_tool_call` (or async) handler for LangGraph's ToolNode.
  - Extras: `claude-agent`, `openai-agents`, `langgraph`. CI runs them in a separate job.

## 0.5.0 (2026-10-18)
**Added**
- `graybox.mcp_proxy` (`pip install "greybox-monitor[mcp]"`, MCP SDK 2.x): a monitor in front of any MCP server; each
  tool call is checked before it is forwarded; a blocked call returns an error result. Tested end to end with a real MCP
  client and server. The cost of seeing only the tool calls was measured (scripts/117).
- Audit log: `Guard(audit_log=..., on_decision=...)` and the server's `GB_AUDIT_LOG` record every decision, including
  blocked malformed calls; a failing log never changes a decision.
- Canonical mode (`train_text --canonical`, `train_reader --canonical`, the browser trainer's "Canonical actions" box):
  the monitor reads only the decisive fields of each call. A field named like free text (body, subject, reason, ...)
  becomes "[free text]"; quoted prose of 3 or more words and `#` comments are removed. The model file records
  `canonical`, and Guard, the server and the browser then read actions the same way, so an agent cannot move the score
  by rewording what it says about its own call. Python and the browser give identical canonical text on 38,105 steps and
  rewrites (scripts/121). Why: a score-querying attacker who rewrote only free text cut the shipped reader's catch rate
  from 96% to 29% (preregistered T25).
  Off by default: in the preregistered test of the canonical reader (T26) clean catch at 5% fell from 94–98% to 51–59%
  on two closed agents, because a message body can be the leaked content itself; the score did not move under attack.

## 0.4.1 (2026-10-18)
From a second fresh-install user test (PyPI 0.4.0) and a paper audit.

**Fixed**
- Privacy filter: the tokenizer split email addresses into names that survived the term filter. Emails and URLs are now
  masked before the words are counted (the personal part and the path; domains and hosts stay, as policies often
  depend on them), in training and in scoring, identically in Python and the browser (`mask_pii` / `maskPII`; the
  model file records `mask_pii`). Names and company names that recur in the logs still stay; the docs now say so.
- `Guard(hold_tools="tool")` held nothing (a string was read as its characters): fail open. A string is one tool name.
- A reviewer that timed out kept the process alive: it now runs in a daemon thread.
- `check_openai_responses` accepts a plain string as input.
- `--group-by` can use a field of a conversation line (e.g. `"family"`).
- The browser and Python trained slightly different models (scores up to 0.9 logit apart): both solvers now stop at
  tolerance 1e-6. On the GPT example: the same vocabulary, weights within 0.005, scores within 0.007, the same review
  cut and decisions. The browser's per-tool table now uses the same 3 repeats as its headline.
- `train_text`: `--budget` outside (0, 1) and a missing `--out` folder are errors before training; status lines go to
  stderr (the JSON report alone on stdout). `train_reader`: `--force`, and the same checks.
- Server: with `GB_API_KEY`, `/docs` and `/openapi.json` need the key too; `/v1/health` no longer shows file paths.
- The task-shift check (a stored cut on new task families) is now reproducible from public data (`scripts/114`): the
  stored cut caught 0 of 115 violations; a live 5% cut 29 of 115.

**Numbers on the GPT example (privacy filter on)**: tasks held out 0.946 (conversations 0.990); task families held out
0.916 (0.799 without the filter). The GPU reader, tasks held out: 0.972.

## 0.4.0 (2026-10-18)
First PyPI release.

**Fixed**
- `Guard.check_anthropic` and `conversation_steps`: Anthropic SDK content blocks inside a dict message (the SDK's own
  loop, `messages.append({"role": "assistant", "content": response.content})`) lost every earlier call and result.
  On the Claude example data, 1 of 68 violations was escalated instead of 29. SDK objects are now converted
  recursively; a test uses the real `anthropic.types`.
- Cross-validation in `graybox.train_text`, `graybox.train_reader` and the browser trainer now groups by task (the
  first user request), not by conversation: on the GPT example, 0.989 (by conversation) vs 0.948 (by task). Folds are
  identical in Python and the browser (`graybox.stats.group_folds`: tasks with a violation spread over the folds, ties
  ordered by a hash); the estimate is the mean of 3 fold assignments, with its range. One assignment had moved the
  estimate from 0.886 to 0.950, and the two trainers had disagreed.
- The browser importer read a file's kind from its first line; it now reads each line, as Python does (labelled steps,
  OpenAI and Anthropic conversations may share a file), and names the mix.
- Catch rates at a budget send exactly the budget to review (`graybox.stats.caught_at_budget`; ties at the cut count
  in proportion).

**Changed**
- Privacy filter on by default (both trainers): the model file keeps no term with a digit, "@", a long code, a
  random-looking token or a rare word. `--no-privacy` keeps them. It also reduces memorization: with whole task
  families held out, 0.898 vs 0.806 on the GPT example.
- `train_text` does not overwrite `--out` without `--force`; `--eval-only` writes no file; a `--group-by` field that a
  step lacks is an error; the report gives the most a budget can catch; clear errors for bad JSON lines (with the line
  number), JSON array files (now read), labels that match no call id, and missing extras.
- Server: optional `GB_API_KEY`; `GB_GATE_STATE` keeps gates across restarts; an empty tool name and a `budget` without
  `gate` on `/text` are refused; a `warning` when `context` is sent to a monitor not trained on it; `/v1/health`
  reports the version, authentication and gates. The Docker image runs as a non-root user.
- `import graybox` exports `Guard`, `TextMonitor`, `QuantileGate` and `conversation_steps`.
- The block message no longer says that a human will review the action.

**Added**
- `Guard(hold_tools={...})`: every call of the listed tools is escalated, on top of the single budget. (A budget per
  tool was measured and rejected: it caught 39% / 46% of violations against 94% / 98%.)
- `Guard(reviewer_timeout=...)`: no answer in time blocks the action.
- `Guard.check_openai_responses` (OpenAI Responses API items) and `acheck_openai`, `acheck_anthropic`,
  `acheck_openai_responses` for asyncio.
- The browser reader (Qwen3-Embedding-0.6B, float16) in the trainer, for new kinds of tasks (preregistered T19), run on
  several CPU threads through a service worker.
- `SECURITY.md`; CI on Python 3.10–3.13.

## 0.3.x
Research releases installed from git; see the repository history.
