# Changelog

The package is `greybox-monitor` on PyPI; you import `graybox`.

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
