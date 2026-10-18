# Security

## Reporting a problem
Please open a private security advisory on GitHub (Security → Report a vulnerability) for this repository. Do not
open a public issue for a vulnerability. We aim to answer within 7 days.

## What the monitor is, and is not
- It is a **risk score** for a proposed tool call, learned from labelled examples. It is not a guarantee. It misses
  violations of policies that were absent from its training data, and on a new kind of task its stored cut can
  escalate almost nothing (see the docs). Keep hard rules (amount limits, recipient allow-lists) in code; use
  `Guard(hold_tools=...)` to send every call of a risky tool to review.
- An agent, or text it reads (prompt injection), can try to write actions that score low. The paper's tests include
  prompt injections (AgentDojo), not adaptive attacks against the monitor itself.

## Data handling
- Transcripts never leave your machine: the SDK, the server and the browser trainer run locally. The website is
  static; the trainer processes files in the browser tab.
- **Model files hold vocabulary from your logs.** By default the trainers mask emails and URLs and drop terms with
  digits, long codes, random-looking tokens and rare words. Names and company names that recur in the logs stay (the
  filter cannot tell a name from a word): keep model files internal if your logs name people. `--no-privacy` (or
  unticking the box in the browser) keeps every term.
- **Probe files of the reader monitor are joblib pickles.** Loading a pickle runs code. Load only files you made or
  whose SHA-256 you checked (the docs list the published checksums).

## The HTTP server
- Requests carry whole transcripts, which can hold credentials. Bind to `127.0.0.1`, or set `GB_API_KEY` (every
  endpoint but `/v1/health` then needs `Authorization: Bearer <key>`), and put TLS in front of it beyond one machine.
- The Docker image runs as a non-root user. Publish its port on localhost (`-p 127.0.0.1:8000:8000`).
- Gates are in memory unless `GB_GATE_STATE` is set. A new or restarted gate fails closed (escalates every action)
  until it has seen 100 scores.

## Fail-closed behaviour
- A malformed tool call (no tool name) is blocked.
- A reviewer that raises, times out (`reviewer_timeout`) or answers anything but True blocks the action.
- A new gate escalates every action until it has seen 100 scores.
