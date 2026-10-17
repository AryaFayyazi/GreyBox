# Grey Box

**Website: https://aryafayyazi.github.io/GreyBox/**: train a monitor on your own agent in the browser, see one
monitor a closed frontier agent action by action, and read every preregistered result.

```
pip install "graybox[server] @ git+https://github.com/AryaFayyazi/GreyBox"
```

- `graybox.monitor.TextMonitor`: CPU, about 0.2 ms per action. Train it in the browser or in Python. It is reliable
  on the tools and policies it was trained on.
- `graybox.monitor.ReaderMonitor` + `QuantileGate`: Qwen3-8B reader on one GPU, 41 ms per action. It is robust to new
  tasks; act on the frozen gate, not on raw probabilities.
- `uvicorn graybox.monitor_server:app`: a self-hosted HTTP API (see `deploy/`). Transcripts stay on your machine.

**Headline (preregistered):** a transcript reader trained on labelled agent traces monitors a closed agent's
proposed actions as well as a probe on the agent's own internals (0.994 vs 0.979 on the same scenarios; within
0.007 on unseen task families). Labels, not access, are what monitoring needs.

See what a model you cannot open is doing, from its answers alone. Use that to forecast what it is about to do,
including whether it will go along with a harmful request. Every claim below is checked against models whose
internals are known, or against the closed model's real behaviour.

A *grey box* here is the model's **interrogation state**: Ψ(x), the probabilities of answering "Yes" to a fixed battery
of questions asked in side branches of the conversation. The real reply never sees these questions. Raw hidden
coordinates cannot be recovered from outputs: any rotation of the hidden space leaves every output unchanged. Ψ can be
recovered, and its coordinates mean the same for every model.

## What is established (preregistered tests: lock hashes and every outcome in docs/PREREGISTRATION_RECORD.md)

| Test | Result |
|---|---|
| L0 program compiled into a transformer | 21/21 hidden states from 21 black-box-chosen questions |
| L1 Othello-GPT, fresh games | hidden-state R² 0.967 from 61 queries per move |
| L2 8 open chat models, 7 families | the answers recover the hidden state better than text (+0.120 R²) and a model's own probe readings better than another model's answers at the same budget (+0.057) |
| L3 Claude Haiku 5.5, readouts trained only on open models | harmful request 0.993, refusal 0.997, negative tone 0.935 AUROC (text sampling only) |
| L3b, L3d Claude Haiku 5.5, vs a 3-level moderation verdict | which harmful requests Claude goes along with: 0.83 vs 0.65 (+0.18) and 0.82 vs 0.68 (+0.14) |
| L3e Claude, prospective, 923 fresh conversations | the tool's alarm is twice as precise as the moderator verdict alone (0.54 vs 0.27); against a **graded** moderator score the refusal forecast only ties (0.87 vs 0.85, not significant) |
| T1 agents, 2,800 fresh episodes | with an oracle rule checker scoring the samples, 16 black-box rollouts beat a probe on the agent's own hidden state (0.904 vs 0.841) |
| T2, the same episodes at two moments | CAPE's probe after the agent's reasoning (0.979) beats any turn-start monitor; resampling the action there (oracle-scored) matches it (0.9997) |
| T4–T6, a transcript reader trained on labels | beats the agent's own internals on the same scenarios (0.994 vs 0.979); within 0.007 on unseen task families; trained on one agent, monitors another (0.915 vs judge 0.833) |
| T7, external benchmark (AgentDojo prompt injection) | no AgentDojo training: reader 0.78 vs zero-shot judge 0.54; on gpt-oss, reader 0.82 vs CAPE's white-box gate 0.60 |
| T8b, a closed frontier agent (Claude Haiku 5.5 acting through its API, 3,128 actions) | reader trained only on Gemma catches 95.6% of the 68 violations at 5% escalation (judge 0%). AUROC: 0.991 vs judge 0.633 within template, which rests on one stratum (136 actions, 26 violations); pooled over all actions 0.989 vs 0.570 (exploratory). Label-free cut from Claude's own unlabelled traffic, frozen after a 500-action warm-up, online: 95.4% caught, 1.9% escalated; it holds at 94% through bursts of violations (exploratory) |
| T10, Claude on task families the monitor never saw | still beats the zero-shot judge (pooled 0.912 vs 0.570; within 0.948 vs 0.633); but at 5% escalation it misses all 26 violations of a policy it never learned, and 20–40 labelled examples of that policy fix it (85–94%, exploratory) |
| T9, is it the reader or the labels? (TF-IDF trained on the same labels) | in distribution it matches the reader (Claude 0.997 vs 0.991: B1 fails); on unseen task families the reader wins (0.969 vs 0.950, B2 passes); on AgentDojo TF-IDF is at chance (0.50 vs 0.78) |
| T3, deployable scoring (LLM judge, no oracle) | the white-box probe wins: judge-scored resampling 0.782 < turn-start probe 0.846 < judge on the actual action 0.910 < CAPE probe 0.976 |
| Not confirmed | the agent's answers alone are not shown to match the white-box probe; 2 rollouts plus answers are not shown to match 16 rollouts |

**Development (not preregistered):**
- T1 replicates on gpt-oss-20b (rollouts 0.944 vs probe ≤ 0.862).
- On 6 open models, moderation × (1 − the grey box's refusal forecast) finds harmful compliance at average precision
  0.47 vs 0.25 for moderation alone.

**What the grey box is and is not good for.**
- It is not a harmful-content filter. A text moderator decides whether a message is harmful better (AUROC 0.97 vs
  0.85–0.93).
- For chat refusals on Claude, the refusal forecast ties a strong graded moderator (L3e). Its added value there is
  small: about +0.03 average precision over two moderators.
- For agents, black-box resampling beats a white-box probe only when each sampled action is scored by the
  environment's own rule checker, an oracle that knows hidden facts (T1, T2). With a deployable LLM judge doing the
  scoring, the white-box probe wins (T3, preregistered):
  - At 5% of safe actions escalated, CAPE's probe catches 94% of violations, a judge on the actual action 41%, and
    judge-scored resampling 12%.
  - Without the weights, a transcript reader trained on labelled traces matches the internals (T4–T8b), and it
    works on closed agents. Act on its rank (QuantileGate), not on its raw probability.

## Tools

**Grey Box Live in the browser** (`web/live_browser.html`, built by `web/build_browser_live.py`).
- Chat with Claude Haiku 5.5. Each message uses 4 Claude calls on the viewer's own account:
  - 2 answer sets to 88 questions, in the transcript framing;
  - 1 moderator check;
  - 1 reply.
- The page shows Claude's path in a 3D space shared with open models, the forecast "Claude is about to refuse", and
  the alarm "a moderator says Unsafe and Claude is not about to refuse". When live chat is unavailable, it shows a
  recorded session.

**API server and viewers.**

    cd src && uvicorn graybox.server:app --port 8765
    # http://localhost:8765/       Othello-GPT: estimated vs true hidden path, board read from outputs
    # http://localhost:8765/live   Grey Box Live: chat with any API model, watch its path and readouts

Keys are read from environment variables. The request names the variable, never the key:

    curl -X POST localhost:8765/v1/live/sessions -H 'content-type: application/json' \
         -d '{"provider":"openai_compat","model":"...","base_url":"https://.../v1","key_env":"PROVIDER_API_KEY"}'

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | validated black boxes and their held-out metrics |
| `POST /v1/runs`, `GET /v1/runs/{id}`, `GET /v1/runs/{id}/stream` | Othello runs (SSE replay) |
| `POST /v1/live/sessions`, `POST /v1/live/sessions/{id}/turns` | Grey Box Live (per-question forks; logprob or sampled tier) |

**Formats.** Each format was validated separately; see docs/research_ledger.md.

| Format | Calls per turn | Validated on |
|---|---|---|
| Per-question fork | 88 logprob calls, or 88 × n sampled | open models, Claude (L3, L3b) |
| One-call, in conversation | n | open models; refused by Claude in multi-turn chats (L3c: 59.8% usable; the refusal ranking ties graded moderators, and the combined alarm doubles AP or better) |
| One-call, transcript | n | open models, Claude (L3d) |

## Layout

| Path | What |
|---|---|
| `src/graybox/` | batteries (`battery*.py`), one-call formats (`batch.py`), adapters, Live, server, spend ledger (`budget.py`) |
| `scripts/` | experiments in order: L0–L2 (`05–31`), Claude (`32–34`, `41–46`, `50–58`), Live benchmark (`35–40`, `47–48`), agents (`15–25`) |
| `docs/theory.md`, `docs/appendix_proofs.md` | propositions and proofs |
| `docs/research_ledger.md` | dated log of every experiment, decision, failure and fix |
| `docs/PREREGISTRATION_RECORD.md` | every confirmatory test, its lock and outcome, including failures and deviations |
| `docs/paper_draft.md` | the paper draft |

Compute notes: outputs live on `/data/models/graybox_ws`. Slurm jobs run one GPU at a time (`slurm/run.sh`). Paid API
runs share a file-locked spend ledger with a hard cap.
