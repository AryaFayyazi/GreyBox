# Grey Box: user guide

Grey Box reads what a closed model is "set to do" from the outside, through its normal API, and turns it into safety
signals you can act on. You need no weights, no special access and no training.

## What it tells you

| Signal | What it means | How it was checked |
|---|---|---|
| **State in a shared 3D space** | Where the model's reading of the conversation sits, next to thousands of open-model readings of benign and harmful conversations | Hidden-state recovery on open models (L0–L2, preregistered) |
| **"About to refuse"** | A forecast, made before the reply, of whether *this* model will refuse | Claude Haiku 5.5, preregistered: AUROC 0.92 for its real refusals; on harmful requests it ties a strong graded moderator (0.87 vs 0.85) |
| **Compliance alarm** | A moderator says the message is Unsafe or Controversial AND the model is under 50% likely to refuse: it may be about to go along with a risky request | Claude, prospective (923 fresh conversations): precision 0.54 vs 0.27 for the moderator verdict alone, 4% alarms on benign messages; a second, graded moderator combined the same way reaches 0.49 |
| **Agent guard** | Before an agent's action executes, the same step is resampled K times; if ≥ τ of the samples break your rules, the action goes to an approver | Only as good as `is_unsafe`. With the environment's exact rule checker: 81% caught at 4% escalation. With an LLM judge as checker: 12% (T3); a judge on the actual action alone does better (41%), and a white-box probe much better (94%). Use the guard only when you have an exact, cheap rule check |

**What it does not do.**
- It is not a better harmful-content filter: a text moderator is better at that, and the alarm uses one.
- It does not see representations that never affect behaviour.
- It is not a lie detector for a model trained to hide things.

## Three ways to use it

### 1. In the browser (no setup)
Open the Grey Box Live page in Claude and chat. Each message uses 4 Claude calls on your account: 2 question sets, a
moderator check and the reply. The readouts arrive at about the same time as the reply. Without live access, the page
shows a recorded session.

### 2. Python SDK (any provider)

```python
import asyncio
from graybox.sdk import Reader, AgentGuard
from graybox.adapters import Anthropic          # or wrap any provider: it needs one async _ask(messages, max_tokens)

reader = Reader.load(Anthropic(model="claude-haiku-5-5", key_env="ANTHROPIC_API_KEY"),
                     bundle="readouts_v4.npz",
                     thresholds={"will_refuse": T_REFUSE, "harmful_latest": T_HARM})   # calibrate on your own benign traffic
r = asyncio.run(reader.read([{"role": "user", "content": "..."}]))
print(r.will_refuse, r.flags, r.point, r.alarm(moderator_verdict="Unsafe"))

async def propose(messages):     # one sample of your agent's next action
    ...
guard = AgentGuard(propose=propose, is_unsafe=my_policy_check, k=8, tau=1/8)
d = asyncio.run(guard.check(messages))   # d.escalate, d.risk, d.samples
```

**Calibration matters.** Thresholds taken from other models gave Claude 25–31% false alarms. Calibrated on 44–48 of
its own benign conversations, they gave about 6%. Use the 95th percentile of the model's own scores on benign
traffic.

### Monitoring an agent's actions (recommended for agents)
Before each tool call runs, score it from the transcript: the agent's instructions, the task, earlier calls with their
results, and the proposed call. You need no access to the agent's model, so this works for closed agents.

```python
from graybox.monitor import ReaderMonitor, QuantileGate, Step
mon = ReaderMonitor.load("reader_probe.joblib")     # Qwen3-8B reader + linear probe; one GPU
gate = QuantileGate(budget=0.05)                    # escalate the riskiest 5% of this agent's actions; no labels
p = mon.score(system_prompt, task, [Step("read_file", {"path": "/hr/x"}, "...")], ("send_message", {...}))
if gate(p):
    hold_for_review()
```

- **Act on the gate, not on p.** The probe ranks a new agent's actions well, but its probabilities do not transfer.
  On Claude Haiku 5.5 a 0.5 cut caught 3% of violations. The gate learns its cut from the agent's own unlabelled
  traffic: the top 5% of its first 500 actions. It then freezes the cut, so a burst of violations cannot raise it.
  On Claude it caught 95% online, with 1.9% of safe actions escalated. With half of a later burst unsafe, it still
  caught 94%; a rolling cut caught 36%.
  - It escalates everything until it has seen 100 actions. Seed it with `gate.fit(scores)` from past transcripts you
    trust.
  - Call `gate.recalibrate()` when the agent, its tools or its policy change.
- **What it was tested on** (preregistered unless noted):
  - Gemma-4 on CAPE-Bench, AUROC 0.994 (same scenarios) and 0.969 (unseen scenario families);
  - gpt-oss after training on Gemma, 0.915;
  - AgentDojo prompt injections, 0.781;
  - Claude Haiku 5.5 after training only on Gemma: 95.6% of violations caught at 5% escalation; AUROC 0.989
    pooled over its 3,128 actions (zero-shot 70B judge: 0.570).
- **Cheaper CPU option: `TextMonitor`.** A TF-IDF classifier trained on your labelled steps. It matches the reader
  when deployment looks like its training data, meaning the same tools and policies. In T9 that held even for another
  agent: on Claude, 0.997 vs the reader's 0.991.
  - It loses ground on unseen task families: 0.950 vs 0.969.
  - It is at chance on a new benchmark: AgentDojo, 0.50 vs 0.78.
  - On Claude, with the label-free cut, it caught 65% of violations where the reader caught 94%.
  - Use the reader when tools, policies or tasks change.
- **Label a few dozen actions of every policy you enforce.** With a policy family removed from training, the monitor
  still ranked Claude's violations of it above safe actions of the same task, but caught 0 of 26 at 5% escalation.
  Adding 20 labelled actions of that policy gave 85%, and 40 gave 94% (T10).
- **Retrain it on your policy.** The shipped probe learned CAPE-Bench's policies. In our tests, 74 labelled examples
  were enough to beat a zero-shot judge.
- **Latency:** 41 ms per action on one GPU (89 ms through the API).
- **The resampling guard (AgentGuard) needs a reliable policy check.** With the environment's own rule checker it
  caught 81% of violations. With an LLM judge as the check it caught 12%. Without such a check, use the reader monitor.

### 3. Server and API
`uvicorn graybox.server:app` serves `/live` (chat with any OpenAI-compatible or Anthropic model and watch its path) and
`/v1/live/sessions`. Keys stay in environment variables, and the request names the variable.
`POST /v1/monitor/action` takes `{instructions, task, history, action, stream, budget}`. It returns `p_unsafe` and
`escalate`, with the escalation cut calibrated per `stream` from that stream's own traffic.

## Cost and speed (measured)

| Mode | Calls per turn | Latency | Cost per turn |
|---|---|---|---|
| Open model, logprobs (88 one-token forks, shared prefix cache) | 88 | 0.6–4 s | GPU only |
| Closed model, one-call transcript (2 samples) | 2 + moderator + reply | about the reply's latency | about $0.001 (Claude Haiku 5.5) |
| Closed model, per-question (88 × 4 samples) | 352 | 25–100 s | about $0.06 at long context |
| Agent guard | K resamples of one step, in parallel | one step | K × one step |
| Agent action monitor (reader) | 1 forward pass of an 8B reader | 41 ms | GPU only |

## Where every number comes from
- docs/PREREGISTRATION_RECORD.md: every confirmatory test, its hash lock and its outcome, including failures.
- docs/research_ledger.md: every experiment and decision, in order.
