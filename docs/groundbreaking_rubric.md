# Rubric: what "groundbreaking" requires, scored honestly

Written 2026-10-20 at the user's request. Each criterion is scored 0 (absent), 1 (partial), 2 (met), with the evidence
and the action that would raise it. The target is 2 on every criterion marked **core**, and at least 1 elsewhere.
Scores change only on preregistered results (exploratory results can move a score to 1, not 2).

| # | Criterion | Score | Evidence now | Action to reach 2 |
|---|---|---|---|---|
| R1 **core** | **A new, one-sentence claim that changes practice.** | 1.5 | "An agent's own API suffices to prevent the violations its transcript decides; what it cannot prevent is exactly the violations decided by facts it has not checked (Corollary 1)." Confirmed on Claude (G10) and Gemma-4 (G9). | Add the predictive part: "how much an agent can police itself is known before deployment from a cheap offline audit" (G12). |
| R2 **core** | **Strength of evidence**: preregistered, locks published first, held-out data, CIs, failures reported. | 2 | G1–G10 locks published before data; failed endpoints reported (G8 C1, G9 L2, G11/G11b deviations). | Keep. |
| R3 **core** | **Generality**: ≥ 2 benchmarks, ≥ 3 model families, open and closed, more than one behaviour. | 1 | Self-policing: one benchmark (CAPE); 2 live agents (Gemma-4, Claude); Claude's violations are two behaviours. Offline audits on 4 open agents. | A second benchmark (AgentDojo prompt injection; open agents first, free), offline then live; G12 adds live points for gpt-oss, Qwen3.8, Muse-Glimmer. |
| R4 **core** | **A predictive theory confirmed prospectively** (a quantitative prediction made before the data). | 1 | Prevention law; Corollary 1's boundary predicted and confirmed (G10 C2). G11 stopped (no base rate); G11b's pooled prediction missed Qwen3.8 (composition error, diagnosed before the Muse outcome). Exploratory: per-template and Claude points within 0.08. | G12: composition-weighted law, per-template rates measured offline, 4 agents, locked before any core24 episode. |
| R5 | **Practical impact**: deployable, cheap, released, safety gain without utility loss. | 2 | graybox.selfaudit in greybox-monitor 0.9.0 (PyPI, verified); G9 success +10 pts; G10 success unchanged; cost ~$0.004 per Claude episode. | Keep. |
| R6 | **Failure modes characterized**: when it fails, why, and what to do instead. | 2 | Agents differ (97%…5% self-recognition); plain question fails on Claude, rule-by-rule works; unchecked facts need a world check; hold-and-verify did not work; cross-audit shows verification, not self-knowledge. | Keep; add AgentDojo. |
| R7 | **Prior art differentiated.** | 1.5 | ODCV-Bench (post-hoc recognition), CoT Red-Handed, Monitoring Monitorability, Reason Less Verify More, Ctrl-Z resampling, Carlini, Picozzi, Huh. | A short, explicit "what is new" paragraph in related work. |
| R8 | **One coherent story and a clear figure.** | 1 | Two threads (observable geometry; monitoring); self-policing table; law figure (exploratory). | Unify: "what crosses an API boundary, and what it lets an outsider (or the agent itself) do"; the law figure with preregistered points in the main text. |

## Work queue derived from the rubric (in order)
1. G12 step A (offline per-template rates; running after G11b) → lock G12 → step B (live, core24, 4 agents). [R1, R3, R4]
2. AgentDojo, offline self-audit of Gemma-4's and gpt-oss's own injected and benign calls (stored traces; free) →
   if the audit separates them, a preregistered live AgentDojo test on an open agent. [R3, R6]
3. Paper: unify the story; related-work "what is new"; law figure; update scores here after each result. [R7, R8]
