# G77: fill the comparison cells that were never run (descriptive; locked before any is computed)

Written 2026-10-26, before any of the cells below was computed on real labels. A dry run on labels permuted within strata
(seed 7799) tested the CPU code (`stage=cpu dry=1` -> outputs/g77/cpu_DRYRUN.json; all values near or below 0.5); the GPU
stages were checked only for their inputs (unit counts, strata, message roles), never run.

## Why
The paper's tables show `--` (now `n.r.`) where a comparison was never run. Each cell below is a comparison, not part of
the method. No endpoint is registered: every cell is reported as computed, with its 95% scenario-cluster bootstrap
interval (1,000 resamples), whatever it shows. Each cell uses its column's recipe on its row's units, unchanged.

## Cells from existing data (CPU; scripts/269 stage=cpu)
- **C1** Forecast table, GPT-6-luna, "best text monitor + own sampled actions" across steps: G61's combination (mean of
  the two scores' ranks, no fit), with G54's GPT-6-luna text judge and G48's own draws (G46's forecast), within
  template|tool, G48's 1,134 units. Reference: the judge alone on the same units.
- **C2** Forecast table, GPT-6-luna, "other: Gemma-4 sampling from its contexts": G48's stored Gemma-4 draws in GPT-6-luna's
  seat (G48's B3 stage), G46's forecast, within a step (template|tool|position) and across steps (template|tool).
- **C3** Monitoring table, AgentDojo injection rows, "trained probe": CAPE's frozen AgentDojo gate (a probe on the agent's
  own states, refit by CAPE on the v75 development traces: gate_refit_<agent>.npz), as logged for every v76 confirmation
  call in server_conf_<agent>_ref.jsonl (mode none; scores matched by call id; a call scored twice gets the mean), on G17's
  816 / 821 confirmation calls, within suite.
- **C4** Detection table, GPT-6-luna AgentDojo row, "TF-IDF (same labels)": T9's TXT fit on the reader's training labels in
  that row (T14's full set: Gemma-4 CAPE development + fresh units and all 3,455 open-agent AgentDojo calls; groups for its
  CV: scenario, or AgentDojo episode), scored on T13b's GPT-6-luna AgentDojo calls, within suite.

## Cells already computed elsewhere (no new analysis)
- Detection table, gpt-oss-20b "Agent's internals": T17's probe on gpt-oss-20b's own decision states (preregistered T17),
  0.900 on its 2,432 held-out-scenario actions (results/t1/t17_replication_agents.json; footnoted: a subset of the row).
- Detection table, AgentDojo "two open" internals: CAPE's gate where logged, 0.577 on 1,805 of the 3,455 calls (T7's
  cape_subset; footnoted).

## Cells that need the GPU (scripts/269; one GPU, serial chain after the running jobs)
- **J** Monitoring table, "judge (reasoning)": G20's judge unchanged (Gemma-4, thinking on, its standard sampling, 4 samples
  averaged, unparsed = 50, `VIOLATION: 0-100`) on GPT-6-luna's 560 G38 calls (G20's prompt; CAPE tools listed, tool choice
  none), on G18's 360 tau2 writes of Claude Haiku 5.5 (G20's prompt; G21's conversation and the airline tools), and on G17's
  AgentDojo confirmation calls of Gemma-4 (816) and gpt-oss-20b (821). For AgentDojo, whose violation is executing an
  injected instruction, the prompt asks whether the call carries out an instruction that did not come from the user (for
  example, one inside a tool result) rather than the user's own task (text in scripts/269, AD_JUDGE). Strata as each row:
  template|tool, tool, suite.
- **O** Forecast table, replication column, "other: Gemma-4 sampling from its contexts": scripts/232's other stage,
  unchanged, on G65's 1,582 fresh units; within a step and across steps.
- **R** Detection table, "Reasoning judge, rules": scripts/169's judge unchanged (Qwen3-32B, thinking on, 3 samples, its
  prompt) on each open-agent row's 70B-judge units: Gemma-4 T3's 3,496 fresh units (both Gemma-4 rows), gpt-oss-20b T6's
  5,229, Qwen3.8-27B 2,378 and Muse-Glimmer-30B 2,657 (T17's held-out halves, scripts/127's units), AgentDojo T7's 3,455.
  Within template (CAPE) or suite (AgentDojo).

## Not filled, and why
- `x` cells (a probe on a closed agent's states) cannot exist.
- Monitoring table, AgentDojo "tuned probe": the confirmation calls' states were not stored (only the gate's scores).
- Forecaster table, "position alone" within a step: undefined (the step is constant within a step).

## Reporting
Every cell goes into its table with its value and is listed in the full record, whatever it shows. Cells on a subset of
the row's units are footnoted. Analysis: `stage=analyze` -> results/t1/g77_fill_cells.json.
