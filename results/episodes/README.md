# Episode-level data of the live self-policing tests

One row per episode; `turns` holds every step as JSON: the candidate draws (tool, arguments, CAPE checker label,
audit score), whether the step went to review, and what ran.

- `gemma4_v4fresh_g9_c100.parquet`, `gemma4_v4fresh_g9_c075.parquet`: G9 (Gemma-4; arms none and gate@1.0; S4@0.75).
  Recompute: `python scripts/152_live_selfaudit_analysis.py "files=[gemma4_v4fresh_g9_c100,gemma4_v4fresh_g9_c075]" endpoints=g9`
  (the script reads them from outputs/live_selfaudit).
- `live_g10_p60.parquet`, `live_g10_p61.parquet`: G10 (Claude Haiku 5.5; arms none and gate; two passes).
  Recompute: `python scripts/157_g10_analysis.py` (reads the .jsonl twins from outputs/claude_selfaudit; the parquet
  files hold the same rows).
