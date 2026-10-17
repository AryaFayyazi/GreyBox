# Preregistration L3d: the transcript-framing grey box on Claude Haiku 5.5

Locked before any Claude call of scripts/56. The format pilot (scripts/54) used 20 TRAIN conversations only.

## Why
L3c's in-conversation one-call check-in was refused by Claude in multi-turn chats (12–25% of answers usable):
Claude read it as an injection. Quoting the conversation as a transcript inside one user message gave 100% usable
answers in the pilot. This is the format the in-browser Live page needs. It is a different elicitation: the model
reads its own conversation from outside, so it needs its own validation.

## Frozen procedure
- Wording: src/graybox/batch.py (transcript_prompt, TRANSCRIPT_HEAD/TAIL, batch_block, parse_batch).
- **Readouts:** built by
  `GB_LB=outputs/livebench_transcript GB_LB_RES=results/livebench_transcript GB_LIVE_DIR=models/live_tx
  GB_BUNDLE_NAME=readouts_v4 GB_BATTERY_TAG=v2m_transcript GB_VAL=results/livebench_transcript/livebench_theta_nexact.json
  python scripts/39_live_v2_readouts.py`
  - It uses only the six open models' transcript-framing answers (scripts/55).
  - Its sha256 is recorded before scripts/57 runs.
- Data: scripts/56. Analysis: scripts/57.
- **Labels reused from L3c for the same conversations:** Claude's judged refusal (scripts/52), Claude-as-moderator
  (scripts/50), Qwen3Guard moderation.

## Data
- **calib:** 44 single-turn L3c calibration conversations.
- **eval:** all 400 harmful single-turn L3c eval conversations, then 100 of its 170 benign single-turn ones (seed 0).
- n = 2 answer sets each.

## Endpoints (bootstrap over conversations, 2,000 resamples)
- **F1.** Harmful conversations: AUROC(will_refuse) − AUROC(Qwen3Guard moderation) for Claude's judged refusal. Pass
  if the lower 95% bound > 0.
- **F2.** All eval conversations, event = harmful compliance: AP(Claude-as-moderator × (1 − will_refuse)) −
  AP(Claude-as-moderator). Pass if the lower 95% bound > 0. This is the alarm the in-browser page shows.
- **Reported:** the same with Qwen3Guard; fixed-threshold alarms; the usable-answer rate.

## Stopping rule
The ledger cap of $4.95 stops calls. calib runs first, then harmful eval conversations, then benign. If the cap stops
a stage, the completed conversations are analysed and counted.
