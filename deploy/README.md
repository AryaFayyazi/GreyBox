# Self-hosting the Grey Box monitor

Transcripts never leave your machine.

**CPU, real time (text monitor trained in the browser trainer):**

```
docker build -t greybox -f deploy/Dockerfile .
docker run -p 8000:8000 -v $PWD/greybox-text-monitor.json:/models/text.json -e GB_TEXT_MONITOR=/models/text.json greybox
curl -X POST localhost:8000/v1/monitor/text -H 'content-type: application/json' \
  -d '{"instructions": "...", "task": "...", "history": [], "action": {"tool": "send_message", "args": {"recipient": "..."}}}'
```

**GPU reader monitor (robust to new tasks; about 16 GB of GPU memory, 41 ms per action):**

```
docker build -t greybox-reader --build-arg READER=1 -f deploy/Dockerfile .
docker run --gpus all -p 8000:8000 -v $PWD/models:/models -e GB_MONITOR_PROBE=/models/reader_probe.joblib greybox-reader
```

The first request downloads Qwen/Qwen3-8B from Hugging Face. Act on `escalate`, not on `p_unsafe`. Each `stream`
calibrates its own frozen cut from its first 500 actions.
