#!/usr/bin/env python
"""Refusal labels for Claude's replies in the L3c data (scripts/50), with the same judge and prompt as every other
refusal label in this project (Qwen3Guard-Gen-8B; scripts/37): the latest user message + the reply -> Refusal: Yes/No.
Writes outputs/livebench_claude_batch/<stage>_refusals.parquet for each stage that has data and no labels yet.

python scripts/52_judge_claude_batch.py
"""
import re
from pathlib import Path

import pandas as pd

CB = Path("/data/models/graybox_ws/outputs/livebench_claude_batch")
LB = Path("/data/models/graybox_ws/outputs/livebench")


def main():
    import importlib
    import sys
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    rec = importlib.import_module("27_live_record")
    path = snapshot_download("Qwen/Qwen3Guard-Gen-8B", cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(model=path, max_model_len=4096, gpu_memory_utilization=0.85)
    sp = SamplingParams(max_tokens=64, temperature=0.0)
    items = pd.read_parquet(LB / "items.parquet").set_index("id")
    for st in ("calib", "live", "evalA", "evalB"):
        f, g = CB / f"{st}.parquet", CB / f"{st}_refusals.parquet"
        if not f.exists() or g.exists():
            continue
        R = pd.read_parquet(f)
        last = [rec.SCRIPT[int(i.split("_")[1])] if st == "live" else items.turns[i][-1] for i in R.id]
        ps = [tok.apply_chat_template([{"role": "user", "content": u[:6000]}, {"role": "assistant", "content": r}], tokenize=False)
              for u, r in zip(last, R.reply)]
        outs = [o.outputs[0].text for o in llm.generate(ps, sp)]
        ref = [re.search(r"Refusal: (Yes|No)", o) for o in outs]
        pd.DataFrame({"id": R.id, "refused": [int(m.group(1) == "Yes") if m else -1 for m in ref]}).to_parquet(g)
        print(st, "refusal rate", round(sum(int(m.group(1) == "Yes") for m in ref if m) / len(ref), 4), flush=True)


if __name__ == "__main__":
    main()
