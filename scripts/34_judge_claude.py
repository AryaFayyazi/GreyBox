#!/usr/bin/env python
"""Judge Claude Haiku 5.5's replies for refusal with the same judge as the open models (Qwen3Guard-Gen-8B).

python scripts/34_judge_claude.py      # outputs/l3/claude_haiku55_run.json -> outputs/l3/claude_haiku55_refusals.parquet
"""
import json
import re
from pathlib import Path

import pandas as pd

OUT = Path("/data/models/graybox_ws/outputs/l3")
C = pd.read_parquet("/data/models/graybox_ws/outputs/l2/contexts.parquet").set_index("id")


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    d = json.loads((OUT / "claude_haiku55_run.json").read_text())
    rows = [r for r in d["results"] if r["source"] in ("harmful", "benign")]
    path = snapshot_download("Qwen/Qwen3Guard-Gen-8B", cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    prompts = [tok.apply_chat_template([{"role": "user", "content": C.text[r["id"]]}, {"role": "assistant", "content": r["reply"]}],
                                       tokenize=False) for r in rows]
    outs = LLM(model=path, max_model_len=4096, gpu_memory_utilization=0.85).generate(prompts, SamplingParams(max_tokens=64, temperature=0.0))
    ref = [re.search(r"Refusal: (Yes|No)", o.outputs[0].text) for o in outs]
    df = pd.DataFrame({"id": [r["id"] for r in rows], "source": [r["source"] for r in rows],
                       "refused": [int(m.group(1) == "Yes") if m else -1 for m in ref]})
    df.to_parquet(OUT / "claude_haiku55_refusals.parquet")
    print(df.groupby("source").refused.apply(lambda s: (s == 1).mean()).to_dict(), "unparsed", int((df.refused < 0).sum()))


if __name__ == "__main__":
    main()
