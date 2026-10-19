#!/usr/bin/env python
"""G6 contexts: fresh contexts that no earlier test used, for the shared-geometry test.
  old kinds (6): items of L2's six sources that are NOT in L2's contexts (exact text match excluded)
  new kinds (4): AG News articles, MBPP programming tasks, JailbreakBench behaviours (harmful and benign goals),
                 deepset prompt injections (injections and benign prompts); never used in this project
100 contexts per kind, numpy seed 6. Output: /data/models/graybox_ws/outputs/g6/contexts.parquet
Run with HF_HOME on /data (the root disk is full).
python scripts/136_g6_contexts.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

OUT = Path("/data/models/graybox_ws/outputs/g6")
L2 = Path("/data/models/graybox_ws/outputs/l2")
N = 100
rng = np.random.default_rng(6)


def take(items, n, used):
    items = [x for x in items if x["text"] not in used]
    idx = rng.choice(len(items), min(n, len(items)), replace=False)
    return [items[int(i)] for i in np.sort(idx)]


def main():
    used = set(pd.read_parquet(L2 / "contexts.parquet").text)
    out = []
    sst = load_dataset("stanfordnlp/sst2", split="train")
    out += [dict(x, kind="sst2") for x in take([{"text": f"What do you think of this movie review?\n\n\"{r['sentence'].strip()}\""} for r in sst], N, used)]
    lat = load_dataset("LLM-LAT/harmful-dataset", split="train")
    out += [dict(x, kind="harmful") for x in take([{"text": r["prompt"].strip()} for r in lat], N, used)]
    ben = load_dataset("LLM-LAT/benign-dataset", split="train").select(range(20000))
    out += [dict(x, kind="benign") for x in take([{"text": r["prompt"].strip()} for r in ben], N, used)]
    bq = load_dataset("google/boolq", split="validation")          # L2 used the train split
    out += [dict(x, kind="boolq") for x in take([{"text": f"{r['passage'].strip()}\n\nQuestion: {r['question'].strip()}?"} for r in bq], N, used)]
    gs = load_dataset("openai/gsm8k", "main", split="test")        # L2 used the train split
    out += [dict(x, kind="gsm8k") for x in take([{"text": r["question"].strip()} for r in gs], N, used)]
    wk = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")   # L2 used train
    paras = [t.strip() for t in wk["text"] if 300 < len(t.strip()) < 1200 and not t.strip().startswith("=")]
    out += [dict(x, kind="wiki") for x in take([{"text": f"Summarize this paragraph:\n\n{t}"} for t in paras], N, used)]
    ag = load_dataset("fancyzhx/ag_news", split="test")                      # the canonical repo id of ag_news
    out += [dict(x, kind="news") for x in take([{"text": r["text"].strip()} for r in ag], N, used)]
    mb = load_dataset("google-research-datasets/mbpp", split="test")
    out += [dict(x, kind="code") for x in take([{"text": r["text"].strip()} for r in mb], N, used)]
    jb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    jbb = [{"text": r["Goal"].strip(), "harmful": 1} for r in jb["harmful"]] + [{"text": r["Goal"].strip(), "harmful": 0} for r in jb["benign"]]
    out += [dict(x, kind="jailbreak_bench") for x in take(jbb, N, used)]
    pi = load_dataset("deepset/prompt-injections", split="train")
    out += [dict(x, kind="injection") for x in take([{"text": r["text"].strip(), "injection": int(r["label"])} for r in pi], N, used)]
    df = pd.DataFrame(out)
    df["id"] = [f"g6_{k}_{i}" for i, k in enumerate(df.kind)]
    df["new_kind"] = df.kind.isin(["news", "code", "jailbreak_bench", "injection"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "contexts.parquet")
    print(df.groupby("kind").size().to_dict(), "overlap with L2:", int(df.text.isin(used).sum()))


if __name__ == "__main__":
    main()
