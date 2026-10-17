#!/usr/bin/env python
"""L2 context set: diverse single-turn user messages with concept labels, for grey-box recovery on open LLMs.

Sources (n per source; labels used for concept-transfer tests):
  sst2       movie-review sentences framed as "What do you think of this review?"       label: sentiment (1 = positive)
  harmful    LLM-LAT harmful requests                                                   label: harmful = 1
  benign     LLM-LAT benign counterparts (same dataset's 'chosen'-style benign set)     label: harmful = 0
  boolq      passage + yes/no question                                                  label: gold answer
  gsm8k      grade-school maths word problems                                           label: none
  wiki       WikiText paragraphs framed as "Summarize this:"                            label: none
Output: /data/models/graybox_ws/outputs/l2/contexts.parquet (id, source, text, sentiment, harmful, answer, split)
Split: 70% dev / 30% confirmation, stratified by source, seeded (confirmation rows are not analysed until a
preregistration is locked).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

OUT = Path("/data/models/graybox_ws/outputs/l2")
N = 800
rng = np.random.default_rng(0)


def take(rows, n):
    idx = rng.choice(len(rows), min(n, len(rows)), replace=False)
    return [rows[int(i)] for i in idx]


def main():
    out = []
    sst = load_dataset("stanfordnlp/sst2", split="train")
    for r in take(sst, N):
        out.append({"source": "sst2", "text": f"What do you think of this movie review?\n\n\"{r['sentence'].strip()}\"",
                    "sentiment": int(r["label"])})
    lat = load_dataset("LLM-LAT/harmful-dataset", split="train")
    cols = lat.column_names
    print("LAT columns:", cols)
    for r in take(lat, N):
        out.append({"source": "harmful", "text": r["prompt"].strip(), "harmful": 1})
    try:
        ben = load_dataset("LLM-LAT/benign-dataset", split="train")
        for r in take(ben, N):
            out.append({"source": "benign", "text": r["prompt"].strip(), "harmful": 0})
    except Exception as ex:  # noqa: BLE001
        print("benign set unavailable:", ex)
    bq = load_dataset("google/boolq", split="train")
    for r in take(bq, N):
        out.append({"source": "boolq", "text": f"{r['passage'].strip()}\n\nQuestion: {r['question'].strip()}?",
                    "answer": int(r["answer"])})
    gs = load_dataset("openai/gsm8k", "main", split="train")
    for r in take(gs, N):
        out.append({"source": "gsm8k", "text": r["question"].strip()})
    wk = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    paras = [t.strip() for t in wk.select(range(200000))["text"] if 300 < len(t.strip()) < 1200 and not t.strip().startswith("=")]
    for t in take(paras, N):
        out.append({"source": "wiki", "text": f"Summarize this paragraph:\n\n{t}"})
    df = pd.DataFrame(out)
    for c in ("sentiment", "harmful", "answer"):
        df[c] = df[c].astype("Int64") if c in df else pd.NA
    df["id"] = [f"{s}_{i}" for i, s in enumerate(df.source)]
    df["split"] = "dev"
    for s, g in df.groupby("source"):
        conf = rng.choice(g.index, int(round(0.3 * len(g))), replace=False)
        df.loc[conf, "split"] = "confirm"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "contexts.parquet")
    print(df.groupby(["source", "split"]).size().unstack().to_string())
    print("median chars", df.text.str.len().median())


if __name__ == "__main__":
    main()
