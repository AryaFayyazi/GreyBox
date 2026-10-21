#!/usr/bin/env python
"""Run scripts/151's live protocol (imported unchanged) on chosen templates of core24b's HELD-OUT scenario hash.
Used by G12 (docs/prereg_G12_composition_law.md): any split, chosen templates, every scenario hash. Output: outputs/live_selfaudit/{actor}_{split}_{tag}.parquet
  python scripts/158_live_targeted.py actor=qwen38 "templates=[con_ci_green,tp_campaign_test]" seeds=[0,1,2,3,4,5] arms=[none,gate] cut=1.0 escalate=review tag=g11b_c100
"""
import asyncio
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
L = importlib.import_module("151_live_selfaudit")


def held(s):  # kept for reference; G12 uses every scenario of the split
    return int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % 2 == 1      # T17's hash: odd = held-out


def main():
    cfg = OmegaConf.merge(L.D, {"templates": [], "split": "core24"}, OmegaConf.from_cli())
    a = L.srv.ACTORS[cfg.actor]
    specs = [json.loads(x) for x in open(L.CAPE / f"outputs/capebench/{cfg.split}/scenarios.jsonl")]
    specs = [s for s in specs if s["cf_value"] == "unsafe" and not s["nuisance"] and s["template"] in set(cfg.templates)]
    print(len(specs), "scenarios in", list(cfg.templates), flush=True)
    L.OUT.mkdir(parents=True, exist_ok=True)
    if cfg.vllm_python:
        sys.executable = cfg.vllm_python
    proc = L.srv.start_server(a, cfg.port)
    try:
        out = asyncio.run(L.run(cfg, a, specs))
    finally:
        proc.terminate()
    df = pd.DataFrame(out)
    df.to_parquet(L.OUT / f"{cfg.actor}_{cfg.split}_{cfg.tag}.parquet")
    print(json.dumps(df.groupby("arm")[["unsafe_episode", "success", "n_reviews", "n_steps"]].mean().round(4).to_dict(), indent=1), flush=True)


if __name__ == "__main__":
    main()
