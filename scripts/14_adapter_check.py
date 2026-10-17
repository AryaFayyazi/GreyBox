#!/usr/bin/env python
"""Check the L3 adapter end to end on an open model served behind an OpenAI-compatible API (vLLM).

The same 100 dev contexts x 88 battery items go through `adapters.OpenAICompatible` in the logprob tier and the
sample tier (n=16). We compare θ(q) with the offline interrogation (scripts/12): per-entry correlation and
mean |Δθ|. Passing this means the API path measures the same Ψ as the offline harness.

python scripts/14_adapter_check.py model=qwen3_8b
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.adapters import OpenAICompatible  # noqa: E402
from graybox.battery_v2 import BATTERY_V2  # noqa: E402
import importlib  # noqa: E402

run12 = importlib.import_module("12_l2_run")
L2 = Path("/data/models/graybox_ws/outputs/l2")
RES = Path("/home/arya/graybox/results/l2")
D = OmegaConf.create({"model": "qwen3_8b", "n_ctx": 100, "port": 8141})


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    m = run12.MODELS[cfg.model]
    path = run12.path_of(m["hf"])
    env = dict(os.environ, LD_LIBRARY_PATH="/data/models/cider_ws/envs/cider/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
    proc = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", path,
                             "--served-model-name", "bb", "--port", str(cfg.port), "--max-model-len", "4096",
                             "--max-logprobs", "20", "--gpu-memory-utilization", "0.85"], env=env)
    try:
        for _ in range(360):
            try:
                if httpx.get(f"http://127.0.0.1:{cfg.port}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(5)
        C = pd.read_parquet(L2 / "contexts.parquet")
        idx = np.flatnonzero((C.split == "dev").values)[:: max(1, (C.split == "dev").sum() // cfg.n_ctx)][: cfg.n_ctx]
        ctx = C.text.values[idx].tolist()
        extra = {"chat_template_kwargs": m["kw"]} if m["kw"] else {}
        out = {}
        for tier, lp in (("logprob", True), ("sample16", False)):
            a = OpenAICompatible(model="bb", base_url=f"http://127.0.0.1:{cfg.port}/v1", key_env=None, logprobs=lp,
                                 n_samples=16, concurrency=64, extra=extra)
            t0 = time.time()
            r = asyncio.run(a.battery(ctx, BATTERY_V2))
            out[tier] = np.array([x[0] for x in r], float).reshape(len(ctx), len(BATTERY_V2))
            print(tier, f"{time.time()-t0:.0f}s", a.usage, flush=True)
        Z = np.load(L2 / cfg.model / "psi.npz")
        q_off = ((Z["p_yes"] + 1e-6) / (Z["p_yes"] + Z["p_no"] + 2e-6))[idx]
        th = lambda q: 2 * np.arcsin(np.sqrt(np.clip(q, 0, 1)))  # noqa: E731
        res = {"model": cfg.model, "n_ctx": len(ctx)}
        for tier, q in out.items():
            ok = np.isfinite(q)
            res[tier] = {"corr_theta": round(float(np.corrcoef(th(q[ok]), th(q_off[ok]))[0, 1]), 4),
                         "mean_abs_dtheta": round(float(np.abs(th(q[ok]) - th(q_off[ok])).mean()), 4),
                         "frac_parsed": round(float(ok.mean()), 4)}
        RES.mkdir(parents=True, exist_ok=True)
        (RES / f"adapter_check_{cfg.model}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
