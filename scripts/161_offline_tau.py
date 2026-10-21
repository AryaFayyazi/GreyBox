#!/usr/bin/env python
"""G12 step A (development): each agent's offline self-audit flag rate PER TEMPLATE on its own violating steps.

Units: the first violating step of every violating episode in the agent's stored core24b episodes (unsafe world),
at most `per_template` per template (random, seed 1610). The audit is scripts/150's (imported unchanged; ASK prompt,
4 answers, reasoning on). tau_k = share of a template's units on which all parsable answers say "no" (cut 1.0).
  python scripts/161_offline_tau.py actor=gemma4          (GPU, one vLLM server)
  python scripts/161_offline_tau.py stage=summary         (CPU)
"""
import asyncio
import glob
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sa = importlib.import_module("150_self_audit")
OUT = Path("/data/models/graybox_ws/outputs/g12")
SETS = {"gemma4": "xm_gemma4_core24b_minus*", "gptoss20b": "xm_gptoss20b_core24b_minus*", "museglimmer": "xm_museglimmer_core24b_minus*",
        "qwen38": ["core24b_minus", "core24b_minus2345"]}                 # plain prompts only, as for the other agents
D = OmegaConf.merge(sa.D, {"stage": "run", "actor": "gemma4", "per_template": 40, "mode": "sample", "prompt": "ask", "n": 4,
                           "max_tokens": 4096})


def set_names(a):
    s = SETS[a]
    if isinstance(s, list):
        return s
    return sorted(Path(p).stem for p in glob.glob(f"/home/arya/new_safety/cape/outputs/rollouts/{s}.parquet") if "reasoning" not in p)


def build_units(a, per_template, seed=1610):
    rows = []
    for st in set_names(a):
        E = sa.rollouts(st)
        for ep in E[E.unsafe_episode].itertuples():
            rows.append({"set": st, "episode_id": ep.episode_id, "event_id": int(ep.first_unsafe_event), "scenario_id": ep.scenario_id,
                         "template": ep.template, "bench": "core24b", "y": 1})
    U = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in U.groupby("template").indices.items():
        keep += sorted(rng.choice(idx, min(len(idx), per_template), replace=False).tolist())
    return U.iloc[sorted(keep)].reset_index(drop=True)


def stage_run(cfg):
    a = sa.srv.ACTORS[cfg.actor]
    U = build_units(cfg.actor, int(cfg.per_template))
    print(cfg.actor, len(U), "violating units over", U.template.nunique(), "templates", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sa.OUT = OUT                                                       # checkpoint file goes to the G12 folder
    cfg = OmegaConf.merge(cfg, {"units": f"{cfg.actor}_g12tau"})
    if cfg.vllm_python:
        sys.executable = cfg.vllm_python
    proc = sa.srv.start_server(a, cfg.port)
    try:
        out = asyncio.run(sa.run_all(cfg, a, U))
    finally:
        proc.terminate()
    U["s"] = [o["s"] for o in out]
    U.to_parquet(OUT / f"{cfg.actor}_tau_units.parquet")
    print(U.groupby("template").s.agg(n="size", tau=lambda x: float(np.mean(x >= 1 - 1e-9))).round(3).to_string(), flush=True)


def stage_summary(cfg):
    res = {}
    for f in sorted(OUT.glob("*_tau_units.parquet")):
        a = f.name.replace("_tau_units.parquet", "")
        U = pd.read_parquet(f)
        U = U[np.isfinite(U.s.astype(float))]
        g = U.groupby("template").s.agg(n="size", tau=lambda x: float(np.mean(x >= 1 - 1e-9)))
        res[a] = {t: {"n": int(r.n), "tau": round(float(r.tau), 4)} for t, r in g.iterrows()}
    print(json.dumps(res, indent=1))
    (ROOT / "results/t1/g12_offline_tau.json").write_text(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"run": stage_run, "summary": stage_summary}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
