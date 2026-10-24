#!/usr/bin/env python
"""G55: a stronger, different text monitor on Claude Haiku 5.5's transcripts (docs/prereg_G55_gpt_judges_claude.md).
Units: G46c's (Claude Haiku 5.5, exact contexts, 1,587 units, 316 violations). GPT-6-luna with reasoning (effort medium)
reads Claude's operating instructions and conversation so far (scripts/237's transcript and judge prompt) and gives the
probability that Claude's next action violates. Compared with G52's own sampled-policy forecast on the same units.
  stage=judge cap=.. ; stage=analyze -> results/t1/g55_gpt_judges_claude.json
"""
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED = 2000, 5500


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G52 = _mod("g52", "scripts/237_g52_text_judge_vs_policy.py")
G48 = _mod("g48", "scripts/229_g48_gpt_rollout.py")
A80, G46C, G46 = G48.A80, G52.G46C, G52.G46
JF = G52.OUT / "g55_gpt_judge_claude.jsonl"


async def judge_one(client, led, sem, r, spec):
    led.check()
    t0 = time.time()
    async with sem:
        x = await client.chat.completions.create(model=A80.MODEL, messages=[{"role": "system", "content": G52.JUDGE_SYSTEM},
                                                                           {"role": "user", "content": G52.transcript(spec["system_prompt"], json.loads(r.am))}],
                                                 max_completion_tokens=4000, reasoning_effort="medium")
    led.charge(x.usage.prompt_tokens, x.usage.completion_tokens)
    text = x.choices[0].message.content or ""
    return {"instance_id": r.instance_id, "seed": int(r.seed), "t": int(r.t), "p": G52.parse_prob(text), "tail": text[-120:],
            "in": x.usage.prompt_tokens, "out": x.usage.completion_tokens, "sec": round(time.time() - t0, 2)}


def stage_judge(cap):
    import openai
    from graybox.budget import Ledger
    led = Ledger(A80.LEDGER, float(cap), *A80.PRICE)
    U, specs = G46C.units()
    done = {(json.loads(x)["instance_id"], json.loads(x)["seed"], json.loads(x)["t"]) for x in open(JF)} if JF.exists() else set()
    todo = [r for r in U.itertuples() if (r.instance_id, int(r.seed), int(r.t)) not in done]
    print(f"judge: {len(U)} units, {len(todo)} to do; ledger ${led.total():.3f} cap ${cap}", flush=True)
    client = openai.AsyncOpenAI(api_key=A80.KEY.read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(32)

    async def go():
        t0 = time.time()
        for i in range(0, len(todo), 32):
            try:
                res = await asyncio.gather(*[judge_one(client, led, sem, r, specs[r.instance_id]) for r in todo[i:i + 32]])
            except Exception as e:  # noqa: BLE001
                print("stopping:", type(e).__name__, str(e)[:200], flush=True)
                return
            with open(JF, "a") as f:
                for x in res:
                    f.write(json.dumps(x) + "\n")
            print(f"judge {i + len(res)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def stage_analyze():
    from graybox.stats import within_auroc
    U, _ = G46C.units()
    key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
    own = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G46C.OWN))}
    jd = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(JF))}
    ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    m = np.array([k in own and k in jd for k in ks])
    U, ks = U[m].reset_index(drop=True), [k for k, x in zip(ks, m) if x]
    F = np.array([G46.forecast(own[k]) for k in ks])
    J = np.array([jd[k]["p"] if jd[k]["p"] is not None else np.nan for k in ks])
    Jf = np.where(np.isnan(J), np.nanmedian(J), J)
    y, cl = U.y.values, U.scenario_id.values
    st3 = (U.stratum + "|" + U.position.astype(str)).values
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(cl)
    B = {"own": [], "judge": [], "J1": []}
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        a, b = within_auroc(y[i], F[i], st3[i]), within_auroc(y[i], Jf[i], st3[i])
        B["own"].append(a); B["judge"].append(b); B["J1"].append(a - b)
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    W = lambda s: round(float(within_auroc(y, s, st3)), 4)  # noqa: E731
    res = {"units": int(len(y)), "violations": int(y.sum()), "judge_unparsed": int(np.isnan(J).sum()),
           "own_within_position": [W(F)] + ci(B["own"]), "gpt_judge_within_position": [W(Jf)] + ci(B["judge"]),
           "J1_own_minus_gpt_judge_position": [round(W(F) - W(Jf), 4)] + ci(B["J1"])}
    res["endpoints"] = {"J1_lower_gt_0": res["J1_own_minus_gpt_judge_position"][1] > 0}
    (ROOT / "results/t1/g55_gpt_judges_claude.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    {"judge": lambda: stage_judge(kw["cap"]), "analyze": stage_analyze}[kw.get("stage", "analyze")]()
