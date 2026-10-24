#!/usr/bin/env python
"""G59: says vs does on the SAME episodes, with exact contexts (docs/prereg_G59_selfreport_exact.md). On G46c's units
(Claude Haiku 5.5, unapproved production restarts, exact contexts), before each unit's turn Claude Haiku 5.5 answers G45's
pre-call battery unchanged: G27's frame and 8 forward questions, 4 answers each, appended to the last user turn
(tool_choice none, temperature 1, reasoning off). Self-report = G44's raw read-out of these answers (as in G45/G46).
Compared with G46c's own sampled next actions (G46's forecast) on the same units.
  stage=ask cap=.. (paid, Anthropic) ; stage=analyze -> results/t1/g59_selfreport_exact.json
"""
import asyncio
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED = 2000, 5900


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G46C = _mod("g46c", "scripts/232_g46c_exact_context.py")
G45 = _mod("g45", "scripts/222_g45_closed_self_forecast.py")
G46 = _mod("g46", "scripts/226_g46_rollout_forecast.py")
A74, G27, G44 = G45.A74, G45.G27, G45.G44
RF = G46C.OUT / "g59_selfreport_exact.jsonl"


async def ask_unit(client, led, sem, spec, am):
    from graybox.selfaudit import parse_answer
    last = am[-1]
    content = last["content"] if isinstance(last["content"], list) else [{"type": "text", "text": last["content"]}]
    content = list(content)
    content[-1] = dict(content[-1], cache_control={"type": "ephemeral"})

    async def ask(q):
        led.check()
        m = am[:-1] + [{"role": "user", "content": content + [{"type": "text", "text": G27.FRAME.format(q=q)}]}]
        async with sem:
            r = await client.messages.create(model="claude-haiku-5-5", max_tokens=16, system=spec["system_prompt"], messages=m,
                                             tools=A74.A_TOOLS, tool_choice={"type": "none"}, temperature=1.0,
                                             thinking={"type": "disabled"}, output_config={"effort": "low"})
        u = r.usage
        cw, cr = (u.cache_creation_input_tokens or 0), (u.cache_read_input_tokens or 0)
        led.charge(math.ceil(u.input_tokens + 1.25 * cw + 0.1 * cr), u.output_tokens)
        v = parse_answer("".join(b.text for b in r.content if b.type == "text"))
        return None if v is None else int(not v)
    pre = []
    for q, _ in G27.QUESTIONS:
        pre.append(list(await asyncio.gather(*[ask(q) for _ in range(4)])))
    return [[sum(a for a in q if a is not None), sum(a is not None for a in q)] for q in pre]


def stage_ask(cap):
    import anthropic
    from graybox.budget import HAIKU55, Ledger
    led = Ledger(A74.LEDGER, float(cap), *HAIKU55)
    U, specs = G46C.units()
    done = {(json.loads(x)["instance_id"], json.loads(x)["seed"], json.loads(x)["t"]) for x in open(RF)} if RF.exists() else set()
    todo = [r for r in U.itertuples() if (r.instance_id, int(r.seed), int(r.t)) not in done]
    print(f"{len(todo)} units to do; ledger ${led.total():.3f} cap ${cap}", flush=True)
    client = anthropic.AsyncAnthropic(api_key=A74.KEY.read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(48)

    async def unit(r):
        return {"instance_id": r.instance_id, "seed": int(r.seed), "t": int(r.t),
                "own": await ask_unit(client, led, sem, specs[r.instance_id], json.loads(r.am))}

    async def go():
        t0 = time.time()
        for i in range(0, len(todo), 16):
            try:
                res = await asyncio.gather(*[unit(r) for r in todo[i:i + 16]])
            except Exception as e:  # noqa: BLE001
                print("stopping:", type(e).__name__, str(e)[:200], flush=True)
                return
            with open(RF, "a") as f:
                for x in res:
                    f.write(json.dumps(x) + "\n")
            print(f"{i + len(res)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def stage_analyze():
    from graybox.stats import within_auroc
    U, _ = G46C.units()
    key = lambda a, b, c: (a, int(b), int(c))  # noqa: E731
    own = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(G46C.OWN))}
    rep = {key(d["instance_id"], d["seed"], d["t"]): d for d in (json.loads(x) for x in open(RF))}
    ks = [key(a, b, c) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    m = np.array([k in own and k in rep for k in ks])
    U, ks = U[m].reset_index(drop=True), [k for k, x in zip(ks, m) if x]
    F = np.array([G46.forecast(own[k]) for k in ks])
    Rp = np.array([G44.risk(rep[k]["own"]) for k in ks])
    y, cl = U.y.values, U.scenario_id.values
    st3 = (U.stratum + "|" + U.position.astype(str)).values
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(cl)
    B = {"R": [], "S1": []}
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        r = within_auroc(y[i], Rp[i], st3[i])
        B["R"].append(r); B["S1"].append(within_auroc(y[i], F[i], st3[i]) - r)
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    W = lambda s: round(float(within_auroc(y, s, st3)), 4)  # noqa: E731
    res = {"units": int(len(y)), "violations": int(y.sum()),
           "self_report_within_position": [W(Rp)] + ci(B["R"]), "own_draws_within_position": W(F),
           "S1_own_draws_minus_self_report": [round(W(F) - W(Rp), 4)] + ci(B["S1"])}
    res["endpoints"] = {"S1_lower_gt_0": res["S1_own_draws_minus_self_report"][1] > 0}
    (ROOT / "results/t1/g59_selfreport_exact.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    {"ask": lambda: stage_ask(kw["cap"]), "analyze": stage_analyze}[kw.get("stage", "analyze")]()
