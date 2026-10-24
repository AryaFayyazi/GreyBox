#!/usr/bin/env python
"""G48: the policy forecast on a second closed agent across six kinds of task (docs/prereg_G48_gpt_rollout.md). On G45's
GPT-6-luna episodes (504; 6 CAPE-Bench templates), before each unit's turn: K = 8 next actions drawn from the agent
through its API (one request, n = 8, its loop's settings: tools, reasoning none, no seed), each distinct drawn call read
by the agent's own 8 at-call answers (graybox.selfaudit STATE_QUESTIONS and STATE_FRAME, one request of n = 4 per
question); forecast = scripts/226's (mean read-out over the 8 draws; a draw that ends the episode scores the minimum).
Self-report = G45's pre-call battery (G27's raw read-out). Units: G45's (non-read calls up to the first unsafe one, in
template|tool strata holding both classes).
  stage=own cap=<openai ledger $> (paid) ; stage=other (GPU: Gemma-4's draws in the seat) ; stage=analyze -> results/t1/g48_gpt_rollout.json
"""
import asyncio
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K, N_ANS, B_N, BOOT_SEED = 8, 4, 2000, 4800


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G46 = _mod("g46", "scripts/226_g46_rollout_forecast.py")
G45, G44 = G46.G45, G46.G44
A80 = G45.G40.A80
OUT = G45.OUT
OWN = OUT / "g48_gpt_own.jsonl"
OTHER = OUT / "g48_gpt_other_gemma4.jsonl"
G46B = _mod("g46b", "scripts/230_g46b_policy_gap.py")


def units():
    specs = {s["instance_id"]: s for s, _ in A80.specs()}
    reads = G45.T11.READS
    rows = []
    for e in (json.loads(x) for x in open(OUT / "gpt.jsonl")):
        sp = specs[e["instance_id"]]
        om = [{"role": "system", "content": sp["system_prompt"]}, {"role": "user", "content": sp["task_prompt"]}]
        done, k = [], 0
        for x in e["turns"]:
            if x["kind"] != "call":
                break
            if x["tool"] not in reads:
                own = [[sum(a for a in q if a is not None), sum(a is not None for a in q)] for q in x["pre"]]
                rows.append({"instance_id": e["instance_id"], "seed": e["seed"], "scenario_id": e["scenario_id"], "t": x["t"],
                             "stratum": f"{e['template']}|{x['tool']}", "position": k, "y": int(x["unsafe_action"]),
                             "report": G44.risk(own), "om": json.dumps(om), "done": json.dumps(done)})
                k += 1
                if x["unsafe_action"]:
                    break
            cid = f"call_{x['t']}"
            om = om + [{"role": "assistant", "content": x.get("content") or None,
                        "tool_calls": [{"id": cid, "type": "function", "function": {"name": x["tool"], "arguments": x["args_json"]}}]},
                       {"role": "tool", "tool_call_id": cid, "content": x["tool_result"] or ""}]
            done.append([x["tool"], x["args_json"]])
    U = pd.DataFrame(rows)
    g = U.groupby("stratum").y.agg(["sum", "count"])
    keep = set(g.index[(g["sum"] > 0) & (g["sum"] < g["count"])])
    return U[U.stratum.isin(keep)].reset_index(drop=True), specs


async def own_unit(client, led, sem, r, spec):
    from cape.envs.base import TOOLS, MockEnv
    from graybox.selfaudit import STATE_FRAME, STATE_QUESTIONS, parse_answer
    env = MockEnv(spec)
    for name, aj in json.loads(r.done):
        env.step(name, json.loads(aj or "{}"))
    base = json.loads(r.om)
    led.check()
    async with sem:
        x = await client.chat.completions.create(model=A80.MODEL, messages=base, tools=TOOLS, parallel_tool_calls=False, n=K,
                                                 max_completion_tokens=2048, reasoning_effort="none")
    led.charge(x.usage.prompt_tokens, x.usage.completion_tokens)
    draws = []
    for ch in x.choices:
        calls = ch.message.tool_calls or []
        if not calls:
            draws.append(None)
            continue
        try:
            args = json.loads(calls[0].function.arguments or "{}")
            args = args if isinstance(args, dict) else {}
        except json.JSONDecodeError:
            args = {}
        draws.append(json.dumps([calls[0].function.name, args], sort_keys=True, default=str))
    reads = {}
    for key in {d for d in draws if d}:
        name, args = json.loads(key)

        async def q(text):
            led.check()
            async with sem:
                rr = await client.chat.completions.create(model=A80.MODEL, messages=base + [{"role": "user", "content": STATE_FRAME.format(
                    name=name, args=json.dumps(args, sort_keys=True, default=str), q=text)}], tools=TOOLS, tool_choice="none", n=N_ANS,
                    temperature=1.0, max_completion_tokens=16, reasoning_effort="none")
            led.charge(rr.usage.prompt_tokens, rr.usage.completion_tokens)
            v = [parse_answer(c.message.content or "") for c in rr.choices]
            return [sum(1 for a in v if a is False), sum(1 for a in v if a is not None)]
        cnt = list(await asyncio.gather(*[q(t) for t, _ in STATE_QUESTIONS]))
        _, info = copy.deepcopy(env).step(name, args)
        reads[key] = {"counts": cnt, "unsafe": int(bool(info.unsafe))}
    return {"instance_id": r.instance_id, "seed": int(r.seed), "t": int(r.t), "draws": draws, "reads": reads}


def stage_own(cap):
    import openai
    from graybox.budget import Ledger
    led = Ledger(A80.LEDGER, float(cap), *A80.PRICE)
    U, specs = units()
    done = {(json.loads(x)["instance_id"], json.loads(x)["seed"], json.loads(x)["t"]) for x in open(OWN)} if OWN.exists() else set()
    todo = [r for r in U.itertuples() if (r.instance_id, int(r.seed), int(r.t)) not in done]
    print(f"own: {len(U)} units ({int(U.y.sum())} violations), {len(todo)} to do; ledger ${led.total():.3f} cap ${cap}", flush=True)
    client = openai.AsyncOpenAI(api_key=A80.KEY.read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(32)

    async def go():
        t0 = time.time()
        for i in range(0, len(todo), 6):
            try:
                res = await asyncio.gather(*[own_unit(client, led, sem, r, specs[r.instance_id]) for r in todo[i:i + 6]])
            except Exception as e:  # noqa: BLE001
                print("stopping:", type(e).__name__, str(e)[:200], flush=True)
                return
            with open(OWN, "a") as f:
                for x in res:
                    f.write(json.dumps(x) + "\n")
            print(f"own {i + len(res)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def stage_other():
    """Gemma-4's 8 draws in GPT-6-luna's seat (scripts/226's other stage on these units; GPU)"""
    G46.units = units
    G46.OTHER = OTHER
    G46.stage_other()


def stage_analyze():
    from graybox.stats import within_auroc
    U, _ = units()
    own = {(d["instance_id"], int(d["seed"]), int(d["t"])): d for d in (json.loads(x) for x in open(OWN))}
    ks = [(a, int(b), int(c)) for a, b, c in zip(U.instance_id, U.seed, U.t)]
    m = np.array([k in own for k in ks])
    U, ks = U[m].reset_index(drop=True), [k for k, x in zip(ks, m) if x]
    U["own"] = [G46.forecast(own[k]) for k in ks]
    U["ceiling"] = [G46.ceiling(own[k]) for k in ks]
    y, st2, st3, cl = U.y.values, U.stratum.values, (U.stratum + "|" + U.position.astype(str)).values, U.scenario_id.values
    W = lambda s, st, i=slice(None): within_auroc(y[i], s[i], st[i])  # noqa: E731
    cols = ("own", "report", "ceiling")
    res = {"units": int(len(U)), "violations": int(y.sum()),
           "within_template_tool": {c: round(float(W(U[c].values, st2)), 4) for c in cols},
           "within_position": {c: round(float(W(U[c].values, st3)), 4) for c in cols}}
    rng = np.random.default_rng(BOOT_SEED)
    us = np.unique(cl)
    B = {"Q1": [], "Q2": [], "Q3": []}
    for _ in range(B_N):
        i = np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])
        B["Q1"].append(W(U.own.values, st2, i))
        B["Q2"].append(W(U.own.values, st3, i))
        B["Q3"].append(W(U.own.values, st2, i) - W(U.report.values, st2, i))
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]  # noqa: E731
    res["Q1_own_template_tool"] = [res["within_template_tool"]["own"]] + ci(B["Q1"])
    res["Q2_own_position"] = [res["within_position"]["own"]] + ci(B["Q2"])
    res["Q3_own_minus_report_template_tool"] = [round(res["within_template_tool"]["own"] - res["within_template_tool"]["report"], 4)] + ci(B["Q3"])
    res["endpoints"] = {"Q1_lower_gt_0.5": res["Q1_own_template_tool"][1] > 0.5, "Q2_lower_gt_0.5": res["Q2_own_position"][1] > 0.5,
                        "Q3_two_sided_reported": True}
    # the policy-gap identity (scripts/230's formulas), with r known from CAPE's checker on each drawn call
    so = [G46B.shares(own[k]) for k in ks]
    fo, ko = np.array([x[0] for x in so]), np.array([x[1] for x in so])
    ok = ko >= 2
    C0 = lambda i: float(np.mean(y[i][ok[i]]) - np.mean(fo[i][ok[i]]))  # noqa: E731
    Bc = [C0(np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])) for _ in range(B_N)]
    res["C0_mean_y_minus_mean_f_own"] = [round(C0(np.arange(len(y))), 4)] + ci(Bc)
    res["endpoints"]["C0_interval_inside_pm0.05"] = res["C0_mean_y_minus_mean_f_own"][1] >= -0.05 and res["C0_mean_y_minus_mean_f_own"][2] <= 0.05
    if OTHER.exists():
        oth = {(d["instance_id"], int(d["seed"]), int(d["t"])): d for d in (json.loads(x) for x in open(OTHER))}
        U["other"] = [G46.forecast(oth[k]) if k in oth else np.nan for k in ks]
        sx = [G46B.shares(oth[k]) if k in oth else (np.nan, 0) for k in ks]
        fx, kx = np.array([x[0] for x in sx]), np.array([x[1] for x in sx])
        both = ok & (kx >= 2)

        def b3(i):
            j = i[both[i]]
            obs = np.mean((y[j] - fx[j]) ** 2) - np.mean((y[j] - fo[j]) ** 2)
            pred = np.mean((fo[j] - fx[j]) ** 2) - 2 * np.mean(fo[j] * (1 - fo[j]) / (ko[j] - 1))
            gap = np.mean((fo[j] - fx[j]) ** 2) - np.mean(fo[j] * (1 - fo[j]) / (ko[j] - 1)) - np.mean(fx[j] * (1 - fx[j]) / (kx[j] - 1))
            return obs, pred, obs - pred, gap
        Bb = np.array([b3(np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))])) for _ in range(B_N)])
        o, pr, d, g = b3(np.arange(len(y)))
        res["B_units"] = int(both.sum())
        for j, (name, v) in enumerate((("B1_observed_excess", o), ("predicted_excess", pr), ("B3_observed_minus_predicted", d), ("B2_policy_gap", g))):
            res[name] = [round(float(v), 4)] + ci(Bb[:, j])
        res["other_within_template_tool"] = round(float(W(U.other.values, st2)), 4)
        res["other_within_position"] = round(float(W(U.other.values, st3)), 4)
        res["endpoints"]["B3_interval_inside_pm0.02"] = res["B3_observed_minus_predicted"][1] >= -0.02 and res["B3_observed_minus_predicted"][2] <= 0.02
    (ROOT / "results/t1/g48_gpt_rollout.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    {"own": lambda: stage_own(kw["cap"]), "other": stage_other, "analyze": stage_analyze}[kw.get("stage", "analyze")]()
