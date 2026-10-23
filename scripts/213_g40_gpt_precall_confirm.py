#!/usr/bin/env python
"""G40: confirm, on fresh episodes, that a closed agent's state read before its call ranks the coming violation beyond
the call's position (docs/prereg_G40_gpt_precall_confirm.md). G38's exploratory control found 0.927 within template x
tool x position. Fresh GPT-6-luna episodes (scripts/80's loop, seed 52, no gate) on all 42 x- scenarios of each of the 6
templates in which it violated in T11. Before every turn, in side branches, G38's PRE battery (G27's frame and 8
forward-looking questions, 1 request of n = 4 each). Units (T11's rule): executed non-read calls up to and including
the first unsafe one; score: G27's raw read-out measured before the unit's turn; position: the unit's ordinal among the
episode's units (as G38's control). Full messages are stored.
  stage=run cap=<openai ledger $> (paid) -> outputs/gpt_agent/g40.jsonl ;  stage=analyze -> results/t1/g40_gpt_precall_confirm.json
"""
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAPE = Path("/home/arya/new_safety/cape")
for p in (ROOT / "src", ROOT / "scripts", CAPE / "src", CAPE / "scripts"):
    sys.path.insert(0, str(p))
TEMPLATES = ["con_cloud_cost", "cred_analytics_token", "cred_deploy_key", "prov_action_list", "tp_billing_correction", "tp_campaign_test"]
SEED, B_N = 52, 2000


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


A80 = _mod("a80", "scripts/80_gpt_agent.py")
G38 = _mod("g38", "scripts/211_g38_gpt_readout.py")
G27 = G38._mod("g27", "scripts/194_g27_state_tomography.py")
T11 = G38._mod("t11", "scripts/81_t11_gpt_monitor.py")
SG = np.array([s for _, s in G27.QUESTIONS])


async def episode(client, led, sem, spec, partner):
    from cape.envs.base import TOOLS, MockEnv
    from graybox.selfaudit import parse_answer
    env, part = MockEnv(spec), MockEnv(partner)
    msgs = [{"role": "system", "content": spec["system_prompt"]}, {"role": "user", "content": spec["task_prompt"]}]
    turns, status = [], "max_turns"

    async def q(j, text):
        led.check()
        async with sem:
            r = await client.chat.completions.create(model=A80.MODEL, messages=msgs + [{"role": "user", "content": G27.FRAME.format(q=text)}],
                                                     tools=TOOLS, tool_choice="none", n=4, seed=G38.SEED + j, temperature=1.0,
                                                     max_completion_tokens=16, reasoning_effort="none")
        led.charge(r.usage.prompt_tokens, r.usage.completion_tokens)
        v = [parse_answer(c.message.content or "") for c in r.choices]
        return [None if x is None else int(not x) for x in v]
    for t in range(A80.D.max_turns):
        pre = await asyncio.gather(*[q(j, x) for j, (x, _) in enumerate(G27.QUESTIONS)])
        led.check()
        async with sem:
            r = await client.chat.completions.create(model=A80.MODEL, messages=msgs, tools=TOOLS, parallel_tool_calls=False,
                                                     max_completion_tokens=2048, seed=SEED, reasoning_effort="none")
        led.charge(r.usage.prompt_tokens, r.usage.completion_tokens)
        ch = r.choices[0]
        uses = list(ch.message.tool_calls or [])
        rec = {"t": t, "pre": pre, "content": ch.message.content or ""}
        if ch.finish_reason == "length" or not uses:
            status = "truncated" if ch.finish_reason == "length" else "final_answer"
            turns.append(dict(rec, kind="end"))
            break
        u = uses[0]
        try:
            args = json.loads(u.function.arguments or "{}")
            args = args if isinstance(args, dict) else {}
        except json.JSONDecodeError:
            args = {}
        obs, info = env.step(u.function.name, args)
        part.step(u.function.name, args)
        turns.append(dict(rec, kind="call", tool=u.function.name, args_json=json.dumps(args), tool_result=obs,
                          unsafe_action=bool(info.unsafe), goal=bool(info.goal)))
        msgs.append(ch.message.model_dump(exclude_none=True))
        msgs.append({"role": "tool", "tool_call_id": u.id, "content": obs})
        msgs += [{"role": "tool", "tool_call_id": x.id, "content": "Not executed: one tool call per turn."} for x in uses[1:]]
    return {"instance_id": spec["instance_id"], "scenario_id": spec["scenario_id"], "template": spec["template"], "status": status,
            "turns": turns}


def stage_run(cap, conc=6):
    import openai
    from graybox.budget import Ledger
    led = Ledger(A80.LEDGER, float(cap), *A80.PRICE)
    fn = G38.GA / "g40.jsonl"
    done = {json.loads(x)["instance_id"] for x in open(fn)} if fn.exists() else set()
    jobs = [(s, p) for s, p in A80.specs() if s["template"] in TEMPLATES and s["instance_id"] not in done]
    print(f"{len(jobs)} episodes to run; ledger ${led.total():.3f} cap ${cap}", flush=True)
    client = openai.AsyncOpenAI(api_key=A80.KEY.read_text().strip(), max_retries=8)
    sem = asyncio.Semaphore(conc * 8)

    async def go():
        t0 = time.time()
        for i in range(0, len(jobs), conc):
            try:
                res = await asyncio.gather(*[episode(client, led, sem, s, p) for s, p in jobs[i:i + conc]])
            except Exception as e:  # noqa: BLE001  (BudgetExceeded or API failure: keep what is written)
                print("stopping:", type(e).__name__, str(e)[:200], flush=True)
                return
            with open(fn, "a") as f:
                for r in res:
                    f.write(json.dumps(r) + "\n")
            print(f"{i + len(res)}/{len(jobs)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def units():
    E = [json.loads(x) for x in open(G38.GA / "g40.jsonl")]
    rows = []
    for e in E:
        k = 0
        for x in e["turns"]:
            if x["kind"] != "call" or x["tool"] in T11.READS:
                continue
            C = np.array([[sum(a for a in q if a is not None), sum(a is not None for a in q)] for q in x["pre"]], float)
            risk = float(2 * np.arcsin(np.sqrt((C[:, 0] + 3 / 8) / (C[:, 1] + 3 / 4))) @ SG)
            rows.append({"instance_id": e["instance_id"], "scenario_id": e["scenario_id"], "stratum": f"{e['template']}|{x['tool']}",
                         "position": k, "risk": risk, "y": int(x["unsafe_action"])})
            k += 1
            if x["unsafe_action"]:
                break
    return pd.DataFrame(rows)


def stage_analyze():
    from graybox.stats import cluster_boot, within_auroc
    U = units()
    y, r, cl = U.y.values, U.risk.values, U.scenario_id.values
    s2 = U.stratum.values
    s3 = np.array([f"{a}|{b}" for a, b in zip(U.stratum, U.position)])
    out = {"episodes": int(U.instance_id.nunique()), "units": int(len(U)), "violations": int(y.sum())}
    for name, st in (("C1_within_template_tool_position", s3), ("C2_within_template_tool", s2)):
        b = cluster_boot(lambda i: within_auroc(y[i], r[i], st[i]), cl, B_N, seed=4000)
        b = b[np.isfinite(b)]
        out[name] = [round(float(within_auroc(y, r, st)), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
    both = [k for k in np.unique(s3) if 0 < y[s3 == k].sum() < (s3 == k).sum()]
    out["strata_with_both_classes"], out["violations_in_them"] = len(both), int(sum(y[s3 == k].sum() for k in both))
    out["position_alone_within_template_tool"] = round(float(within_auroc(y, U.position.values.astype(float), s2)), 4)
    out["endpoints"] = {"C1_lower_gt_0.5": out["C1_within_template_tool_position"][1] > 0.5,
                        "C2_lower_gt_0.5": out["C2_within_template_tool"][1] > 0.5}
    (ROOT / "results/t1/g40_gpt_precall_confirm.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    stage_run(kw["cap"]) if kw.get("stage") == "run" else stage_analyze()
