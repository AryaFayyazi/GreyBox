#!/usr/bin/env python
"""G37 with the recorded deviation (docs/deviation_G37_2026-10-22.md): pass R (R4 on all writes) then pass D (the 3 other
wordings, 1 answer each, in G37's order until the cap). Imports the locked script (scripts/210) unchanged.
  stage=R cap=59.55 | stage=D cap=59.55 | stage=analyze
"""
import asyncio
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
s = importlib.util.spec_from_file_location("g37", ROOT / "scripts/210_g37_closed_precall.py")
G37 = importlib.util.module_from_spec(s)
s.loader.exec_module(G37)
G18 = G37.G18
FR, FD = G37.OUT / "g37_conf_R.jsonl", G37.OUT / "g37_conf_D.jsonl"


def run(stage, cap, wave=8, conc=16):
    import anthropic
    from graybox.budget import HAIKU55, BudgetExceeded, Ledger
    p101 = G37._mod("gb101", "scripts/101_claude_openai_proxy.py")
    led = Ledger(G37.LEDGER, float(cap), *HAIKU55)
    tools = p101.to_tools(json.loads((G37.OUT / "g18_context.json").read_text())["tools"])
    U = pd.read_json(G37.OUT / "g18_units_conf.jsonl", orient="records", lines=True, dtype={"task_id": str})
    U = G18.sample(U, G18.CFG["n_units"], G18.CFG["seed"])
    U = U.assign(k=U.groupby(["sim_id", "msg_idx"]).cumcount())
    fn = FR if stage == "R" else FD
    done = {(json.loads(x)["sim_id"], json.loads(x)["msg_idx"], json.loads(x)["k"]) for x in open(fn)} if fn.exists() else set()
    todo = [r for r in U.itertuples() if (r.sim_id, r.msg_idx, r.k) not in done]
    print(f"pass {stage}: {len(todo)} to do; ledger ${led.total():.3f} cap ${cap}", flush=True)
    cl = anthropic.AsyncAnthropic(api_key=Path("/data/models/graybox_ws/.secrets/anthropic_key").read_text().strip(), max_retries=8)

    async def ask(sem, system, msgs):
        led.check()
        async with sem:
            x = await cl.messages.create(model="claude-haiku-5-5", max_tokens=16, system=system, messages=msgs, tools=tools,
                                         tool_choice={"type": "none"}, temperature=1.0, thinking={"type": "disabled"},
                                         output_config={"effort": "low"})
        u = x.usage
        cw, cr = (u.cache_creation_input_tokens or 0), (u.cache_read_input_tokens or 0)
        led.charge(math.ceil(u.input_tokens + 1.25 * cw + 0.1 * cr), u.output_tokens)
        v = G18.parse("".join(b.text for b in x.content if b.type == "text"))
        return None if v is None else int(not v)

    async def battery(sem, r):
        system, base = p101.to_anthropic(json.loads(r.messages))
        system = system + G18.SYSTEM_NOTE
        assert base[-1]["role"] == "user"
        last = dict(base[-1]["content"][-1], cache_control={"type": "ephemeral"})
        base = base[:-1] + [{"role": "user", "content": base[-1]["content"][:-1] + [last]}]

        def with_q(q):
            return base[:-1] + [{"role": "user", "content": base[-1]["content"] + [{"type": "text", "text": G37.FRAME.format(q=q)}]}]
        if stage == "R":
            first = await ask(sem, system, with_q(G37.VERSIONS[0][0][0]))
            jobs = [j for j in range(8) for _ in range(G37.N_R)][1:]
            res = await asyncio.gather(*[ask(sem, system, with_q(G37.VERSIONS[j][0][0])) for j in jobs])
            R = [[] for _ in range(8)]
            R[0].append(first)
            for j, a in zip(jobs, res):
                R[j].append(a)
            return {"R": R}
        first = await ask(sem, system, with_q(G37.VERSIONS[0][1][0]))
        jobs = [(j, v) for j in range(8) for v in (1, 2, 3)][1:]
        res = await asyncio.gather(*[ask(sem, system, with_q(G37.VERSIONS[j][v][0])) for j, v in jobs])
        D = [[None, None, None] for _ in range(8)]
        D[0][0] = first
        for (j, v), a in zip(jobs, res):
            D[j][v - 1] = a
        return {"D_other": D}

    async def go():
        sem, t0 = asyncio.Semaphore(conc), time.time()
        for i in range(0, len(todo), wave):
            batch = todo[i:i + wave]
            try:
                res = await asyncio.gather(*[battery(sem, r) for r in batch])
            except BudgetExceeded:
                print("ledger cap reached; stopping (this wave is not written)", flush=True)
                return
            with open(fn, "a") as f:
                for r, o in zip(batch, res):
                    f.write(json.dumps({"sim_id": r.sim_id, "msg_idx": int(r.msg_idx), "k": int(r.k), "task_id": r.task_id,
                                        "tool": r.tool, "y": int(r.y), **o}) + "\n")
            print(f"{i + len(batch)}/{len(todo)} {time.time() - t0:.0f}s ledger ${led.total():.3f}", flush=True)
    asyncio.run(go())


def analyze():
    """G37's analysis (scripts/210 stage_analyze, same code) on pass R for P1 and on the writes with both passes for P2"""
    key = ["sim_id", "msg_idx", "k"]
    R = pd.read_json(FR, orient="records", lines=True, dtype={"task_id": str})
    D = pd.read_json(FD, orient="records", lines=True, dtype={"task_id": str}) if FD.exists() else pd.DataFrame(columns=key + ["D_other"])
    J = R.merge(D[key + ["D_other"]], on=key, how="left")
    both = J.D_other.notna().values
    J["answers"] = [{"R": r, "D_other": d if isinstance(d, list) else [[None] * 3] * 8} for r, d in zip(J.R, J.D_other)]
    tmp = G37.OUT / "g37_conf.jsonl"
    assert not tmp.exists(), "the locked analysis reads g37_conf.jsonl; it must be written only here"
    out = {}
    for name, sub in (("all_R", J), ("both_arms", J[both])):
        sub[["sim_id", "msg_idx", "k", "task_id", "tool", "y", "answers"]].to_json(tmp, orient="records", lines=True)
        G37.stage_analyze()
        out[name] = json.loads((ROOT / "results/t1/g37_closed_precall.json").read_text())
        tmp.unlink()
    res = {"deviation": "docs/deviation_G37_2026-10-22.md", "writes_R": int(len(J)), "writes_both_arms": int(both.sum()),
           "P1_R4_all_writes": out["all_R"]["P1_R4"], "precall_R4_minus_at_call_all_writes": out["all_R"]["precall_R4_minus_at_call"],
           "unanimity_all_writes": out["all_R"]["unanimity"], "within_tool_auroc_all_writes": out["all_R"]["within_tool_auroc"],
           "P2_D4_minus_R4_both_arms": out["both_arms"]["P2_D4_minus_R4"], "D4_both_arms": out["both_arms"]["D4"],
           "R4_both_arms": out["both_arms"]["P1_R4"], "violations_both_arms": out["both_arms"]["violations"],
           "violations_all": out["all_R"]["violations"]}
    res["endpoints"] = {"P1_lower_gt_0.5": res["P1_R4_all_writes"][1] > 0.5, "P2_lower_gt_0": res["P2_D4_minus_R4_both_arms"][1] > 0}
    (ROOT / "results/t1/g37_closed_precall.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    if kw.get("stage") in ("R", "D"):
        run(kw["stage"], kw["cap"])
    else:
        analyze()
