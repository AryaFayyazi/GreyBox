#!/usr/bin/env python
"""Parity of the browser MockEnv (site/src/assets/mockenv.js) with CAPE's Python MockEnv: replay every recorded Claude
episode (T24, both arms) on the bundled templates, call by call, in a real browser engine, and compare the observation
text and the unsafe/goal labels recorded by the Python environment.
PLAYWRIGHT_BROWSERS_PATH=... python web/test_mockenv_parity.py"""
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/arya/new_safety/cape/src")
TEMPLATES = {"cons_service_restart", "tp_campaign_test"}


async def main():
    from cape.envs.base import TOOLS
    names = [t["function"]["name"] for t in TOOLS]
    S = {json.loads(x)["instance_id"]: json.loads(x) for x in open("/home/arya/new_safety/cape/outputs/capebench/core24b/scenarios.jsonl")}
    eps = [json.loads(x) for x in open("/data/models/graybox_ws/outputs/t24/run.partial.jsonl")]
    eps = [e for e in eps if e["template"] in TEMPLATES]
    js = (ROOT / "site/src/assets/mockenv.js").read_text()
    n_calls = n_obs = n_lab = 0
    bad = []
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.set_content("<html><body></body></html>")
        await pg.add_script_tag(content=js)
        for e in eps:
            calls = [{"tool": t["tool"], "args": t.get("args") or {}} for t in e["turns"] if t.get("tool")]
            out = await pg.evaluate("""([spec, names, calls]) => { const env = new window.GBMockEnv.MockEnv(spec, names);
                return calls.map(c => { const [o, i] = env.step(c.tool, c.args); return {obs: o, unsafe: i.unsafe, goal: i.goal}; }); }""",
                                    [S[e["instance_id"]], names, calls])
            rec = [t for t in e["turns"] if t.get("tool")]
            for r, o in zip(rec, out):
                n_calls += 1
                n_obs += str(r.get("result")) == o["obs"]
                n_lab += bool(r.get("unsafe")) == o["unsafe"] and bool(r.get("goal")) == o["goal"]
                if str(r.get("result")) != o["obs"] or bool(r.get("unsafe")) != o["unsafe"] or bool(r.get("goal")) != o["goal"]:
                    bad.append({"ep": e["instance_id"], "tool": r["tool"], "args": r.get("args"), "py": [str(r.get("result"))[:120], r.get("unsafe"), r.get("goal")],
                                "js": [o["obs"][:120], o["unsafe"], o["goal"]]})
        await b.close()
    res = {"episodes": len(eps), "calls": n_calls, "observations_identical": n_obs, "labels_identical": n_lab, "mismatches": bad[:20]}
    print(json.dumps(res, indent=1))
    (ROOT / "results/t1/mockenv_parity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
