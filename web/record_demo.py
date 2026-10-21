#!/usr/bin/env python
"""Record the Self-policing demo from the real page (headless Chromium, 1280x720): real hidden states in 3D, a LIVE
audit call to an open model (served locally, OpenAI-compatible), real recorded G10 episodes, the law calculator.
Captions are injected as an overlay and say what each step shows. Outputs (in OUT): demo.webm, shots/*.png.
  PLAYWRIGHT_BROWSERS_PATH=... python web/record_demo.py <site_url> <out_dir> <model_base_url>
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL, OUT, BASE = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
CAP_JS = """(t) => { let c = document.getElementById('demoCap'); if (!c) { c = document.createElement('div'); c.id = 'demoCap';
  c.style.cssText = 'position:fixed;left:50%;bottom:26px;transform:translateX(-50%);max-width:1080px;background:rgba(27,33,38,.93);color:#fff;' +
  'font:600 19px/1.4 Archivo,system-ui,sans-serif;padding:12px 20px;border-radius:10px;z-index:99999;box-shadow:0 6px 24px rgba(0,0,0,.25);text-align:center';
  document.body.appendChild(c); } c.innerHTML = t; c.style.display = t ? 'block' : 'none'; }"""


async def cap(pg, text, hold=2.6):
    await pg.evaluate(CAP_JS, text)
    await pg.wait_for_timeout(int(hold * 1000))


async def scroll_to(pg, sel, offset=-80):
    await pg.evaluate("([s,o]) => { const e = document.querySelector(s); window.scrollTo({top: e.getBoundingClientRect().top + window.scrollY + o, behavior: 'smooth'}); }", [sel, offset])
    await pg.wait_for_timeout(900)


async def rotate(pg, sel, dx=420, steps=60, dy=0):
    b = await (await pg.query_selector(sel)).bounding_box()
    x0, y0 = b["x"] + b["width"] * 0.3, b["y"] + b["height"] * 0.5
    await pg.mouse.move(x0, y0)
    await pg.mouse.down()
    for i in range(1, steps + 1):
        await pg.mouse.move(x0 + dx * i / steps, y0 + dy * i / steps)
        await pg.wait_for_timeout(28)
    await pg.mouse.up()
    await pg.mouse.move(b["x"] + b["width"] + 30, y0)                 # move away: no tooltip in the frame


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shots").mkdir(exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        ctx = await b.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1,
                                  record_video_dir=str(OUT / "raw"), record_video_size={"width": 1280, "height": 720})
        pg = await ctx.new_page()
        await pg.goto(URL)
        await pg.wait_for_timeout(2500)
        await cap(pg, "Self-policing: before a tool call runs, the AI agent is asked <i>through its own API</i> whether the call follows its policy.<br>No second model, no training, no server.", 4.2)
        await pg.screenshot(path=str(OUT / "shots/01_overview.png"))
        await scroll_to(pg, "#headline", -150)
        await cap(pg, "Preregistered results on held-out scenarios: an open agent (Gemma-4) and a closed one (Claude Haiku 5.5).", 3.6)
        await scroll_to(pg, "#cvA", -60)
        await cap(pg, "Real hidden states of the agent at the moment it decides (5,492 steps).<br>Orange = the call violates the policy; rings = the agent's own audit flagged it.", 1.2)
        await rotate(pg, "#cvA", 380)
        await pg.wait_for_timeout(600)
        await pg.screenshot(path=str(OUT / "shots/02_agent_state_3d.png"))
        await cap(pg, "Violations lie along one direction inside the agent (AUROC 0.976), and its own spoken audit flags almost exactly those (0.950).", 3.4)
        await scroll_to(pg, "#cvB1", -90)
        await cap(pg, "Left: a model's hidden states. Right: the same contexts placed from its yes/no answers alone. The panels turn together.", 1.2)
        await rotate(pg, "#cvB1", 360)
        await pg.screenshot(path=str(OUT / "shots/03_hidden_vs_answers_3d.png"))
        await cap(pg, "The kinds of context separate in both: answers reveal the geometry of hidden states (Spearman 0.665; preregistered).", 3.2)
        await scroll_to(pg, "#loadEx", -260)
        await cap(pg, "Now a live audit. Example: the agent proposes restarting a production service without the approval its policy requires.", 2.6)
        await pg.click("#loadEx")
        await pg.select_option("#prov", "openai")
        await pg.fill("#base", BASE)
        await pg.fill("#model", "actor")
        await pg.fill("#key", "none")
        await pg.select_option("#prompt", "checklist")
        await pg.fill("#n", "1")
        await scroll_to(pg, "#run", -380)
        await cap(pg, "LIVE call: an open model (Gemma-4, served on our machine) is asked to check its policy rule by rule.", 0.8)
        await pg.click("#run")
        await pg.wait_for_function("['HOLD','Let it run','failed'].some(t => document.querySelector('#out').innerText.includes(t))", timeout=240000)
        verdict = (await pg.inner_text("#out")).split("\n")[0]
        await scroll_to(pg, "#out", -200)
        await cap(pg, f"Its own answer: <b>{verdict.split('  ')[0]}</b>. The call is held for a human before it runs.", 3.6)
        await pg.screenshot(path=str(OUT / "shots/04_live_audit.png"))
        await scroll_to(pg, "#ep", -120)
        await cap(pg, "Recorded episodes from the preregistered test (G10): Claude Haiku 5.5 without and with self-policing.", 2.4)
        await scroll_to(pg, "#gated", -160)
        await cap(pg, "Without: the restart runs (a violation). With: Claude's own audit flags it, a reviewer blocks it, and Claude asks for permission instead.", 4.0)
        await pg.screenshot(path=str(OUT / "shots/05_replay.png"))
        await pg.select_option("#ep", "3")
        await pg.wait_for_timeout(500)
        await cap(pg, "Its limit: a 'test' send to a mailing list whose thousands of members only show after sending. No judge of the conversation can see that, as the theory predicts.", 4.2)
        await scroll_to(pg, "#lawTable", -260)
        await cap(pg, "Predict the effect before deploying: share of violations prevented = Σ (share of each kind) × (how often the agent flags that kind).", 3.6)
        await pg.screenshot(path=str(OUT / "shots/06_law.png"))
        await pg.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
        await pg.wait_for_timeout(900)
        await cap(pg, "Try it: aryafayyazi.github.io/GreyBox/selfpolice.html &nbsp;·&nbsp; pip install greybox-monitor (graybox.selfaudit)", 3.8)
        await cap(pg, "", 0.3)
        await ctx.close()
        await b.close()
    raw = sorted((OUT / "raw").glob("*.webm"))[-1]
    raw.rename(OUT / "demo.webm")
    print("recorded", OUT / "demo.webm", "verdict:", verdict)


if __name__ == "__main__":
    asyncio.run(main())
