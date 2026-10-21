#!/usr/bin/env python
"""Record the narrated explainer from the real Self-policing page, in sync with the narration.

The screen drives the timing: each segment's narration starts exactly when its on-screen step starts (the start times
are recorded here and used to place each WAV in the audio track), and every segment lasts at least as long as its
narration. Steps use the real page: real hidden states in 3D, a LIVE audit call to an open model on this machine,
recorded preregistered episodes. Output: <out>/screen.webm and <out>/timeline.json.
  python web/record_explainer.py <site_url> <out_dir> <model_base_url> web/explainer_script.json
"""
import asyncio
import json
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

URL, OUT, BASE, SCRIPT = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
KEY = Path("/data/models/graybox_ws/.secrets/anthropic_key").read_text().strip()   # set in the page by script; never shown or saved
S = json.loads(SCRIPT.read_text())
DUR = json.loads((OUT / "durations.json").read_text())
CAPS = {s["id"]: s["caption"] for s in S["segments"]}
CAP_JS = """(t) => { let c = document.getElementById('demoCap'); if (!c) { c = document.createElement('div'); c.id = 'demoCap';
  c.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);max-width:1120px;width:max-content;background:rgba(27,33,38,.93);' +
  'color:#fff;font:600 20px/1.35 Archivo,system-ui,sans-serif;padding:12px 22px;border-radius:10px;z-index:99999;box-shadow:0 6px 24px rgba(0,0,0,.25);text-align:center';
  document.body.appendChild(c); } c.innerHTML = t; c.style.display = t ? 'block' : 'none'; }"""


async def scroll_to(pg, sel, offset=-80):
    await pg.evaluate("([s,o]) => { const e = document.querySelector(s); window.scrollTo({top: e.getBoundingClientRect().top + window.scrollY + o, behavior: 'smooth'}); }", [sel, offset])
    await pg.wait_for_timeout(1000)


async def slow_rotate(pg, sel, seconds, dx=520):
    b = await (await pg.query_selector(sel)).bounding_box()
    x0, y0 = b["x"] + b["width"] * 0.25, b["y"] + b["height"] * 0.5
    steps = max(10, int(seconds * 20))
    await pg.mouse.move(x0, y0)
    await pg.mouse.down()
    for i in range(1, steps + 1):
        await pg.mouse.move(x0 + dx * i / steps, y0)
        await pg.wait_for_timeout(50)
    await pg.mouse.up()
    await pg.mouse.move(b["x"] + b["width"] + 40, b["y"] - 40)            # away from the dots: no tooltip


async def main():
    timeline = []
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        t_rec = time.monotonic()
        t_rec_ref = [t_rec]
        ctx = await b.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=str(OUT / "raw"),
                                  record_video_size={"width": 1280, "height": 720})
        pg = await ctx.new_page()
        await pg.goto(URL)
        await pg.wait_for_timeout(3000)

        async def segment(sid, actions):
            t0 = time.monotonic()
            timeline.append({"id": sid, "start": round(t0 - t_rec, 3)})
            await pg.evaluate(CAP_JS, CAPS[sid])
            await actions()
            left = DUR[sid] - (time.monotonic() - t0)
            if left > 0:
                await pg.wait_for_timeout(int(left * 1000))
            timeline[-1]["end"] = round(time.monotonic() - t_rec, 3)

        async def noop():
            pass

        async def a_results():
            await scroll_to(pg, "#headline", -170)

        async def a_agent():
            await scroll_to(pg, "#cvA", -70)
            await pg.wait_for_timeout(1500)
            await slow_rotate(pg, "#cvA", DUR["agent3d"] * 0.6)

        async def a_geom():
            await scroll_to(pg, "#cvB1", -100)
            await pg.wait_for_timeout(1200)
            await slow_rotate(pg, "#cvB1", DUR["geometry3d"] * 0.6, dx=480)

        async def a_live():
            await scroll_to(pg, "#loadEx", -250)
            await pg.click("#loadEx")
            await pg.wait_for_timeout(800)
            await pg.select_option("#prov", "openai")
            await pg.fill("#base", "")
            await pg.type("#base", BASE, delay=35)
            await pg.fill("#model", "actor")
            await pg.fill("#key", "none")
            await pg.select_option("#prompt", "checklist")
            await pg.fill("#n", "1")
            if await pg.query_selector("#obs"):
                await pg.fill("#n", "2")
                await pg.check("#obs")                                   # the live observable state, 8 questions x 2 answers
            await scroll_to(pg, "#run", -380)
            await pg.click("#run")

        async def a_verdict():
            await pg.wait_for_function("['HOLD','Let it run','failed'].some(t => document.querySelector('#out').innerText.includes(t))", timeout=300000)
            await scroll_to(pg, "#out", -170)
            timeline[-1]["verdict"] = (await pg.inner_text("#out")).split("\n")[0]

        async def a_observable():
            await scroll_to(pg, "#cvO", -150)
            await pg.wait_for_function("document.querySelector('#obsRead').innerText.length > 10", timeout=600000)
            timeline[-1]["observable"] = await pg.inner_text("#obsRead")

        async def a_session_intro():
            await scroll_to(pg, "#lsScen", -170)
            await pg.evaluate("k => { document.getElementById('prov').value = 'anthropic'; document.getElementById('model').value = 'claude-haiku-5-5';"
                              " document.getElementById('key').value = k; }", KEY)
            await pg.evaluate("() => { window.GB_AUTO_DECIDE = (kind, truth) => kind === 'gate' ? (truth ? 'block' : 'allow') : 'allow'; window.GB_AUTO_DELAY = 1500; window.GB_AUTO_HOLD = true; }")
            await pg.select_option("#lsScen", "0")
            await pg.check("#lsGate")
            await pg.wait_for_timeout(1500)
            await pg.click("#lsStart")

        async def a_session_watch():
            await scroll_to(pg, "#lsLog", -90)

        async def a_session_warn():
            try:
                await pg.wait_for_function("!document.getElementById('lsWarn').hidden || document.getElementById('lsStatus').innerText.startsWith('Done')", timeout=150000)
            except Exception:  # noqa: BLE001
                pass
            shown = await pg.evaluate("() => !document.getElementById('lsWarn').hidden")
            timeline[-1]["warning_shown"] = shown
            if shown:                        # the first flagged call is held on screen while the narration explains it
                await scroll_to(pg, "#lsWarn", -40)
                await pg.wait_for_timeout(int(max(1.0, DUR["session_warn"] * 0.65 - 1.0) * 1000))
                await pg.evaluate("() => window.GB_AUTO_RELEASE && window.GB_AUTO_RELEASE()")
                timeline[-1]["released_at"] = round(time.monotonic() - t_rec_ref[0], 3)
            left = DUR["session_warn"] - (time.monotonic() - timeline[-1]["start"] - t_rec_ref[0])
            if left > 0:
                await pg.wait_for_timeout(int(left * 1000))      # let the warning narration finish first
            await pg.evaluate(CAP_JS, "Live: the agent continues after the reviewer's decision.")
            await scroll_to(pg, "#lsLog", -90)
            await pg.wait_for_function("['Done','Stopped'].some(t => document.getElementById('lsStatus').innerText.startsWith(t))", timeout=300000)
            await pg.wait_for_timeout(800)

        async def a_session_end():
            await pg.wait_for_function("['Done','Stopped'].some(t => document.getElementById('lsStatus').innerText.startsWith(t))", timeout=300000)
            await scroll_to(pg, "#lsStatus", -520)
            timeline[-1]["status"] = await pg.inner_text("#lsStatus")
            timeline[-1]["flagged_steps"] = (await pg.inner_text("#lsLog")).count("FLAGGED")

        async def a_replay():
            await scroll_to(pg, "#ep", -110)
            await pg.wait_for_timeout(3500)
            await scroll_to(pg, "#gated", -150)

        async def a_limit():
            await scroll_to(pg, "#ep", -110)
            await pg.select_option("#ep", "3")
            await pg.wait_for_timeout(1500)
            await scroll_to(pg, "#gated", -150)

        async def a_law():
            await scroll_to(pg, "#lawTable", -260)

        async def a_use():
            await pg.evaluate("window.scrollTo({top:0,behavior:'smooth'})")

        for sid, act in (("intro", noop), ("results", a_results), ("agent3d", a_agent), ("geometry3d", a_geom), ("session_intro", a_session_intro),
                         ("session_watch", a_session_watch), ("session_warn", a_session_warn), ("session_end", a_session_end),
                         ("replay", a_replay), ("limit", a_limit), ("law", a_law), ("use", a_use)):
            await segment(sid, act)
        await pg.wait_for_timeout(800)
        await ctx.close()
        await b.close()
    raw = sorted((OUT / "raw").glob("*.webm"))[-1]
    raw.rename(OUT / "screen.webm")
    (OUT / "timeline.json").write_text(json.dumps(timeline, indent=1))
    print(json.dumps(timeline, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
