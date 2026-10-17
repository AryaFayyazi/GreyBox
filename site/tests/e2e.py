"""End-to-end check of the built site in headless Chromium: trainer flow on the example data, JS errors on every page,
screenshots (desktop 1280 and phone 390). Run with the playwright venv: python site/tests/e2e.py <dist> <shots_dir>"""
import functools, http.server, json, sys, threading, time
from pathlib import Path
from playwright.sync_api import sync_playwright
dist, shots = Path(sys.argv[1]), Path(sys.argv[2]); shots.mkdir(parents=True, exist_ok=True)
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(dist))
h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"
res = {"errors": {}}
with sync_playwright() as p:
    b = p.chromium.launch()
    for name in ("index", "train", "monitor", "results", "docs", "paper"):
        for w, tag in ((1280, "d"), (390, "m")):
            pg = b.new_page(viewport={"width": w, "height": 900})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(f"{base}/{name}.html"); pg.wait_for_timeout(1500)
            sw = pg.evaluate("document.documentElement.scrollWidth")
            res.setdefault("hscroll", {})[f"{name}_{tag}"] = sw > w + 1
            pg.screenshot(path=str(shots / f"{name}_{tag}.png"), full_page=False)
            if errs: res["errors"][f"{name}_{tag}"] = errs
            pg.close()
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.goto(f"{base}/train.html")
    pg.click("#exampleBtn"); pg.wait_for_function("document.getElementById('trainBtn').disabled === false", timeout=60000)
    res["dataInfo"] = pg.inner_text("#dataInfo")
    t0 = time.time(); pg.click("#trainBtn")
    pg.wait_for_function("!document.getElementById('results').hidden", timeout=900000)
    res["train_seconds"] = round(time.time() - t0, 1)
    res["stats"] = pg.inner_text("#stats"); res["details"] = pg.inner_text("#details")
    pg.click("#scoreBtn"); pg.wait_for_timeout(300); res["score"] = pg.inner_text("#scoreOut")
    pg.screenshot(path=str(shots / "train_results.png"), full_page=True)
    b.close()
srv.shutdown()
print(json.dumps(res, indent=1))
