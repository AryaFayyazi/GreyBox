"""Screenshot a page with headless chromium (WebGL via SwiftShader). usage: shot.py URL OUT.png [width height] [dark]"""
import sys
from playwright.sync_api import sync_playwright
url, out = sys.argv[1], sys.argv[2]
w, h = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1440, 900)
dark = len(sys.argv) > 5 and sys.argv[5] == "dark"
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": w, "height": h}, color_scheme="dark" if dark else "light")
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    pg.goto(url, wait_until="networkidle", timeout=120000)
    pg.wait_for_timeout(25000)
    pg.screenshot(path=out)
    print("\n".join(logs[:20]))
    b.close()
