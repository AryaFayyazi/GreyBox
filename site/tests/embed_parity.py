#!/usr/bin/env python
"""Browser reader parity: the production worker (site/src/assets/embed-worker.js, transformers.js in headless Chromium)
vs graybox.embed.Encoder (Python, CPU, float32) on 40 real steps (T13b: GPT-6-luna on AgentDojo).
Reports per-view cosine similarity and the largest absolute difference of each feature.
python site/tests/embed_parity.py minilm|mpnet|qemb06   (cape venv; needs internet for the model files)"""
import importlib, json, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.embed import ENCODERS, Encoder  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "minilm"
m = importlib.import_module("100_t19_small_readers")
d = m.data("T13b").sample(40, random_state=1)
items = [{"ctx": c, "act": a} for c, a in zip(d.ctx, d.act)]
Xp = Encoder(name, "cpu", "/data/models/graybox_ws/hf", batch=1).encode(list(d.ctx), list(d.act))
tmp = Path(tempfile.mkdtemp())
shutil.copy(ROOT / "site/src/assets/embed-worker.js", tmp / "embed-worker.js")
(tmp / "items.json").write_text(json.dumps(items))
(tmp / "run.html").write_text("""<!doctype html><meta charset=utf-8><script type=module>
const items = await (await fetch('items.json')).json();
const w = new Worker('embed-worker.js', {type: 'module'});
w.onmessage = ev => { const m = ev.data;
  if (m.type === 'ready') { window.INFO = m; w.postMessage({cmd: 'embed', id: 1, items}); }
  else if (m.type === 'done') window.OUT = {X: Array.from(m.X), dim: m.dim, info: window.INFO};
  else if (m.type === 'error') window.OUT = {error: m.msg}; };
w.postMessage({cmd: 'load', encoder: '%s'});
</script>""" % name)
(tmp / "drive.py").write_text(f"""
import json, http.server, threading, functools, time
from playwright.sync_api import sync_playwright
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory="{tmp}"); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); t0 = time.time()
    pg.goto(f"http://127.0.0.1:{{srv.server_port}}/run.html"); pg.wait_for_function("window.OUT", timeout=1800000)
    o = pg.evaluate("window.OUT"); o["seconds"] = time.time() - t0
    open("{tmp}/out.json", "w").write(json.dumps(o)); b.close()
srv.shutdown()
""")
subprocess.run(["/data/models/graybox_ws/venv_pw/bin/python", str(tmp / "drive.py")], check=True,
               env={"PLAYWRIGHT_BROWSERS_PATH": "/data/models/graybox_ws/pw-browsers", "PATH": "/usr/bin:/bin"})
o = json.loads((tmp / "out.json").read_text())
assert "error" not in o, o
Xb = np.array(o["X"], np.float32).reshape(len(items), o["dim"])
assert Xb.shape == Xp.shape, (Xb.shape, Xp.shape)
nv = ENCODERS[name]["views"]
dv = Xp.shape[1] // nv
cos = [float((Xp[i, k * dv:(k + 1) * dv] * Xb[i, k * dv:(k + 1) * dv]).sum()) for i in range(len(items)) for k in range(nv)]
res = {"encoder": name, "steps": len(items), "views": nv, "browser": o["info"], "seconds_incl_download": round(o["seconds"], 1),
       "cosine_min": round(min(cos), 6), "cosine_mean": round(float(np.mean(cos)), 6), "max_abs_diff": float(np.abs(Xp - Xb).max())}
print(json.dumps(res, indent=1))
out = ROOT / "site/tests/embed_parity_result.json"
allr = json.loads(out.read_text()) if out.exists() else {}
allr[name] = res
out.write_text(json.dumps(allr, indent=1))
