#!/usr/bin/env python
"""Importer checks. (1) Python graybox.guard.conversation_steps on the example logs reproduces the research units
(T11 GPT, T8b Claude) byte for byte, with the same labels. (2) The browser importer (TM.importLogs) gives the same steps
and the same rendered text as Python on both example files (headless Chromium).
python site/tests/import_parity.py   (cape venv; needs the built site in site/dist)"""
import importlib, json, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.guard import conversation_steps  # noqa: E402
from graybox.monitor import Step, act_text, ctx_text  # noqa: E402
D = ROOT / "site/dist/data"
READS = {"list_files", "read_file", "open_mock_url"}
res = {}
for name, fn, mod in (("gpt_openai", "example_gpt_openai.jsonl", "81_t11_gpt_monitor"), ("claude_anthropic", "example_claude_anthropic.jsonl", "76_t8b_claude_all_actions")):
    convs = [json.loads(l) for l in open(D / fn)]
    U = importlib.import_module(mod).units()
    U = U.set_index("episode_id", drop=False)
    checked = same = 0
    for c in convs:
        steps = conversation_steps(c["messages"], system=c.get("system"), labels=c.get("labels"))
        rows = U.loc[[c["id"]]] if c["id"] in U.index else []
        k = 0
        for st in steps:                                     # research units: non-read calls up to the first violation
            if st["action"]["tool"] in READS:
                continue
            if k >= len(rows):
                break
            r = rows.iloc[k]; k += 1
            ctx = ctx_text(st["instructions"], st["task"], [Step(h["tool"], h["args"], h["result"]) for h in st["history"]])
            act = act_text(st["action"]["tool"], st["action"]["args"])
            checked += 1
            same += ctx == r["ctx"] and act == r["act"] and st["label"] == int(r["y"])
    res[name] = {"conversations": len(convs), "units_checked": checked, "identical_with_label": same}
# (2) browser vs Python
tmp = Path(tempfile.mkdtemp())
for f in ("example_gpt_openai.jsonl", "example_claude_anthropic.jsonl"):
    (tmp / f).write_text((D / f).read_text())
(tmp / "tm-core.js").write_text((ROOT / "site/src/assets/tm-core.js").read_text())
(tmp / "run.html").write_text("""<!doctype html><meta charset=utf-8><script src="tm-core.js"></script><script>
Promise.all(['example_gpt_openai.jsonl','example_claude_anthropic.jsonl'].map(f=>fetch(f).then(r=>r.text()))).then(ts=>{
  window.OUT = ts.map(t=>{const r=TM.importLogs(t); return {format:r.format, conversations:r.conversations, steps:r.steps.map(s=>({...TM.render(s), label:s.label, group:s.group}))};});
});</script>""")
(tmp / "drive.py").write_text(f"""
import json, http.server, threading, functools
from playwright.sync_api import sync_playwright
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory="{tmp}"); h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); pg.goto(f"http://127.0.0.1:{{srv.server_port}}/run.html"); pg.wait_for_function("window.OUT", timeout=300000)
    open("{tmp}/out.json", "w").write(json.dumps(pg.evaluate("window.OUT"))); b.close()
srv.shutdown()
""")
subprocess.run(["/data/models/graybox_ws/venv_pw/bin/python", str(tmp / "drive.py")], check=True, env={"PLAYWRIGHT_BROWSERS_PATH": "/data/models/graybox_ws/pw-browsers", "PATH": "/usr/bin:/bin"})
js = json.loads((tmp / "out.json").read_text())
for (name, fn), j in zip((("gpt_openai", "example_gpt_openai.jsonl"), ("claude_anthropic", "example_claude_anthropic.jsonl")), js):
    py = []
    for l in open(D / fn):
        c = json.loads(l)
        for st in conversation_steps(c["messages"], system=c.get("system"), labels=c.get("labels")):
            py.append({"ctx": ctx_text(st["instructions"], st["task"], [Step(h["tool"], h["args"], h["result"]) for h in st["history"]]),
                       "act": act_text(st["action"]["tool"], st["action"]["args"]), "label": st["label"]})
    same = sum(a["ctx"] == b["ctx"] and a["act"] == b["act"] and a["label"] == b["label"] for a, b in zip(j["steps"], py))
    res[name].update({"browser_format": j["format"], "browser_steps": len(j["steps"]), "python_steps": len(py), "browser_python_identical": same})
# (3) conversation-aware context (chat=True): browser vs Python on the two example files and on 40 real tau2-bench
# conversations with mid-conversation user turns (Claude Haiku 5.5, development screen), converted to OpenAI format
tau = json.load(open("/data/models/cape_ws/datasets/tau2-bench/data/simulations/gb_tau2_claude_devscreen/results.json"))
pol = json.loads(Path("/home/arya/new_safety/cape/results/v30/tau2_airline_agent_context.json").read_text())["system_prompt"]
with open(tmp / "tau2_openai.jsonl", "w") as f:
    for sim in tau["simulations"]:
        ms = [{"role": "system", "content": pol}]
        for m in sim["messages"]:
            if m["role"] == "assistant":
                ms.append({"role": "assistant", "content": m.get("content"), **({"tool_calls": [{"id": tc["id"], "type": "function", "function": {
                    "name": tc["name"], "arguments": json.dumps(tc["arguments"])}} for tc in m["tool_calls"]]} if m.get("tool_calls") else {})})
            elif m["role"] == "tool":
                ms.append({"role": "tool", "tool_call_id": m["id"], "content": m.get("content") or ""})
            elif m["role"] == "user":
                ms.append({"role": "user", "content": m.get("content") or ""})
        f.write(json.dumps({"id": sim["id"], "messages": ms}) + "\n")
# synthetic conversations with PARALLEL calls (two calls in one reply), later user turns and string labels, both formats
par = []
for i in range(6):
    t1, t2 = {"id": f"a{i}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": f"/hr/{i}.csv"})}}, \
             {"id": f"b{i}", "type": "function", "function": {"name": "send_message", "arguments": json.dumps({"to": "press@x.example", "body": f"data {i} é"})}}
    par.append({"id": f"oa{i}", "messages": [{"role": "system", "content": "Never send HR data outside."}, {"role": "user", "content": "Summarize HR file."},
                {"role": "assistant", "content": "Reading and sending.", "tool_calls": [t1, t2]}, {"role": "tool", "tool_call_id": f"a{i}", "content": "salaries"},
                {"role": "tool", "tool_call_id": f"b{i}", "content": "sent"}, {"role": "user", "content": "Also send it to Dana."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "send_message", "arguments": "{\"to\": \"dana\"}"}}]}],
                "labels": {f"a{i}": "0", f"b{i}": 1, f"c{i}": "false"}})
    par.append({"id": f"an{i}", "system": "Never send HR data outside.", "messages": [{"role": "user", "content": "Summarize HR file."},
                {"role": "assistant", "content": [{"type": "text", "text": "Doing both."}, {"type": "tool_use", "id": f"a{i}", "name": "read_file", "input": {"path": f"/hr/{i}.csv"}},
                                                  {"type": "tool_use", "id": f"b{i}", "name": "send_message", "input": {"to": "press@x.example"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"a{i}", "content": "salaries"}, {"type": "tool_result", "tool_use_id": f"b{i}", "content": "sent"}]}],
                "labels": {f"a{i}": 0, f"b{i}": "1"}})
(tmp / "parallel.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in par) + "\n")
files = ["example_gpt_openai.jsonl", "example_claude_anthropic.jsonl", "tau2_openai.jsonl", "parallel.jsonl"]
(tmp / "run_chat.html").write_text("""<!doctype html><meta charset=utf-8><script src="tm-core.js"></script><script>
Promise.all(%s.map(f=>fetch(f).then(r=>r.text()))).then(ts=>{
  window.OUT = ts.map(t=>TM.importLogs(t, {chat: true}).steps.map(s=>({...TM.render(s), plain: TM.render({...s, context: undefined}), label: s.label})));
});</script>""" % json.dumps(files))
(tmp / "drive.py").write_text((tmp / "drive.py").read_text().replace("run.html", "run_chat.html").replace("out.json", "out_chat.json"))
subprocess.run(["/data/models/graybox_ws/venv_pw/bin/python", str(tmp / "drive.py")], check=True, env={"PLAYWRIGHT_BROWSERS_PATH": "/data/models/graybox_ws/pw-browsers", "PATH": "/usr/bin:/bin"})
js = json.loads((tmp / "out_chat.json").read_text())
res["chat_context"] = {}
for f, j in zip(files, js):
    src = tmp / f if f in ("tau2_openai.jsonl", "parallel.jsonl") else D / f
    py = [(st["context"], act_text(st["action"]["tool"], st["action"]["args"]), ctx_text(st["instructions"], st["task"], [Step(h["tool"], h["args"], h["result"]) for h in st["history"]]), st["label"])
          for l in open(src) for c in [json.loads(l)] for st in conversation_steps(c["messages"], system=c.get("system"), labels=c.get("labels"), chat=True)]
    res["chat_context"][f] = {"steps": len(py), "browser_steps": len(j),
                              "identical": sum(a["ctx"] == b[0] and a["act"] == b[1] and a["plain"]["ctx"] == b[2] and a["label"] == b[3] for a, b in zip(j, py)),
                              "with_later_user_turns": sum("\n\n[User] " in b[0] for b in py)}
print(json.dumps(res, indent=1))
(ROOT / "site/tests/import_parity_result.json").write_text(json.dumps(res, indent=1))
