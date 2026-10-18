"""Parity of the canonical form: Python (graybox.monitor.canonical_ctx_text / canonical_act_text) vs the browser
(site/src/assets/tm-core.js TM.canonical), on every rendered step the paper uses and on every T25 rewrite.

The browser trainer and the SDK must read the same canonical text, or a model trained in one and served by the other
would score a different input. Runs the browser code with Node (Playwright's bundled Node if `node` is not on PATH).

python scripts/121_canonical_parity.py   -> results/canonical_parity.json
"""
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", Path("/home/arya/new_safety/cape/src"), Path("/home/arya/new_safety/cape/scripts")):
    sys.path.insert(0, str(p))
from graybox.monitor import act_text, canonical_act_text, canonical_ctx_text  # noqa: E402

NODE = shutil.which("node") or "/data/models/graybox_ws/venv_pw/lib/python3.13/site-packages/playwright/driver/node"
JS = r"""
const fs = require('fs'); global.self = global;
eval(fs.readFileSync(process.argv[2], 'utf8'));
const rows = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
fs.writeFileSync(process.argv[4], JSON.stringify(rows.map(r => TM.canonical(r))));
"""

HAND = [  # edge cases: nested fields, lists, case of field names, comments, short and long quotes, unparseable text
    ("[Agent called] send {\"to\": \"a@x.example\", \"Body\": \"hi\"}", "send {\"to\": \"a@x.example\", \"Subject\": \"QA ok\"}"),
    ("", "run {\"command\": \"rm -rf /tmp/x # cleanup approved by ops\"}"),
    ("", "run {\"command\": \"deploy --note 'approved by the CFO today' --to 'prod'\"}"),
    ("", "post {\"items\": [{\"text\": \"x\"}, {\"title\": 7}], \"meta\": {\"reason\": [\"a\", \"b\"]}}"),
    ("", "final answer to the user (no tool call)"),
    ("", "tool not-json {"),
    ("[Agent called] a {\"q\": \"\\\"three word quote\\\" kept?\"}\n\n[Tool result] {\"body\": \"results are not changed\"}", "a {}"),
    ("", "x {\"command\": \"echo \\\"été à Paris\\\" 　 done\"}"),
    # number text and key order survive a parse and a dump (integer-like keys, duplicate keys, floats, big ints)
    ("", "x {\"2\": \"b\", \"1\": 50.0, \"note\": \"y\", \"n\": 1e-05, \"big\": 12345678901234567890, \"note\": [1.50, -0, 2E3]}"),
    ("", "x [1, 2]"),
    ("", "x {\"a\": NaN, \"b\": Infinity, \"c\": \"\\ud83d\\ude00 'one two three' ok\", \"d\": -0.0}"),
]


def main():
    m19 = importlib.import_module("100_t19_small_readers")
    t8b, t11 = importlib.import_module("76_t8b_claude_all_actions"), importlib.import_module("81_t11_gpt_monitor")
    rows = []
    for name, df in {"train": m19.data("train"), "T8b": t8b.units(), "T11": t11.units(), "T7": m19.data("T7"),
                     "T13b": m19.data("T13b")}.items():
        rows += [(name, c, a) for c, a in zip(df.ctx, df.act)]
    t25 = Path("/data/models/graybox_ws/outputs/t25")
    T = [json.loads(x) for x in (t25 / "targets.jsonl").read_text().splitlines()]
    for f in ("gen1.json", "gen2_READER.json", "gen2_TXT.json"):
        for g in json.loads((t25 / f).read_text()):
            t = T[g["i"]]
            rows += [("T25_rewrites", t["ctx"], act_text(t["tool"], c)) for c in g["cands"] if isinstance(c, dict)]
    rows += [("hand", c, a) for c, a in HAND]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run.js").write_text(JS)
        (d / "in.json").write_text(json.dumps([{"ctx": c, "act": a} for _, c, a in rows], ensure_ascii=False))
        subprocess.run([NODE, str(d / "run.js"), str(ROOT / "site/src/assets/tm-core.js"), str(d / "in.json"), str(d / "out.json")], check=True)
        js = json.loads((d / "out.json").read_text())
    res, bad = {}, []
    for (name, c, a), j in zip(rows, js):
        py = {"ctx": canonical_ctx_text(c), "act": canonical_act_text(a)}
        r = res.setdefault(name, {"n": 0, "differ": 0, "act_changed": 0})
        r["n"] += 1
        r["act_changed"] += py["act"] != a
        if py != j:
            r["differ"] += 1
            if len(bad) < 5:
                bad.append({"set": name, "py": {k: v[:300] for k, v in py.items()}, "js": {k: v[:300] for k, v in j.items()}})
    out = {"sets": res, "all_identical": all(r["differ"] == 0 for r in res.values()), "examples_of_differences": bad}
    (ROOT / "results/canonical_parity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:3000])
    sys.exit(0 if out["all_identical"] else 1)


if __name__ == "__main__":
    main()
