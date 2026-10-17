#!/usr/bin/env python
"""Build a self-contained Live v2 replay page: web/live_replay_v2.html + one recorded session + page text.

python web/build_replay.py <session.json> <meta.json> <out.html>
meta.json: title, subtitle, note, facts (HTML), trust_harmful, trust_refuse (all written from measured numbers).
"""
import json
import sys
from pathlib import Path


def main(session, meta, out):
    d = json.loads(Path(session).read_text())
    d["meta"] = json.loads(Path(meta).read_text())
    keep = {"events", "cloud", "cloud_label", "meta", "model", "battery"}
    d = {k: v for k, v in d.items() if k in keep}
    for e in d["events"]:
        e.pop("answers", None)
        e.pop("usage", None)
    tpl = (Path(__file__).parent / "live_replay_v2.html").read_text()
    blob = json.dumps(d, separators=(",", ":")).replace("</", "<\\/")
    Path(out).write_text(tpl.replace("<!--DATA-->", f"<script>window.LIVE_SESSION={blob};</script>"))
    print("wrote", out, len(blob) // 1024, "KB data")


if __name__ == "__main__":
    main(*sys.argv[1:4])
