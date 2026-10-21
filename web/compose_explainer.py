#!/usr/bin/env python
"""Compose the explainer: narration placed at the recorded segment starts, the page-load lead-in trimmed, subtitles
(sentence timings proportional to text length within each segment). Writes narration.wav and explainer.srt.
python web/compose_explainer.py <media_dir> web/explainer_script.json"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

M, S = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
T = json.loads((M / "timeline.json").read_text())
lead = max(0.0, T[0]["start"] - 0.6)                                   # trim the page-load lead-in
end = T[-1]["end"] - lead + 0.8
sr = 24000
track = np.zeros(int(end * sr) + sr, dtype=np.float32)
srt, k = [], 1
fmt = lambda t: f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(round((t % 1) * 1000)) % 1000:03d}"  # noqa: E731
text = {s["id"]: s["text"] for s in S["segments"]}
for seg in T:
    a, _ = sf.read(M / f"{seg['id']}.wav", dtype="float32")
    i0 = int((seg["start"] - lead) * sr)
    track[i0:i0 + len(a)] += a
    speech = (len(a) / sr) - 0.45
    sents = [x for x in re.split(r"(?<=[.?!:])\s+", text[seg["id"]]) if x]
    total = sum(len(x) for x in sents)
    t = seg["start"] - lead
    for x in sents:
        d = speech * len(x) / total
        srt.append(f"{k}\n{fmt(t)} --> {fmt(t + d)}\n{x}\n")
        k, t = k + 1, t + d
peak = np.abs(track).max()
sf.write(M / "narration.wav", (track / max(peak, 1e-9) * 0.9).astype(np.float32), sr)
(M / "explainer.srt").write_text("\n".join(srt))
(M / "trim.json").write_text(json.dumps({"lead": lead, "end": end}))
print("lead", round(lead, 2), "length", round(end, 1), "s;", k - 1, "subtitles")
