#!/usr/bin/env python
"""Narration for the explainer video: one WAV per segment (Kokoro-82M, open model, CPU) and their durations.
python web/tts_explainer.py web/explainer_script.json <out_dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

S, OUT = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
pipe = KPipeline(lang_code="a")
durs = {}
for seg in S["segments"]:
    audio = np.concatenate([np.asarray(a) for _, _, a in pipe(seg["text"], voice=S["voice"], speed=1.0)])
    audio = np.concatenate([audio, np.zeros(int(0.45 * 24000), dtype=audio.dtype)])     # a short pause after each segment
    sf.write(OUT / f"{seg['id']}.wav", audio, 24000)
    durs[seg["id"]] = round(len(audio) / 24000, 3)
    print(seg["id"], durs[seg["id"]], flush=True)
(OUT / "durations.json").write_text(json.dumps(durs, indent=1))
print("total", round(sum(durs.values()), 1), "s")
