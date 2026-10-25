#!/usr/bin/env python
"""G68 deviation (mechanical): Muse-Glimmer's vLLM server could not start under vLLM 0.26 (no 'muse_glimmer' tool-call
parser), so the G68 job drew nothing for it. G27b ran Muse-Glimmer under vLLM 0.30 by pointing the server launcher at
/data/models/cape_ws/envs/v030 (scripts/197 vllm_python=...). This wrapper does the same, then calls the LOCKED
scripts/260 stage_draws for Muse-Glimmer unchanged (smoke first, separate file, never analysed), then the locked analysis.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("g68", ROOT / "scripts/260_g68_open_says_does.py")
M = importlib.util.module_from_spec(s)
s.loader.exec_module(M)
sys.executable = "/data/models/cape_ws/envs/v030/bin/python"     # the server launcher uses sys.executable (as in G27b)
M.stage_draws("museglimmer", 4, M.G27.OUT / "g68smoke_museglimmer.jsonl")
M.stage_draws("museglimmer")
M.stage_analyze()
