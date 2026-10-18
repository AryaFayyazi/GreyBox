"""Load a reader probe file (graybox.train_reader or the release) for deployment.

ReaderMonitor.load (graybox.monitor) is the code that the preregistered tests ran; it looks for the reader in a fixed
local cache. load_reader uses the standard Hugging Face cache (HF_HOME / HF_HUB_CACHE; downloads the reader once)
and keeps the probe's "chat" setting, which Guard and the server then follow.

    from graybox.reader_load import load_reader
    reader = load_reader("my_reader_probe.joblib")          # GPU; device="cpu" works but takes ~2.5 s per action (8B)
    guard = Guard(reader, gate=QuantileGate(0.05))         # chat follows the probe file
"""
from __future__ import annotations


def make_reader(probe, layer: int, reader: str = "Qwen/Qwen3-8B", device: str = "cuda", cache_dir: str | None = None):
    """A ReaderMonitor (same attributes and methods) whose model comes from the standard Hugging Face cache."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .monitor import ReaderMonitor
    kw = {"cache_dir": cache_dir} if cache_dir else {}
    m = object.__new__(ReaderMonitor)
    m.tok = AutoTokenizer.from_pretrained(reader, **kw)
    m.model = AutoModelForCausalLM.from_pretrained(reader, dtype=torch.bfloat16, device_map=device, **kw).eval()
    m.probe, m.layer, m.torch = probe, layer, torch
    return m


def load_reader(path, device: str = "cuda", cache_dir: str | None = None):
    import joblib
    d = joblib.load(path)
    m = make_reader(d["probe"], d["layer"], d.get("reader", "Qwen/Qwen3-8B"), device, cache_dir)
    m.chat = bool(d.get("chat", False))
    m.canonical = bool(d.get("canonical", False))     # trained on canonical actions: Guard reads actions the same way
    m.trained_on = d.get("trained_on")
    return m
