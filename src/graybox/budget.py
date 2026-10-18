"""Shared spend ledger for paid APIs: one JSON file, updated under an exclusive file lock, so concurrent processes
(a benchmark run, the Live server) add to the same total and all stop at the same hard cap.

Ledger(path, cap_usd, price_in, price_out): call .check() before each paid call (it raises BudgetExceeded once the
total has reached the cap), then .charge(input_tokens, output_tokens), which adds the cost and returns the new total.
Prices are $ per token.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path


CAP_USD = float(os.environ.get("GB_SPEND_CAP", "4.95"))   # user, 2026-10-16 09:00: "the api was 5$, use all of it"
HAIKU55 = (0.10 / 1e6, 0.50 / 1e6)                          # $ per input / output token, claude-haiku-5-5


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, cap_usd: float, price_in: float, price_out: float):
        self.path, self.cap, self.pin, self.pout = Path(path), float(cap_usd), price_in, price_out
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock")

    def _update(self, fn):
        with open(self.lock, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                d = json.loads(self.path.read_text()) if self.path.exists() else \
                    {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
                out = fn(d)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(d))
                os.replace(tmp, self.path)
                return out
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def check(self):
        """Raise before a call if the cap is already reached."""
        d = json.loads(self.path.read_text()) if self.path.exists() else {"usd": 0.0}
        if d["usd"] >= self.cap:
            raise BudgetExceeded(f"spend ${d['usd']:.3f} reached the cap ${self.cap:.2f}")

    def charge(self, input_tokens: int, output_tokens: int) -> float:
        def f(d):
            d["input_tokens"] += int(input_tokens)
            d["output_tokens"] += int(output_tokens)
            d["calls"] += 1
            d["usd"] = d["input_tokens"] * self.pin + d["output_tokens"] * self.pout
            return d["usd"]
        return self._update(f)

    def total(self) -> float:
        return json.loads(self.path.read_text())["usd"] if self.path.exists() else 0.0
