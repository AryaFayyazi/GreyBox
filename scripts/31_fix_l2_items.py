"""Repair the `items` metadata of L2 psi.npz files written before the shadowing fix (scripts/12, 2026-10-15 20:45):
the Ψ columns are in battery-v2 order and correct; only the saved item labels were overwritten by token ids."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, "/home/arya/graybox/src")
from graybox.battery_v2 import BATTERY_V2
for d in sorted(Path("/data/models/graybox_ws/outputs/l2").iterdir()):
    p = d / "psi.npz"
    if not p.exists():
        continue
    z = dict(np.load(p, allow_pickle=True))
    if z["p_yes"].shape[1] == len(BATTERY_V2) and list(z["items"]) != [k for k, _ in BATTERY_V2]:
        z["items"] = np.array([k for k, _ in BATTERY_V2])
        np.savez_compressed(p, **z)
        print("repaired", d.name)
