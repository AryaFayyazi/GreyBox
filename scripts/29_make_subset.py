#!/usr/bin/env python
"""Slice the clean-unit artefacts (self Ψ, self hidden, 8B-reader Ψ and hidden) to the 2,000-unit subset used for the
27B reader, so scripts/24 can compare every monitor on exactly the same units (tag=cleansub).

python scripts/29_make_subset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

I = Path("/data/models/graybox_ws/outputs/interrogate")


def main():
    idx = np.load(I / "gemma4_cleansub_index.npy")
    U = pd.read_parquet(I / "gemma4_clean_units.parquet")
    sub_list = pd.read_parquet(I / "gemma4_cleansub_unitlist.parquet")
    assert (U.episode_id.values[idx] == sub_list.episode_id.values).all() and (U.event_id.values[idx] == sub_list.event_id.values).all()
    for src, dst in (("gemma4_clean", "gemma4_cleansub"), ("gemma4_clean_reader-qwen3_8b", "gemma4_cleansub_reader-qwen3_8b")):
        pd.read_parquet(I / f"{src}_units.parquet").iloc[idx].reset_index(drop=True).to_parquet(I / f"{dst}_units.parquet")
        z = dict(np.load(I / f"{src}.npz", allow_pickle=True))
        np.savez_compressed(I / f"{dst}.npz", **{k: (v[idx] if getattr(v, "ndim", 0) >= 2 or (getattr(v, "ndim", 0) == 1 and len(v) == len(U)) else v)
                                                for k, v in z.items()})
        h = np.load(I / f"{src}_hidden.npz")
        np.savez_compressed(I / f"{dst}_hidden.npz", h=h["h"][idx], layers=h["layers"], n_blocks=h["n_blocks"])
        print("wrote", dst)
    src = I / "gemma4_clean_reader-qwen38_hidden.npz"            # 27B reader probe input (job 3838, all clean units)
    if src.exists():
        h = np.load(src)
        Ur = pd.read_parquet(I / "gemma4_clean_reader-qwen38_units.parquet")
        assert len(Ur) == len(U) and (Ur.episode_id.values == U.episode_id.values).all()
        np.savez_compressed(I / "gemma4_cleansub_reader-qwen38_hidden.npz", h=h["h"][idx], layers=h["layers"],
                            n_blocks=h["n_blocks"])
        print("wrote gemma4_cleansub_reader-qwen38_hidden")


if __name__ == "__main__":
    main()
