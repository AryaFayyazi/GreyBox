#!/usr/bin/env python
"""Clean units for the behavioural-sufficiency test (theory T1): inclusion depends only on the past.

From CAPE rollouts of one actor:
  x− informed episodes: every proposed non-read action after the reveal, up to and including the first unsafe action.
      Y = 1 if that action is unsafe (cls XUNSAFE), else 0 (cls XSAFE).
  x+ twins: every proposed non-read action after the reveal (cls XPLUS, Y = 0).
Negatives are subsampled uniformly at random (independent of the future); all positives are kept.
Contrast with CAPE v57 units, where X-OTHER came only from episodes that never violate (future-dependent selection).

python scripts/15_clean_units.py actor=gemma4 minus='[xm_gemma4_core24b_minus6789,xm_gemma4_core24b_reasoning_minus6789]' plus='[xm_gemma4_core24b_plus]'
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.cape_bridge import READS, rollouts  # noqa: E402

OUT = Path("/data/models/graybox_ws/outputs/interrogate")
D = OmegaConf.create({"actor": "gemma4", "minus": [], "plus": [], "n_safe": 3000, "n_plus": 1500, "seed": 0,
                      "bench": "core24b", "tag": "clean", "episodes_minus": None, "episodes_plus": None})
# episode mode (episodes_minus / episodes_plus set): sample whole episodes uniformly and keep EVERY clean turn of each,
# so each sampled episode can be replayed turn by turn; selection still depends only on the past within an episode


def bench_of(set_name):
    """CAPE set names encode the benchmark split: <prefix>_<actor>_<bench>_<minus|plus><seeds>."""
    import re
    m = re.match(r"^(?:xm|gb)_[a-z0-9]+_(.+?)_(?:minus|plus)", set_name)
    return m.group(1) if m else "core24b"


def units_of(set_name, world, bench):
    rows = []
    for ep in rollouts(set_name).itertuples():
        if pd.isna(ep.reveal_event):
            continue                                   # uninformed episode
        rev = int(ep.reveal_event)
        for e in ep.events:
            t = int(e["event_id"])
            if t <= rev or not e["is_tool_call"] or e["tool"] in (None, *READS) or e["parse_error"] is not None:
                continue
            y = int(bool(e["unsafe_action"])) if world == "minus" else 0
            rows.append({"set": set_name, "episode_id": ep.episode_id, "event_id": t, "scenario_id": ep.scenario_id,
                         "template": f"{ep.template}|{e['tool']}", "bench": bench, "world": world, "y": y,
                         "cls": ("XUNSAFE" if y else "XSAFE") if world == "minus" else "XPLUS"})
            if y:
                break                                  # stop at the first violation
    return rows


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    rows = [r for s in cfg.minus for r in units_of(s, "minus", bench_of(s))] + \
           [r for s in cfg.plus for r in units_of(s, "plus", bench_of(s))]
    U = pd.DataFrame(rows)
    print("pool:", U.cls.value_counts().to_dict())
    rng = np.random.default_rng(cfg.seed)
    if cfg.episodes_minus is not None:
        keep = []
        for world, n in (("minus", cfg.episodes_minus), ("plus", cfg.episodes_plus or 0)):
            eps = U[U.world == world].episode_id.unique()
            pick = set(rng.choice(eps, min(int(n), len(eps)), replace=False))
            keep.append(U[U.episode_id.isin(pick)])
        U = pd.concat(keep).sort_values(["set", "episode_id", "event_id"]).reset_index(drop=True)
        U["I"] = np.nan
        U.to_parquet(OUT / f"{cfg.actor}_{cfg.tag}_unitlist.parquet")
        print("episode mode units:", U.cls.value_counts().to_dict(), "episodes:", U.episode_id.nunique())
        return
    keep = [U[U.cls == "XUNSAFE"]]
    for cls, n in (("XSAFE", cfg.n_safe), ("XPLUS", cfg.n_plus)):
        g = U[U.cls == cls]
        keep.append(g.iloc[rng.choice(len(g), min(n, len(g)), replace=False)])
    U = pd.concat(keep).sort_values(["set", "episode_id", "event_id"]).reset_index(drop=True)
    U["I"] = np.nan
    OUT.mkdir(parents=True, exist_ok=True)
    U.to_parquet(OUT / f"{cfg.actor}_{cfg.tag}_unitlist.parquet")
    print("units:", U.cls.value_counts().to_dict())


if __name__ == "__main__":
    main()
