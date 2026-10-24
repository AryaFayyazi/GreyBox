#!/usr/bin/env python
"""G66: prospective replication of the geometry-only monitor (G56) and forecast (G57) on G65's fresh Claude Haiku 5.5
episodes (docs/prereg_G66_geometry_replication.md). The functions of scripts/241 (G56: calls, the monitor) and scripts/242
(G57: analyse, the forecast) run unchanged on G65's own-draw file and units. No label, no question sign.
  -> results/t1/g66_geometry_replication.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_N, BOOT_SEED = 2000, 6600


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    from graybox.stats import within_auroc
    REP = _mod("g65", "scripts/254_g65_replication.py")
    G56, G57 = _mod("g56", "scripts/241_g56_geometric_monitor.py"), _mod("g57", "scripts/242_g57_geometric_forecast.py")
    U, _ = REP.G46C.units()
    D = G56.calls(REP.G46C.OWN, U, lambda d: "cons_service_restart")
    y, st, cl = D.y.values, D.stratum.values, D.scenario_id.values
    rng, us = np.random.default_rng(BOOT_SEED), np.unique(cl)
    B = [within_auroc(y[i], D.geo.values[i], st[i]) for i in (np.concatenate([np.flatnonzero(cl == u) for u in rng.choice(us, len(us))]) for _ in range(B_N))]
    g = float(within_auroc(y, D.geo.values, st))
    mon = {"calls": int(len(D)), "violations": int(y.sum()), "M1_geometric_within_stratum": [round(g, 4), round(float(np.nanpercentile(B, 2.5)), 4), round(float(np.nanpercentile(B, 97.5)), 4)],
           "signed_readout_within_stratum": round(float(within_auroc(y, D.readout.values, st)), 4)}
    fc = G57.analyse(U, REP.G46C.OWN)
    res = {"monitor": mon, "forecast": fc,
           "endpoints": {"M1_lower_gt_0.5": mon["M1_geometric_within_stratum"][1] > 0.5,
                         "G1_lower_gt_0.5": fc["G1_geometric_forecast_within_position"][1] > 0.5}}
    (ROOT / "results/t1/g66_geometry_replication.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
