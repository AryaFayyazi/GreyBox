#!/usr/bin/env python
"""Exploratory multiplicity check (not preregistered). The family: every preregistered endpoint behind a closed-agent claim
in the abstract (15). For each, a one-sided p-value against its registered null from the stored estimate and 95% cluster
bootstrap interval (normal approximation: SE = (upper - lower) / 3.92), then Holm's step-down at family-wise 0.05.
-> results/t1/headline_holm_exploratory.json"""
import json
from math import erf, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = lambda f: json.loads((ROOT / f"results/t1/{f}.json").read_text())  # noqa: E731
F = [("G38 A1 GPT-6-luna read-out at the call", J("g38_gpt_readout")["A1_at_call"], 0.5),
     ("G15 T3 Claude restarts read-out", [J("g15_training_free")["claude"]["restart_mean8"]] + J("g15_training_free")["claude"]["restart_mean8_ci"], 0.65),
     ("G40 C1 GPT-6-luna pre-call read-out", J("g40_gpt_precall_confirm")["C1_within_template_tool_position"], 0.5),
     ("G46 R1 Claude own draws", J("g46_rollout_forecast")["R1_own_within_position"], 0.5),
     ("G46 R3 own draws minus self-report", J("g46_rollout_forecast")["R3_own_minus_report_position"], 0.0),
     ("G52 J1 own draws minus Claude text judge", J("g52_text_judge_vs_policy")["J1_own_minus_judge_position"], 0.0),
     ("G55 J1 own draws minus GPT text judge", J("g55_gpt_judges_claude")["J1_own_minus_gpt_judge_position"], 0.0),
     ("G56 M1 geometric monitor, Claude", J("g56_geometric_monitor")["claude"]["M1_geometric_within_stratum"], 0.5),
     ("G56 M1 geometric monitor, GPT-6-luna", J("g56_geometric_monitor")["gpt"]["M1_geometric_within_stratum"], 0.5),
     ("G57 G1 geometric forecast, Claude", J("g57_geometric_forecast")["claude"]["G1_geometric_forecast_within_position"], 0.5),
     ("G57 G1 geometric forecast, GPT-6-luna", J("g57_geometric_forecast")["gpt"]["G1_geometric_forecast_within_position"], 0.5),
     ("G47 P1 Claude self-gate reduction", J("g47_selfgate")["self_reduction"], 0.0),
     ("G47 P2 Claude self-gate minus random", J("g47_selfgate")["self_minus_random"], 0.0),
     ("G51 P1 GPT-6-luna self-gate reduction", J("g51_gpt_selfgate")["P1_self_reduction"], 0.0),
     ("G51 P2 GPT-6-luna self-gate minus random", J("g51_gpt_selfgate")["P2_self_minus_random"], 0.0)]
rows = []
for name, (est, lo, hi), null in F:
    z = (est - null) / ((hi - lo) / 3.92)
    rows.append({"endpoint": name, "estimate": est, "ci": [lo, hi], "null": null, "z": round(z, 2), "p_one_sided": 0.5 * (1 - erf(z / sqrt(2)))})
rows.sort(key=lambda r: r["p_one_sided"])
k, ok = len(rows), True
for j, r in enumerate(rows):
    r["holm_threshold"] = 0.05 / (k - j)
    ok = ok and r["p_one_sided"] <= r["holm_threshold"]
    r["rejected_holm"] = ok
res = {"family_size": k, "all_survive_holm": all(r["rejected_holm"] for r in rows), "largest_p": max(r["p_one_sided"] for r in rows), "rows": rows}
(ROOT / "results/t1/headline_holm_exploratory.json").write_text(json.dumps(res, indent=1))
for r in rows:
    print(f'{r["endpoint"]:45s} z={r["z"]:6.2f} p={r["p_one_sided"]:.2e} thr={r["holm_threshold"]:.4f} {r["rejected_holm"]}')
print(res["all_survive_holm"])
