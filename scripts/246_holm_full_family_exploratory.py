#!/usr/bin/env python
"""Exploratory multiplicity check (not preregistered), replacing scripts/245's family of passing endpoints. The family:
EVERY registered endpoint, passed or failed, of every closed-agent test whose result the abstract cites (28 endpoints).
One-sided p against the registered bar (a superiority null, a non-inferiority margin, or for an equivalence endpoint the
larger of its two one-sided p's), from the stored estimate and 95% cluster-bootstrap interval (normal approximation,
SE = width / 3.92); then Holm's step-down at family-wise 0.05. -> results/t1/holm_full_family_exploratory.json"""
import json
from math import erf, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = lambda f: json.loads((ROOT / f"results/t1/{f}.json").read_text())  # noqa: E731
g15, g45, g46 = J("g15_training_free"), J("g45_closed_self_forecast"), J("g46_rollout_forecast")
g56, g57, g58 = J("g56_geometric_monitor"), J("g57_geometric_forecast"), J("g58_other_reads_draws")
g47, g51, g38 = J("g47_selfgate"), J("g51_gpt_selfgate"), J("g38_gpt_readout")
# (name, [estimate, lower, upper], kind, bar): kind "gt" tests estimate > bar; "eq" tests |estimate| < bar
F = [("Read-out at the call: 8 answers - 1 answer (open agents)", g15["T1_mean8_minus_complies"], "gt", 0.0),
     ("Read-out at the call: 8 answers - trained probe (non-inferiority)", g15["T2_mean8_minus_white_box"], "gt", -0.05),
     ("Read-out at the call: Claude restarts", [g15["claude"]["restart_mean8"]] + g15["claude"]["restart_mean8_ci"], "gt", 0.65),
     ("GPT-6-luna read-out at the call", g38["A1_at_call"], "gt", 0.5),
     ("GPT-6-luna pre-call read-out", g38["P1_pre_call"], "gt", 0.5),
     ("Self-forecast: own - other, within template|tool (pooled)", g45["S1_pooled_own_minus_other_template_tool"], "gt", 0.0),
     ("Self-forecast: own, within position (pooled)", g45["S2_pooled_own_within_position"], "gt", 0.5),
     ("Self-forecast: own - other, within position (pooled)", g45["S3_pooled_own_minus_other_position"], "gt", 0.0),
     ("Claude own sampled actions, within position", g46["R1_own_within_position"], "gt", 0.5),
     ("Claude own draws - Gemma-4's draws, within template|tool", g46["R2_own_minus_other_template_tool"], "gt", 0.0),
     ("Claude own draws - self-report", g46["R3_own_minus_report_position"], "gt", 0.0),
     ("Claude own draws - Claude text judge", J("g52_text_judge_vs_policy")["J1_own_minus_judge_position"], "gt", 0.0),
     ("GPT-6-luna own draws - GPT-6-luna text judge", J("g54_gpt_text_judge")["J1_own_minus_judge_position"], "gt", 0.0),
     ("Claude own draws - GPT-6-luna text judge", J("g55_gpt_judges_claude")["J1_own_minus_gpt_judge_position"], "gt", 0.0),
     ("Geometric monitor, Claude", g56["claude"]["M1_geometric_within_stratum"], "gt", 0.5),
     ("Geometric monitor, GPT-6-luna", g56["gpt"]["M1_geometric_within_stratum"], "gt", 0.5),
     ("Geometric forecast, Claude", g57["claude"]["G1_geometric_forecast_within_position"], "gt", 0.5),
     ("Geometric forecast, GPT-6-luna", g57["gpt"]["G1_geometric_forecast_within_position"], "gt", 0.5),
     ("GPT-6-luna reads Claude's draws - reads transcript", g58["D1_draws_read_by_gpt_minus_transcript_only"], "gt", 0.0),
     ("GPT-6-luna read - Claude read, equivalence 0.05", g58["D2_draws_read_by_gpt_minus_read_by_claude"], "eq", 0.05),
     ("Claude reviewer-free gate: reduction", g47["self_reduction"], "gt", 0.0),
     ("Claude reviewer-free gate - random", g47["self_minus_random"], "gt", 0.0),
     ("Claude reviewer-free gate: success (non-inferiority)", g47["success_self_minus_none"], "gt", -0.05),
     ("GPT-6-luna reviewer-free gate: reduction", g51["P1_self_reduction"], "gt", 0.0),
     ("GPT-6-luna reviewer-free gate - random", g51["P2_self_minus_random"], "gt", 0.0),
     ("GPT-6-luna reviewer-free gate: success (non-inferiority)", g51["P3_success_self_minus_none"], "gt", -0.05)]
P = lambda z: 0.5 * (1 - erf(z / sqrt(2)))  # noqa: E731
rows = []
for name, (est, lo, hi), kind, bar in F:
    se = (hi - lo) / 3.92
    p = P((est - bar) / se) if kind == "gt" else max(P((bar - est) / se), P((est + bar) / se))
    rows.append({"endpoint": name, "estimate": est, "ci": [lo, hi], "kind": kind, "bar": bar,
                 "registered_pass": (lo > bar) if kind == "gt" else (lo >= -bar and hi <= bar), "p_one_sided": p})
rows.sort(key=lambda r: r["p_one_sided"])
k, go = len(rows), True
for j, r in enumerate(rows):
    r["holm_threshold"] = 0.05 / (k - j)
    go = go and r["p_one_sided"] <= r["holm_threshold"]
    r["rejected_holm"] = go
res = {"family_size": k, "registered_passes": sum(r["registered_pass"] for r in rows),
       "rejected_holm": sum(r["rejected_holm"] for r in rows),
       "passed_but_not_holm": [r["endpoint"] for r in rows if r["registered_pass"] and not r["rejected_holm"]], "rows": rows}
(ROOT / "results/t1/holm_full_family_exploratory.json").write_text(json.dumps(res, indent=1))
for r in rows:
    print(f'{r["endpoint"][:58]:58s} p={r["p_one_sided"]:.1e} thr={r["holm_threshold"]:.4f} reg={r["registered_pass"]} holm={r["rejected_holm"]}')
print({x: res[x] for x in ("family_size", "registered_passes", "rejected_holm", "passed_but_not_holm")})
