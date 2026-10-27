#!/usr/bin/env python
"""Exploratory multiplicity check (not preregistered), replacing scripts/245's family of passing endpoints. The family:
EVERY registered endpoint, passed or failed, of every closed-agent test whose result the abstract cites, and of the
tests that bear on those claims (G62). Open-agent endpoints are left out. (Revised 2026-10-24 14:20 after review: the first
version mixed in two open-agent G15 endpoints and predated G59-G62. Revised 2026-10-26 after review round 18: adds every
CI-based endpoint of the later closed-agent tests G65-G67, G72, G75, G76 and G78, then G79, G80 and G80b; count endpoints (G73 E1, G78 P2) and
dose checks have no interval and stay out.)
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
F = [("Read-out at the call: Claude restarts", [g15["claude"]["restart_mean8"]] + g15["claude"]["restart_mean8_ci"], "gt", 0.65),
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
     ("Claude own draws - self-report, same calls", J("g59_selfreport_exact")["S1_own_draws_minus_self_report"], "gt", 0.0),
     ("Loss predicted, Claude's judge", J("g60_text_monitor_loss")["pairs"]["claude_self"]["observed_minus_predicted"], "eq", 0.02),
     ("Loss predicted, GPT-6-luna judging Claude", J("g60_text_monitor_loss")["pairs"]["claude_gpt"]["observed_minus_predicted"], "eq", 0.02),
     ("Loss predicted, GPT-6-luna judging itself", J("g60_text_monitor_loss")["pairs"]["gpt_self"]["observed_minus_predicted"], "eq", 0.02),
     ("Label-free gap, Claude - GPT-6-luna", J("g60_text_monitor_loss")["E2_claude_self_gap_minus_gpt_self_gap"], "gt", 0.0),
     ("Draws add to GPT-6-luna's judge, across steps", J("g61_draws_add_to_text")["A1_gpt_judge+draws_minus_gpt_judge"], "gt", 0.0),
     ("Draws add to Claude's judge, across steps", J("g61_draws_add_to_text")["A2_claude_judge+draws_minus_claude_judge"], "gt", 0.0),
     ("Draws add to position, across steps", J("g61_draws_add_to_text")["A3_position+draws_minus_position"], "gt", 0.0),
     ("Loss predicted with no label, Claude's judge", J("g62_loss_no_labels")["pairs"]["claude_self"]["observed_minus_predicted"], "eq", 0.02),
     ("Loss predicted with no label, GPT-6-luna judging Claude", J("g62_loss_no_labels")["pairs"]["claude_gpt"]["observed_minus_predicted"], "eq", 0.02),
     ("Loss predicted, recalibrated reader, Claude's judge", J("g63_loss_calibrated_reader")["pairs"]["claude_self"]["observed_minus_predicted"], "eq", 0.02),
     ("Loss predicted, recalibrated reader, GPT-6-luna judging Claude", J("g63_loss_calibrated_reader")["pairs"]["claude_gpt"]["observed_minus_predicted"], "eq", 0.02),
     ("Text-policy agreement, GPT-6-luna - Claude", J("g64_text_policy_agreement")["A1_gpt_minus_claude"], "gt", 0.0),
     ("Claude reviewer-free gate: reduction", g47["self_reduction"], "gt", 0.0),
     ("Claude reviewer-free gate - random", g47["self_minus_random"], "gt", 0.0),
     ("Claude reviewer-free gate: success (non-inferiority)", g47["success_self_minus_none"], "gt", -0.05),
     ("GPT-6-luna reviewer-free gate: reduction", g51["P1_self_reduction"], "gt", 0.0),
     ("GPT-6-luna reviewer-free gate - random", g51["P2_self_minus_random"], "gt", 0.0),
     ("GPT-6-luna reviewer-free gate: success (non-inferiority)", g51["P3_success_self_minus_none"], "gt", -0.05)]
g65, g66, g67, g72 = J("g65_replication")["endpoints"], J("g66_geometry_replication"), J("g67_matched_dose"), J("g72_layers_stack")
E = lambda d: [d["estimate"]] + d["ci"] if isinstance(d, dict) else d  # noqa: E731
F += [("Replication: own draws - self-report", E(g65["P1_own_minus_report"]), "gt", 0.0),
      ("Replication: own draws - Claude text judge", E(g65["P2_own_minus_claude_judge"]), "gt", 0.0),
      ("Replication: own draws - GPT-6-luna text judge", E(g65["P3_own_minus_gpt_judge"]), "gt", 0.0),
      ("Replication: draws add to GPT-6-luna's judge across steps", E(g65["P4_gpt_judge_plus_draws_minus_gpt_judge_across_steps"]), "gt", 0.0),
      ("Replication: own draws within position", E(g65["P5_own_within_position"]), "gt", 0.5),
      ("Replication: loss predicted, Claude's judge", E(g65["P6a_loss_claude_judge_obs_minus_pred"]), "eq", 0.02),
      ("Replication: loss predicted, GPT-6-luna's judge", E(g65["P6b_loss_gpt_judge_obs_minus_pred"]), "eq", 0.02),
      ("Replication: GPT-6-luna reads draws - its judge", E(g65["P7_gpt_reads_draws_minus_gpt_judge"]), "gt", 0.0),
      ("Geometric monitor replicated, Claude", g66["monitor"]["M1_geometric_within_stratum"], "gt", 0.5),
      ("Geometric forecast replicated, Claude", g66["forecast"]["G1_geometric_forecast_within_position"], "gt", 0.5),
      ("Matched dose: self - random", g67["self_minus_random"], "gt", 0.0),
      ("Matched dose: self reduction", g67["self_reduction"], "gt", 0.0),
      ("Matched dose: success (non-inferiority)", g67["success_self_minus_none"], "gt", -0.05),
      ("Geometry gate on reminder: reduction", g72["P1_remind_geo_vs_remind"], "gt", 0.0),
      ("Geometry gate: reduction", g72["P2_geo_vs_none"], "gt", 0.0),
      ("Geometry gate on reminder: success (non-inferiority)", g72["P3_success_remind_geo_minus_none"], "gt", -0.05),
      ("Geometry gate - random, tie at random", J("g75_geo_vs_random")["R1_random_minus_geo_violating"], "gt", 0.0),
      ("Geometry gate - random, tie held", J("g76_geo_vs_random_nocoin")["R1_random_minus_geo_violating"], "gt", 0.0),
      ("Claude's geometry, all 88 questions - strongest text", J("g78_claude_full_battery")["P1_mean_claude_minus_strongest"], "gt", 0.0)]
g79, g80, g80b = J("g79_fresh_worlds"), J("g80_gpt_geo_gate"), J("g80b_gpt_dose_matched_random")
F += [("New worlds: geometric forecast, Claude", g79["forecast"]["F1_geometric"], "gt", 0.5),
      ("New worlds: signed forecast, Claude", g79["forecast"]["F2_signed"], "gt", 0.5),
      ("New worlds: geometric - generic rule, Claude", g79["forecast"]["F3_geometric_minus_generic_rule"], "gt", 0.0),
      ("New worlds: geometry gate - random, Claude", g79["gate"]["P1_random_minus_geo_violating"], "gt", 0.0),
      ("GPT-6-luna geometry gate - random (dose unmatched)", g80["Q1_random_minus_geo_violating"], "gt", 0.0),
      ("GPT-6-luna geometry gate - random at its dose", g80b["R1_random_matched_minus_geo"], "gt", 0.0)]
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
