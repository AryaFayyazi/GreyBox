#!/usr/bin/env python
"""Write paper/second_numbers.tex from the second-benchmark results (the paper quotes them only through these macros):
G18 results/t1/g18_tau2_claude_readout.json (tau2-bench, Claude Haiku 5.5) and, when present, G17
results/t1/g17_agentdojo_readout.json (AgentDojo, open agents).
python paper/make_second_numbers.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f3(x):
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)).replace("-", "$-$")


def ci(v, sign=True):
    return f"{'+' if sign and v[0] >= 0 else ''}{f3(v[0])} [{f3(v[1])}, {f3(v[2])}]"


m = {}
p = ROOT / "results/t1/g18_tau2_claude_readout.json"
if p.exists():
    R = json.loads(p.read_text())
    m.update({"GEeN": f"{R['writes']} writes, {R['violations']} violations",
              "GEeRone": ci(R["R1_within_tool_read_out_8"], sign=False),
              "GEeSingle": f3(R["within_tool_single_answer"]),
              "GEeRtwo": ci(R["R2_read_out_minus_single"]),
              "GEeJudge": f3(R["R3_vs_reasoning_judge"]["judge"]),
              "GEeJudgeOurs": f3(R["R3_vs_reasoning_judge"]["read_out_8"]),
              "GEeRthree": ci(R["R3_vs_reasoning_judge"]["difference"]),
              "GEeCaught": f"{100 * R['caught_at_5pct']['read_out_8']:.0f}\\% vs {100 * R['caught_at_5pct']['single_answer']:.0f}\\%"})
p = ROOT / "results/t1/g16_own_vs_other.json"
if p.exists():
    R = json.loads(p.read_text())
    for a, k in (("gemma4", "Gemma"), ("gptoss20b", "Gpt"), ("qwen38", "Qwen"), ("museglimmer", "Muse")):
        m[f"GSixteen{k}"] = f"{f3(R[a]['own'])}/{f3(R[a]['other'])}"
    m["GSixteenOone"] = ci(R["O1_own_minus_other_mean"])
    m["GSixteenClosedAOwn"] = f3(R["claude"]["restart_own"])
    m["GSixteenClosedAOther"] = f3(R["claude"]["restart_other"])
p = ROOT / "results/t1/g19_own_vs_outside_claude.json"
if p.exists():
    R = json.loads(p.read_text())
    m.update({"GNnOwn": f3(R["within_tool"]["own_8"]), "GNnOut": f3(R["within_tool"]["outside_8"]),
              "GNnDiff": ci(R["O1_own_minus_outside_8"])})
p = ROOT / "results/t1/g20_questionnaire_vs_judge.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GTwJone"] = ci(R["J1_read_out_minus_judge_mean"])
    for a, k in (("gemma4", "Gemma"), ("gptoss20b", "Gpt"), ("qwen38", "Qwen"), ("museglimmer", "Muse")):
        m[f"GTwJudge{k}"] = f3(R[a]["within"]["judge"])
    m["GTwClosedAJudge"] = f3(R["claude"]["restart_judge"])
    m["GTwClosedAPooledJudge"] = f3(R["claude"]["pooled_judge"])
p = ROOT / "results/t1/g21_tau2_open_answerer.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GTwoneGemma"] = f"{f3(R['within_tool']['gemma4_8'])} [{f3(R['A1_gemma4_8_ci'][0])}, {f3(R['A1_gemma4_8_ci'][1])}]"
    m["GTwoneOwn"] = f3(R["within_tool"]["claude_own_8"])
    m["GTwoneDiff"] = ci(R["A2_own_minus_gemma4"])
    m["GTwoneSingle"] = f3(R["within_tool"]["gemma4_single"])
p = ROOT / "results/t1/g25_fisher_snr_own.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GTfFone"] = f"{f3(R['F1_mean_spearman_snr_vs_auroc'][0])} [{f3(R['F1_mean_spearman_snr_vs_auroc'][1])}, {f3(R['F1_mean_spearman_snr_vs_auroc'][2])}]"
p = ROOT / "results/t1/g27_state_tomography.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GTsForecast"] = f3(R["within_stratum"]["forecast_steps_after_first"])
    m["GTsPrior"] = f3(R["prior_only_after_first"])
    m["GTsKone"] = ci(R["K1_forecast_minus_prior_after_first"])
    m["GTsKtwo"] = ci(R["K2_filtered_minus_raw"])
    m["GTsFiltered"] = f3(R["within_stratum"]["filtered"])
    m["GTsRaw"] = f3(R["within_stratum"]["raw"])
    m["GTsNis"] = f"{R['K3_nis_mean_over_K']:.2f}"
p = ROOT / "results/t1/g28_tomography_vs_truth.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GTeFiltered"] = f3(R["rsa_with_true_state"]["filtered"])
    m["GTeRaw"] = f3(R["rsa_with_true_state"]["raw"])
    m["GTeForecast"] = f3(R["rsa_with_true_state"]["forecast"])
    m["GTeTone"] = ci(R["T1_filtered_minus_raw"])
    m["GTeSteps"] = str(R["steps_with_true_state"])
p = ROOT / "results/t1/g27b_replication.json"
if p.exists():
    R = json.loads(p.read_text())
    if "K1_pooled_forecast_minus_prior_after_first" in R:
        m["GTbKone"] = ci(R["K1_pooled_forecast_minus_prior_after_first"])
        m["GTbKtwo"] = ci(R["K2_pooled_filtered_minus_raw"])
p = ROOT / "results/t1/g29_live_forecast_gate.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GTnRed"] = ci(R["P1_relative_reduction"], sign=False)
    m["GTnSucc"] = ci(R["P2_success_difference"])
    m["GTnNone"] = f"{100 * R['violating_episodes']['none']:.1f}\\%"
    m["GTnPre"] = f"{100 * R['violating_episodes']['pre']:.1f}\\%"
    m["GTnReviews"] = f"{R['reviews_per_episode']:.2f}"
p = ROOT / "results/t1/g17_agentdojo_readout.json"
if p.exists():
    R = json.loads(p.read_text())
    for a, k in (("gemma4", "Gemma"), ("gptoss20b", "Gpt")):
        if a in R:
            m[f"GSv{k}"] = f"{f3(R[a]['within_suite']['read_out_8'])} [{f3(R[a]['read_out_8_ci'][0])}, {f3(R[a]['read_out_8_ci'][1])}]"
            m[f"GSv{k}Single"] = f3(R[a]["within_suite"]["single_answer"])
            m[f"GSv{k}N"] = f"{R[a]['calls']} calls, {R[a]['injected']} injected"
    m["GSvVtwo"] = ci(R["V2_read_out_minus_single_mean"])
p = ROOT / "results/t1/g32_where_it_works.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GCwCone"] = ci(R["C1_spearman_snr_auroc"], sign=False)
    m["GCwCtwo"] = ci(R["C2_auroc_snr_pos_minus_nonpos"])
    m["GCwCells"] = str(R["cells"])
    m["GCwPos"] = f"{f3(R['mean_auroc_snr_pos'])} ({R['cells_snr_pos']} cells)"
    m["GCwNeg"] = f"{f3(R['mean_auroc_snr_nonpos'])} ({R['cells_snr_nonpos']} cells)"
    m["GCwPartial"] = f3(R["reported"]["partial_spearman_snr_auroc_given_n_baserate"])
    m["GCwCeil"] = f"{100 * R['reported']['share_cells_within_ceiling_plus_0.05']:.0f}\\%"
    T = R["target_level"]
    m["GCwSnrTau"] = f"{T['tau2_claude']['snr']:.2f}".replace("-", "$-$")
    m["GCwSnrCape"] = f"{T['cape_claude']['snr']:.2f}".replace("-", "$-$")
    m["GCwSnrMinOpen"] = f"{min(v['snr'] for k, v in T.items() if 'claude' not in k):.2f}"
p = ROOT / "results/t1/g34_triangulation.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GVtMae"] = f"{f3(R['V1_mean_abs_error'][0])} [{f3(R['V1_mean_abs_error'][1])}, {f3(R['V1_mean_abs_error'][2])}]"
    m["GVtInside"] = str(R["V2_inside_interval"])
    m["GVtClosed"] = f"{f3(R['E_closed']['predicted'][0])} [{f3(R['E_closed']['predicted'][1])}, {f3(R['E_closed']['predicted'][2])}]"
    o = R["open_own_rho_sampled_design"].values()
    m["GVtOpenRange"] = f"{min(o):.2f}--{max(o):.2f}"
p = ROOT / "results/t1/g38_gpt_readout.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GGlAt"] = f"{f3(R['A1_at_call'][0])} [{f3(R['A1_at_call'][1])}, {f3(R['A1_at_call'][2])}]"
    m["GGlPre"] = f"{f3(R['P1_pre_call'][0])} [{f3(R['P1_pre_call'][1])}, {f3(R['P1_pre_call'][2])}]"
    m["GGlN"] = f"{R['calls']} calls, {R['violations']} violations"
    m["GGlAtShort"] = f"{R['A1_at_call'][0]:.2f}"
    m["GGlPreShort"] = f"{R['P1_pre_call'][0]:.2f}"
p = ROOT / "results/t1/g38_position_exploratory.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GGlPos"] = f3(R["position_alone_within_template_tool"])
    m["GGlPrePos"] = f"{f3(R['PRE_within_template_tool_position'][0])} [{f3(R['PRE_within_template_tool_position'][1])}, {f3(R['PRE_within_template_tool_position'][2])}]"
    m["GGlPrePosShort"] = f"{R['PRE_within_template_tool_position'][0]:.2f}"
    m["GGlPosStrata"] = f"{R['strata_with_both_classes']} strata, {R['violations_in_them']} violations"
p = ROOT / "results/t1/g39_gpt_live_gate.json"
if p.exists():
    R = json.loads(p.read_text())["decidable"]
    m["GLgNone"] = f"{100 * R['violating_episodes']['none']:.1f}\\%"
    m["GLgPre"] = f"{100 * R['violating_episodes']['pre']:.1f}\\%"
    m["GLgPos"] = f"{100 * R['violating_episodes']['position']:.1f}\\%"
    m["GLgRed"] = ci(R["P1_pre_relative_reduction"], sign=False)
    m["GLgPmP"] = ci(R["P3_pre_minus_position"])
    m["GLgSucc"] = ci(R["P2_success_pre_minus_none"])
p = ROOT / "results/t1/g40_gpt_precall_confirm.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GCfPos"] = f"{f3(R['C1_within_template_tool_position'][0])} [{f3(R['C1_within_template_tool_position'][1])}, {f3(R['C1_within_template_tool_position'][2])}]"
    m["GCfTT"] = f"{f3(R['C2_within_template_tool'][0])} [{f3(R['C2_within_template_tool'][1])}, {f3(R['C2_within_template_tool'][2])}]"
    m["GCfPosAlone"] = f3(R["position_alone_within_template_tool"])
    m["GCfN"] = f"{R['episodes']} fresh episodes, {R['units']} calls, {R['violations']} violations"
    m["GCfPosShort"] = f"{R['C1_within_template_tool_position'][0]:.2f}"
p = ROOT / "results/t1/g39b_matched_gate.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GLbNone"] = f"{100 * round(R['violating_episodes']['none'] * R['units']) / R['units']:.1f}\\%"
    m["GLbPre"] = f"{100 * R['violating_episodes']['pre']:.1f}\\%"
    m["GLbPos"] = f"{100 * R['violating_episodes']['position']:.1f}\\%"
    m["GLbRed"] = ci(R["P1_pre_relative_reduction"], sign=False)
    m["GLbPosRed"] = ci(R["position_relative_reduction"], sign=False)
    m["GLbPmP"] = ci(R["P3_pre_minus_position"])
    m["GLbSucc"] = ci(R["P2_success_pre_minus_none"])
    m["GLbRev"] = f"{R['reviews_per_episode']['pre']:.2f} and {R['reviews_per_episode']['position']:.2f}"
p = ROOT / "results/t1/g42_gate_law.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GLwPred"] = f3(R["predicted"]["pre"])
    m["GLwObs"] = f3(R["observed"]["pre"])
p = ROOT / "results/t1/g45_closed_self_forecast.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GFfGptPos"] = f3(R["gpt"]["own_position"])
    m["GFfGptN"] = f"{R['gpt']['units']:,}".replace(",", "{,}") + f" calls, {R['gpt']['violations']} violations"
    m["GFfGptOwnTT"] = f3(R["gpt"]["own_template_tool"])
    m["GFfGptOthTT"] = f3(R["gpt"]["other_template_tool"])
    m["GFfClaudeOwnTT"] = f3(R["claude"]["own_template_tool"])
p = ROOT / "results/t1/g46_rollout_forecast.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GRoN"] = f"{R['units']:,}".replace(",", "{,}") + f" calls, {R['violations']} violations"
    m["GRoOwn"] = ci(R["R1_own_within_position"], sign=False)
    m["GRoReport"] = f3(R["within_position"]["report"])
    m["GRoCeil"] = f3(R["within_position"]["ceiling"])
    m["GRoOwnMinusReport"] = ci(R["R3_own_minus_report_position"])
    if "other" in R["within_position"]:
        m["GRoOther"] = f3(R["within_position"]["other"])
        m["GRoOtherTT"] = f3(R["within_template_tool"]["other"])
        m["GRoOwnTT"] = f3(R["within_template_tool"]["own"])
        m["GRoOwnMinusOther"] = ci(R["R2_own_minus_other_template_tool"])
p = ROOT / "results/t1/g46b_policy_gap.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GPgObs"] = ci(R["B1_observed_excess"])
    m["GPgPred"] = ci(R["predicted_excess"])
    m["GPgDiff"] = ci(R["B3_observed_minus_predicted"])
    m["GPgGap"] = ci(R["B2_policy_gap"])
p = ROOT / "results/t1/g47_selfgate.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GSgNone"] = f"{100 * R['violating_episodes']['none']:.1f}\\%"
    m["GSgSelf"] = f"{100 * R['violating_episodes']['self']:.1f}\\%"
    m["GSgRand"] = f"{100 * R['violating_episodes']['random']:.1f}\\%"
    m["GSgRed"] = ci(R["self_reduction"], sign=False)
    m["GSgDiff"] = ci(R["self_minus_random"])
    m["GSgSucc"] = ci(R["success_self_minus_none"])
    m["GSgHold"] = f"{R['holds_per_call']['self']:.2f} and {R['holds_per_call']['random']:.2f}"
p = ROOT / "results/t1/g49_forecast_steer.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GFsNone"] = f"{100 * R['violating_episodes']['none']:.1f}\\%"
    m["GFsFc"] = f"{100 * R['violating_episodes']['forecast']:.1f}\\%"
    m["GFsRand"] = f"{100 * R['violating_episodes']['random']:.1f}\\%"
    m["GFsRed"] = ci(R["F1_forecast_reduction"], sign=False)
    m["GFsDiff"] = ci(R["F2_forecast_minus_random"])
    m["GFsSucc"] = ci(R["F3_success_forecast_minus_none"])
p = ROOT / "results/t1/g52_text_judge_vs_policy.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GJdOwn"] = ci(R["own_within_position"], sign=False)
    m["GJdJudge"] = ci(R["judge_within_position"], sign=False)
    m["GJdDiff"] = ci(R["J1_own_minus_judge_position"])
p = ROOT / "results/t1/g51_gpt_selfgate.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GSgGNone"] = f"{100 * R['violating_episodes']['none']:.1f}\\%"
    m["GSgGSelf"] = f"{100 * R['violating_episodes']['self']:.1f}\\%"
    m["GSgGRand"] = f"{100 * R['violating_episodes']['random']:.1f}\\%"
    m["GSgGRed"] = ci(R["P1_self_reduction"], sign=False)
    m["GSgGDiff"] = ci(R["P2_self_minus_random"])
    m["GSgGSucc"] = ci(R["P3_success_self_minus_none"])
p = ROOT / "results/t1/g54_gpt_text_judge.json"
if p.exists():
    R = json.loads(p.read_text())
    m["GGjOwn"] = ci(R["own_within_position"], sign=False)
    m["GGjJudge"] = ci(R["judge_within_position"], sign=False)
    m["GGjDiff"] = ci(R["J1_own_minus_judge_position"])
lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
(ROOT / "paper/second_numbers.tex").write_text("% generated by paper/make_second_numbers.py\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
