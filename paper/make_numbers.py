#!/usr/bin/env python
"""Write paper/numbers.tex from the result files (T16 live prevention). Until T16 exists, the macros print PENDING.
python paper/make_numbers.py"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "results/t1/t16_live_gate.json"
out = []
if p.exists():
    d = json.loads(p.read_text())
    u, g = d["unsafe_episode_rate"]["ungated"], d["unsafe_episode_rate"]["gated"]
    ci = d["P1_gated_minus_ungated"]
    lat = d["latency_ms"]
    latb = json.loads((p.parent / "t16b_claude_live_gate.json").read_text())["latency_ms"]
    out += [r"\newcommand{\PREVENT}{episodes with an executed violation %.1f\%% $\rightarrow$ %.1f\%%}" % (100 * u, 100 * g),
            r"\newcommand{\PREVENTTAB}{%.1f\%% $\rightarrow$ %.1f\%% of episodes}" % (100 * u, 100 * g),
            r"\newcommand{\PREVENTOUT}{%s}" % ("pass" if d["P1_pass"] else "fail"),
            r"\newcommand{\LATENCY}{%d~ms on GPT-6-luna and %d~ms on Claude Haiku 5.5 (median; 95th percentiles %d and %d~ms)}" % (
                round(lat["roundtrip_median"]), round(latb["roundtrip_median"]), round(lat["roundtrip_p95"]), round(latb["roundtrip_p95"])),
            r"\newcommand{\TEASERE}{ (e) Live prevention in the loop of the two closed agents: the share of episodes with an executed violation, without and with the gate.}",
            r"\newcommand{\PREVENTTEXT}{The share of episodes with an executed violation fell from %.1f\%% to %.1f\%% (%+.3f [%.3f, %.3f]; the preregistered test %s). The gate blocked %d actions, of which %.0f\%% would have been violations; in safe scenarios it blocked %.2f actions per episode. The agent executed %d violations after a block. Task goals were reached in %.1f\%% (no gate) and %.1f\%% (gate) of the episodes; in these scenarios a goal can require the unauthorized action. The server round trip was %d~ms (median).}"
            % (100 * u, 100 * g, ci[0], ci[1], ci[2], "passes" if d["P1_pass"] else "fails", d["blocked"]["actions"], 100 * d["blocked"]["precision"],
               d["blocks_per_safe_scenario_gated_arm"], d["unsafe_executed_after_a_block"], 100 * d["goal_rate"]["ungated"], 100 * d["goal_rate"]["gated"], round(lat["roundtrip_median"]))]
else:
    out += [r"\newcommand{\PREVENT}{\PENDING{T16 running}}", r"\newcommand{\PREVENTTAB}{\PENDING{running}}", r"\newcommand{\PREVENTOUT}{--}",
            r"\newcommand{\LATENCY}{\PENDING{T16}}", r"\newcommand{\TEASERE}{}", r"\newcommand{\PREVENTTEXT}{\PENDING{T16 running; preregistered and locked}}"]
pb = ROOT / "results/t1/t16b_claude_live_gate.json"
if pb.exists():
    d = json.loads(pb.read_text())
    u, g, ci = d["unsafe_episode_rate"]["ungated"], d["unsafe_episode_rate"]["gated"], d["P1_gated_minus_ungated"]
    out += [r"\newcommand{\PREVENTBTAB}{%.1f\%% $\rightarrow$ %.1f\%% of episodes}" % (100 * u, 100 * g),
            r"\newcommand{\PREVENTBOUT}{%s}" % ("pass" if d["P1_pass"] else "fail"),
            r"\newcommand{\PREVENTBTEXT}{On Claude Haiku 5.5 (%d scenarios $\times$ 2 arms, the cut seeded from Claude Haiku 5.5's past unlabelled traffic), the share of episodes with an executed violation went from %.1f\%% to %.1f\%% (%+.3f [%.3f, %.3f]; the preregistered test %s). The gate blocked %d actions, of which %.0f\%% would have been violations, and the agent executed %d violations after a block.}"
            % (d["scenarios_complete"], 100 * u, 100 * g, ci[0], ci[1], ci[2], "passes" if d["P1_pass"] else "fails", d["blocked"]["actions"], 100 * d["blocked"]["precision"], d["unsafe_executed_after_a_block"])]
else:
    out += [r"\newcommand{\PREVENTBTAB}{\PENDING{running}}", r"\newcommand{\PREVENTBOUT}{--}", r"\newcommand{\PREVENTBTEXT}{\PENDING{T16b on Claude Haiku 5.5, preregistered, running}}"]
p17 = ROOT / "results/t1/t17_replication_agents.json"
if p17.exists():
    d = json.loads(p17.read_text())
    A = [("gpt-oss-20b", "gptoss20b"), ("Qwen3.8-27B", "qwen38"), ("Muse-Glimmer-30B", "museglimmer")]
    r2 = "/".join("%.2f" % d["agents"][k]["r2_fresh"]["READER"] for _, k in A)
    tx = "/".join("%.2f" % d["agents"][k]["r2_fresh"]["TXT"] for _, k in A)
    nb = d["n_agents_both_pass"]
    out += [r"\newcommand{\REPLAGENTSTAB}{$R^2$ %s (text %s); readout holds}" % (r2, tx),
            r"\newcommand{\REPLAGENTSOUT}{%s (%d/3)}" % ("pass" if nb == 3 else "partial" if d["E1_all_pass"] else "fail", nb),
            r"\newcommand{\REPLAGENTSTEXT}{%s}" % " ".join(
                "%s: $R^2$ %.3f vs text %.3f (%+.3f [%.3f, %.3f]); the probe on true states %.3f, on the replica %.3f (%+.3f [%.3f, %.3f])."
                % (n, d["agents"][k]["r2_fresh"]["READER"], d["agents"][k]["r2_fresh"]["TXT"], *d["agents"][k]["E1_R2_READER_minus_TXT"],
                   d["agents"][k]["E2_auroc"]["true"], d["agents"][k]["E2_auroc"]["READER_replica"], *d["agents"][k]["E2_replica_minus_true"]) for n, k in A),
            r"\newcommand{\REPLGEMMASPLIT}{%.3f (text %.3f)}" % (d["exploratory_gemma4_same_split"]["r2_fresh"]["READER"], d["exploratory_gemma4_same_split"]["r2_fresh"]["TXT"])]
else:
    out += [r"\newcommand{\REPLAGENTSTAB}{\PENDING{running}}", r"\newcommand{\REPLAGENTSOUT}{--}", r"\newcommand{\REPLAGENTSTEXT}{\PENDING{T17}}",
            r"\newcommand{\REPLGEMMASPLIT}{\PENDING{T17}}"]
p16c = ROOT / "results/t1/t16c_review_gate.json"
if p16c.exists():
    d = json.loads(p16c.read_text())
    x = json.loads((ROOT / "results/t1/t16c_goal_decomposition_exploratory.json").read_text())
    v, gl, rv = d["R1_review_minus_ungated_violation"], d["R2_review_minus_ungated_goal"], d["reviews"]
    ha, nh = x["held_and_approved_no_block"], x["never_held"]
    out += [r"\newcommand{\REVIEWTAB}{%.1f\%% $\rightarrow$ %.1f\%%; goals %.1f\%% vs %.1f\%%}" % (100 * d["unsafe_episode_rate"]["ungated_T16b"], 100 * d["unsafe_episode_rate"]["review"], 100 * d["goal_rate"]["review"], 100 * d["goal_rate"]["ungated_T16b"]),
            r"\newcommand{\REVIEWOUT}{R1 %s, R2 %s}" % ("pass" if d["R1_pass"] else "fail", "pass" if d["R2_pass"] else "fail"),
            r"\newcommand{\REVIEWTEXT}{On the same %d Claude Haiku 5.5 scenarios, episodes with an executed violation went from %.1f\%% (no gate) to %.1f\%% (%+.3f [%.3f, %.3f]; the preregistered safety test %s), with %d violations after a block. Goals were reached in %.1f\%% of the episodes, against %.1f\%% without the gate and %.1f\%% with blocking; the preregistered non-inferiority (at most 3 points lost) is %s (%+.3f [%.3f, %+.3f]). Exploratory: episodes with a held action that the reviewer let run kept their goals (%.1f\%% vs %.1f\%%, %d episodes); the gap is in the %d episodes in which nothing was held (%.1f\%% vs %.1f\%%), where the monitor did nothing; we suggest this is variation between runs, since the control was not concurrent, but this is a hypothesis that T16c does not test. The reviewer saw %.2f actions per episode (%.1f\%% of the scored actions); %d of the %d held actions were violations.}"
            % (d["scenarios"], 100 * d["unsafe_episode_rate"]["ungated_T16b"], 100 * d["unsafe_episode_rate"]["review"], *v, "passes" if d["R1_pass"] else "fails", rv["violations_executed_after_a_block"],
               100 * d["goal_rate"]["review"], 100 * d["goal_rate"]["ungated_T16b"], 100 * d["goal_rate"]["gated_T16b"], "shown" if d["R2_pass"] else "not shown", *gl,
               100 * ha["goal_review"], 100 * ha["goal_same_instances_ungated"], ha["episodes"], nh["episodes"], 100 * nh["goal_review"], 100 * nh["goal_same_instances_ungated"],
               rv["per_episode"], 100 * rv["share_of_scored_actions"], rv["held_that_were_violations"], rv["held_actions"])]
else:
    out += [r"\newcommand{\REVIEWTAB}{\PENDING{running}}", r"\newcommand{\REVIEWOUT}{--}", r"\newcommand{\REVIEWTEXT}{\PENDING{T16c}}"]
UNUSED = ("\\PREVENTTEXT", "\\REVIEWTEXT", "\\REVIEWOUT")   # report-era macros: not used by the paper; dropped so their ids cannot leak
out = [x for x in out if not any(x.startswith("\\newcommand{" + u + "}") for u in UNUSED)]
(ROOT / "paper/numbers.tex").write_text("\n".join(out) + "\n")
print("numbers.tex:", "T16" if p.exists() else "pending")
