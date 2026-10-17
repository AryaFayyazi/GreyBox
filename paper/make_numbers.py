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
    out += [r"\newcommand{\PREVENT}{episodes with an executed violation %.1f\%% $\rightarrow$ %.1f\%%}" % (100 * u, 100 * g),
            r"\newcommand{\PREVENTTAB}{%.1f\%% $\rightarrow$ %.1f\%% of episodes}" % (100 * u, 100 * g),
            r"\newcommand{\PREVENTOUT}{%s}" % ("pass" if d["P1_pass"] else "fail"),
            r"\newcommand{\LATENCY}{%d~ms (median; 95th percentile %d~ms)}" % (round(lat["roundtrip_median"]), round(lat["roundtrip_p95"])),
            r"\newcommand{\TEASERE}{ (e) Live prevention in the loop of the two closed agents (T16, T16b): the share of episodes with an executed violation, without and with the gate.}",
            r"\newcommand{\PREVENTTEXT}{The share of episodes with an executed violation fell from %.1f\%% to %.1f\%% (%+.3f [%.3f, %.3f]; P1 %s). The gate blocked %d actions, of which %.0f\%% would have been violations; in safe scenarios it blocked %.2f actions per episode. The agent executed %d violations after a block. Task goals were reached in %.1f\%% (no gate) and %.1f\%% (gate) of the episodes; in these scenarios a goal can require the unauthorized action. The server round trip was %d~ms (median).}"
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
            r"\newcommand{\PREVENTBTEXT}{On Claude Haiku~5.5 (T16b, %d scenarios $\times$ 2 arms, the cut seeded from Claude's past unlabelled traffic), the share of episodes with an executed violation went from %.1f\%% to %.1f\%% (%+.3f [%.3f, %.3f]; P1b %s). The gate blocked %d actions, of which %.0f\%% would have been violations, and the agent executed %d violations after a block.}"
            % (d["scenarios_complete"], 100 * u, 100 * g, ci[0], ci[1], ci[2], "passes" if d["P1_pass"] else "fails", d["blocked"]["actions"], 100 * d["blocked"]["precision"], d["unsafe_executed_after_a_block"])]
else:
    out += [r"\newcommand{\PREVENTBTAB}{\PENDING{running}}", r"\newcommand{\PREVENTBOUT}{--}", r"\newcommand{\PREVENTBTEXT}{\PENDING{T16b on Claude, preregistered, running}}"]
(ROOT / "paper/numbers.tex").write_text("\n".join(out) + "\n")
print("numbers.tex:", "T16" if p.exists() else "pending")
