"""Writes paper/tab_forecast.tex: forecasting a closed agent's coming violation before the call, every forecaster on the
same calls. Claude Haiku 5.5: the original exact-context set (G52, G55, G57, G58, G59, G61) and the fresh prospective
replication (G65); GPT-6-luna: G45/G48/G54/G57 and the exploratory both-strata table. Values from results/t1 only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = lambda f: json.loads((ROOT / f"results/t1/{f}.json").read_text()) if (ROOT / f"results/t1/{f}.json").exists() else None  # noqa: E731


def c(v):
    if v is None:
        return "--"
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return f"{v[0]:.2f} {{\\tiny[{v[1]:.2f}, {v[2]:.2f}]}}"


g52, g55, g57, g58, g59, g61 = J("g52_text_judge_vs_policy"), J("g55_gpt_judges_claude"), J("g57_geometric_forecast"), J("g58_other_reads_draws"), J("g59_selfreport_exact"), J("g61_draws_add_to_text")
ox = J("own_vs_other_exact_exploratory")
g77 = J("g77_fill_cells") or {}
G7 = lambda k: (c([g77[k]["auroc"]] + g77[k]["ci"]).split(" ")[0] + "$^*$") if k in g77 else "n.r."  # noqa: E731
g45, g54, bs, g65, g66 = J("g45_per_agent_ci"), J("g54_gpt_text_judge"), J("forecasters_both_strata_exploratory"), J("g65_replication"), J("g66_geometry_replication")
rv = (g65 or {}).get("values", {})
R = lambda k: rv.get(k) if g65 else None  # noqa: E731
gb = bs["gpt"]["scores"]
rows = [
    ("\\multicolumn{4}{@{}l}{\\cellcolor{gbgroup}\\scriptsize\\textsc{among calls at the same step}}", None),
    ("says: self-report", (c(g59["self_report_within_position"]), c(R("self_report")), c(g45["own_within_position"]["gpt"]))),
    ("text: the agent reasoning over its transcript", (c(g52["judge_within_position"]), c(R("claude_judge")), c(g54["judge_within_position"]))),
    ("text: GPT-6-luna reasoning over it", (c(g55["gpt_judge_within_position"]), c(R("gpt_judge")), "=")),
    ("does: own sampled actions, own read", (c(g52["own_within_position"]), c([g65["endpoints"]["P5_own_within_position"]["estimate"]] + g65["endpoints"]["P5_own_within_position"]["ci"]) if g65 else "--", c(g54["own_within_position"]))),
    ("does: same actions, read by GPT-6-luna", (c(g58["gpt_reads_claude_draws_within_position"]), c(R("gpt_reads_draws")), "=")),
    ("other: Gemma-4 sampling from its contexts$^*$", (c(ox["g46c_exact"]["within_position"]["other_gemma4"]), G7("O65_within_step"), G7("C2_gpt_other_gemma4_within_step"))),
    ("does: same actions, geometry only", (c(g57["claude"]["G1_geometric_forecast_within_position"]), c(g66["forecast"]["G1_geometric_forecast_within_position"]) if g66 else "--", c(g57["gpt"]["G1_geometric_forecast_within_position"]))),
    ("\\multicolumn{4}{@{}l}{\\cellcolor{gbgroup}\\scriptsize\\textsc{across steps}}", None),
    ("the step alone", (c(g61["across_steps_auroc"]["position"]), c(ox["g65_across_steps"]["step_alone"]) + "$^*$", c(gb["position alone"]["within_template_tool"]))),
    ("own sampled actions alone", (c(g61["across_steps_auroc"]["draws"]), c(ox["g65_across_steps"]["own_draws"]) + "$^*$", c(gb["own draws, own read"]["within_template_tool"]))),
    ("Gemma-4 sampling from its contexts$^*$", (c(ox["g46c_exact"]["within_template_tool"]["other_gemma4"]), G7("O65_across_steps"), G7("C2_gpt_other_gemma4_across_steps"))),
    ("best text monitor", (c(g61["across_steps_auroc"]["gpt_judge"]), c(R("gpt_judge_across_steps")), c(gb["GPT-6-luna text judge"]["within_template_tool"]))),
    ("best text monitor + own sampled actions", (c(g61["across_steps_auroc"]["gpt_judge+draws"]), c(R("gpt_judge_plus_draws_across_steps")), G7("C1_gpt_judge_plus_draws_across_steps"))),
]
import re
num = lambda x: float(re.match(r"[0-9.]+", x).group(0)) if re.match(r"[0-9.]+", x) else None  # noqa: E731
blocks, cur = [], []
for name, vals in rows:
    if vals is None:
        if cur:
            blocks.append(cur)
        cur = []
    else:
        cur.append(vals)
blocks.append(cur)
best = []
for blk in blocks:
    best.append([max([num(v[j]) for v in blk if num(v[j]) is not None], default=None) for j in range(3)])
body, bi = [], -1
for name, vals in rows:
    if vals is None:
        bi += 1
        body.append(name + "\\\\")
    else:
        cells = []
        for j, v in enumerate(vals):
            x = num(v)
            cells.append(re.sub(r"^([0-9.]+)", lambda mm: "\\best{" + mm.group(1) + "}", v) if x is not None and x == best[bi][j] else v)
        body.append(f"{name} & " + " & ".join(cells) + "\\\\")
n65 = f"{g65['units']:,}".replace(",", "{,}") if g65 else "--"
tex = r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Forecasting the coming violation before the call}, every forecaster on the same calls (AUROC; 95\% interval where measured). Claude Haiku 5.5: unapproved restarts, exact contexts; the original set and a replication on fresh episodes, registered before any existed. ``='': as the row above. n.r.: not run. $^*$Descriptive, not a registered endpoint; the other model is Gemma-4, sampling from the agent's exact contexts.}
\label{tab:forecast}
\resizebox{\columnwidth}{!}{\begin{tabular}{@{}lccc@{}}
\toprule
\rowcolor{gbhead} & \multicolumn{2}{c}{\textbf{Claude Haiku 5.5}} & \textbf{GPT-6-luna}\\
\rowcolor{gbhead}\textbf{Forecaster} & original (1{,}587) & replication (""" + n65 + r""") & (1{,}134)\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(ROOT / "paper/tab_forecast.tex").write_text(tex)
print(tex)
