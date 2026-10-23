#!/usr/bin/env python
"""Write paper/geometry_numbers.tex from the G-series result files (G1, G2, G3, G5, the cross-fit exploratory run).
Until a file exists, its macros print PENDING.  python paper/make_geometry_numbers.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T1 = ROOT / "results/t1"
NAMES = {"gemma3_4b": "Gemma-3 4B", "gptoss20b": "gpt-oss 20B", "llama31_8b": "Llama-3.1 8B", "mistral7b": "Mistral 7B",
         "olmo3_7b": "OLMo-3 7B", "phi4mini": "Phi-4 mini", "qwen25_7b": "Qwen2.5 7B", "qwen3_8b": "Qwen3 8B"}
out = []


def J(name):
    p = T1 / name
    return json.loads(p.read_text()) if p.exists() else None


def mac(name, val):
    out.append(r"\newcommand{\%s}{%s}" % (name, val))


def rng(vals, fmt="%.2f"):
    return (fmt % min(vals)) + "--" + (fmt % max(vals))


def ci(x, fmt="%.3f", sign=True):
    f = ("%+" + fmt[1:]) if sign else fmt
    return (f % x[0]) + r" {\scriptsize[" + (fmt % x[1]) + ", " + (fmt % x[2]) + "]}"


wk = J("within_kind_exploratory.json")
g6 = J("g6_shared_geometry.json")
g7 = J("g7_channels.json")
g1, g2, g3, g5, cf = J("g1_open_geometry.json"), J("g2_claude_geometry.json"), J("g3_exact_recovery.json"), J("g5_answers_vs_replies.json"), J("crossfit_exploratory.json")
if g1:
    M = g1["models"]
    mac("GOneAns", rng([m["rho_answers"] for m in M.values()]))
    mac("GOneText", rng([m["rho_text"] for m in M.values()]))
    mac("GOneGain", "%+.2f to %+.2f" % (min(m["F1_answers_minus_text"][0] for m in M.values()), max(m["F1_answers_minus_text"][0] for m in M.values())))
    mac("GOneGainLow", "%.3f" % min(m["F1_answers_minus_text"][1] for m in M.values()))
    mac("GOneKeep", rng([100 * m["F2_sampled_over_exact"][0] for m in M.values()], "%.0f") + r"\%")
    mac("GOneKeepLow", "%.3f" % min(m["F2_sampled_over_exact"][1] for m in M.values()))
    mac("GOneMore", "%+.2f to %+.2f" % (min(m["F3_full_minus_8"][0] for m in M.values()), max(m["F3_full_minus_8"][0] for m in M.values())))
    rows = []
    order = sorted(M, key=lambda k: -M[k]["rho_answers"])
    for k in order:
        m = M[k]
        rep = r"\PENDING{G5}"
        if g5 and k in g5["models"]:
            rep = "%.2f" % g5["models"][k]["rho_replies_mean8"]
        vals = [m["rho_answers"], m["rho_sampled_n8"], g5["models"][k]["rho_replies_mean8"] if g5 and k in g5["models"] else None, m["rho_text"]]
        top = max(round(v, 2) for v in vals if v is not None)
        cell = lambda v: "--" if v is None else (r"\best{%.2f}" % v if round(v, 2) == top else "%.2f" % v)  # noqa: E731
        g = m["F1_answers_minus_text"]
        rows.append(r"%s%s & %s & %s & %s & %s & \textbf{%+.2f} {\scriptsize\textcolor{gbmute}{[%.2f, %.2f]}}\\" % (
            r"\rowcolor{gbrow}" if len(rows) % 2 else "", NAMES[k], *[cell(v) for v in vals], g[0], g[1], g[2]))
    mac("GTABLE", "\n".join(rows))
else:
    for n in ("GOneAns", "GOneText", "GOneGain", "GOneGainLow", "GOneKeep", "GOneKeepLow", "GOneMore", "GTABLE"):
        mac(n, r"\PENDING{G1}")
if g2:
    mac("GTwoSplit", "%.2f" % g2["C1_split_half_rho"][0] + r" {\scriptsize[%.2f, %.2f]}" % tuple(g2["C1_split_half_rho"][1:]))
    mac("GTwoText", ci(g2["C2_split_half_minus_text"], "%.2f"))
    mac("GTwoRhoText", "%.2f" % g2["rho_claude_full_vs_text"])
    mac("GTwoOpen", rng(list(g2["reported_only_claude_answers_vs_open_hidden"].values())))
else:
    for n in ("GTwoSplit", "GTwoText", "GTwoRhoText", "GTwoOpen"):
        mac(n, r"\PENDING{G2}")
if cf:
    mac("CFClosedANeg", r"%.0f\%%" % (100 * cf["claude"]["neg_mds_subtract"]))
    mac("CFClosedANegCF", r"%.0f\%%" % (100 * cf["claude"]["neg_mds_crossfit"]))
else:
    mac("CFClosedANeg", r"\PENDING{}"); mac("CFClosedANegCF", r"\PENDING{}")
if g3:
    b = g3["by_N"]
    mac("GThreeBelow", "%.2f and %.2f (at $0.25\\times$ and $0.5\\times$)" % (b["0.25x"]["pearson_cos"], b["0.5x"]["pearson_cos"]))
    mac("GThreeOneOne", "%.4f" % b["1.1x"]["pearson_cos"])
    mac("GThreeErr", r"%.0f\times10^{-6}" % (1e6 * b["2.0x"]["median_abs_err_cos"]))
    mac("GThreeMax", r"%.1f\times10^{-4}" % (1e4 * b["2.0x"]["max_abs_err_cos"]))
    mac("GThreeErrBelow", "%.2f" % min(v["median_abs_err_cos"] for k, v in b.items() if float(k[:-1]) < 1))
    mac("GThreeBase", "%.2f and %.2f" % (g3["baseline_raw_centered_logprob_pearson"], g3["baseline_whitened_no_constraint_pearson"]))
    mac("GThreeN", "{:,}".format(b["2.0x"]["N"]).replace(",", "{,}"))
else:
    for n in ("GThreeBelow", "GThreeOneOne", "GThreeErr", "GThreeMax", "GThreeErrBelow", "GThreeBase", "GThreeN"):
        mac(n, r"\PENDING{G3}")
if g5:
    M = g5["models"].values()
    mac("GFiveRep", rng([m["rho_replies_mean8"] for m in M]))
    mac("GFiveAns", rng([m["rho_answers_exact88"] for m in M]))
    mac("GFiveOne", rng([m["rho_reply_1"] for m in M]))
    mac("GFiveHOne", "%d of 8" % sum(m["H1_exact88_minus_replies8"][1] > 0 for m in M))
    mac("GFiveHTwo", "%d of 8" % sum(m["H2_1sample88_minus_reply1"][1] > 0 for m in M))
    mac("GFiveHThree", "%d of 8" % sum(m["H3_replies8_minus_reply1"][2] < 0.05 for m in M))
    mac("GFiveSat", "%.3f" % max(m["H3_replies8_minus_reply1"][2] for m in M))
    mac("GFiveSatPoint", "%.3f" % max(m["H3_replies8_minus_reply1"][0] for m in M))
    mac("GFiveOut", "H1 %s, H2 %s, H3 %s" % tuple("passes" if g5[k] else "fails" for k in ("H1_pass", "H2_pass", "H3_pass")))
else:
    for n in ("GFiveRep", "GFiveAns", "GFiveOne", "GFiveHOne", "GFiveHTwo", "GFiveHThree", "GFiveSat", "GFiveOut"):
        mac(n, r"\PENDING{G5 running}")
if wk:
    mac("WKCells", "%d of %d" % (wk["answers_beat_text_cells"], wk["cells"]))
    mac("WKAns", "%.2f" % wk["median_within_kind"]["answers"])
    mac("WKText", "%.2f" % wk["median_within_kind"]["text"])
    mac("WKRep", ("%d of %d" % (wk["answers_beat_replies_cells"], wk["cells"])) if wk.get("answers_beat_replies_cells") is not None else r"\PENDING{G5}")
else:
    for n in ("WKCells", "WKAns", "WKText", "WKRep"):
        mac(n, r"\PENDING{}")
if g6:
    mac("GSixExcess", "%+.3f" % g6["S1_point"] + r" {\scriptsize[%+.3f, %+.3f]}" % tuple(g6["S1_diagonal_excess"][1:]))
    mac("GSixSOne", "passes" if g6["S1_pass"] else "fails")
    s2 = g6["S2_reader_hidden_minus_own_answers"]
    mac("GSixSTwo", "%d of %d" % (sum(v[1] > 0 for v in s2.values()), len(s2)))
    mac("GSixSTwoRange", "%+.2f to %+.2f" % (min(v[0] for v in s2.values()), max(v[0] for v in s2.values())))
    s3 = g6["S3_new_kinds_answers_minus_text"]
    mac("GSixSThree", "%d of %d" % (sum(v[1] > 0 for v in s3.values()), len(s3)))
    rp = g6["reported"]
    mac("GSixAns", rng([v["answers"] for v in rp.values()]))
    mac("GSixText", rng([v["text"] for v in rp.values()]))
    mac("GSixRep", rng([v["replies"] for v in rp.values() if "replies" in v]))
    mac("GSixAnsNew", rng([v["answers_new_kinds"] for v in rp.values()]))
    mac("GSixTextNew", rng([v["text_new_kinds"] for v in rp.values()]))
    hh = g6["reported_hidden_vs_hidden"]
    off = [hh[a][b] for a in hh for b in hh if a != b]
    mac("GSixHH", rng(off))
    mac("GSixReader", rng([v["reader_hidden"] for v in rp.values() if v["reader_hidden"] is not None]))
    R = g6["rho_answers_A_vs_hidden_B"]
    mac("GSixOwnBest", "%d of 8" % sum(R[a][a] == max(R[a].values()) for a in R))
    mac("GSixOut", "S1 %s, S2 %s, S3 %s" % tuple("passes" if g6[k] else "fails" for k in ("S1_pass", "S2_pass", "S3_pass")))
else:
    for n in ("GSixExcess", "GSixSOne", "GSixSTwo", "GSixSTwoRange", "GSixSThree", "GSixAns", "GSixText", "GSixRep", "GSixAnsNew",
              "GSixTextNew", "GSixHH", "GSixReader", "GSixOwnBest", "GSixOut"):
        mac(n, r"\PENDING{G6}")
if g7:
    LL, CL = g7["channels"]["last_layer"], g7["channels"]["chosen_layer"]
    f = lambda e: "%+.3f" % e[0] + r" {\scriptsize[%+.3f, %+.3f]}" % (e[1], e[2])  # noqa: E731
    for c, nm in (("answers", "Ans"), ("top1", "TopOne"), ("top5", "TopFive"), ("top20", "TopTwenty"), ("full", "Full"), ("logprob", "Logprob")):
        mac("GSevL" + nm, f(LL[c]["excess"]))
        mac("GSevC" + nm, f(CL[c]["excess"]))
        mac("GSevBest" + nm, "%d of 8" % LL[c]["own_best"])
        mac("GSevCBest" + nm, "%d of 8" % CL[c]["own_best"])
    mac("GSevOwnLogprob", "%.2f" % LL["logprob"]["own_mean"])
    mac("GSevOthersLogprob", "%.2f" % LL["logprob"]["others_mean"])
    mac("GSevOut", ", ".join("K%d %s" % (i, "passes" if g7["K%d_pass" % i] else "fails") for i in range(1, 6)))
else:
    for c in ("Ans", "TopOne", "TopFive", "TopTwenty", "Full", "Logprob"):
        for pre in ("GSevL", "GSevC", "GSevBest", "GSevCBest"):
            mac(pre + c, r"\PENDING{G7}")
    for n in ("GSevOwnLogprob", "GSevOthersLogprob", "GSevOut"):
        mac(n, r"\PENDING{G7}")
rf, ag = J("review_followups_exploratory.json"), J("agent_shared_geometry_exploratory.json")
se, sc = J("strong_embedders_exploratory.json"), J("scale_exploratory.json")
if se:
    mac("StrongConfirm", "%d of 8" % sum(v["answers_minus_strongest"][1] > 0 for v in se["confirm"].values()))
    mac("StrongFresh", "%d of 8" % sum(v["answers_minus_strongest"][1] > 0 for v in se["fresh"].values()))
else:
    mac("StrongConfirm", r"\PENDING{}"); mac("StrongFresh", r"\PENDING{}")
if sc and "llama33_70b" in sc["depth75"]["excess_per_model"]:
    a, b = sc["depth75"]["excess_all"], sc["depth75"]["excess_per_model"]
    mac("ScaleAll", "%+.3f" % a[0] + r" {\scriptsize[%+.3f, %+.3f]}" % (a[1], a[2]))
    mac("ScaleLlama", "%+.3f" % b["llama33_70b"])
    mac("ScaleQwen", "%+.3f, %+.3f, %+.3f at 8B, 14B, 32B" % (b["qwen3_8b"], b["qwen3_14b"], b["qwen3_32b"]))
else:
    for n in ("ScaleAll", "ScaleLlama", "ScaleQwen"):
        mac(n, r"\PENDING{}")
if rf:
    pm = rf["g6"]["all"]["per_model"]
    lo, hi = min(pm, key=lambda k: pm[k][0]), max(pm, key=lambda k: pm[k][0])
    mac("GSixPerModel", "per-model excess from %+.3f (%s) to %+.3f (%s)" % (pm[lo][0], NAMES[lo], pm[hi][0], NAMES[hi]))
    e = rf["g6"]["new_kinds"]["pooled"]
    mac("GSixNewExcess", "%+.3f" % e[0] + r" {\scriptsize[%+.3f, %+.3f]}" % (e[1], e[2]))
    o = rf["claude_reference"]["other"]
    mac("ClosedARefOther", "%.2f--%.2f (median %.2f)" % (o[0], o[2], o[1]))
else:
    for n in ("GSixPerModel", "GSixNewExcess", "ClosedARefOther"):
        mac(n, r"\PENDING{}")
if ag:
    v = [x["rho_reader_vs_agent"] for x in ag.values()]
    mac("AgentLink", "%.2f--%.2f" % (min(v), max(v)))
else:
    mac("AgentLink", r"\PENDING{}")
g8 = J("g8_cot_boundary.json")
m0 = len(out)
if g8:
    P, A = g8["pooled_cot_minus_act"], g8["agents"]
    mac("GEightInagent", ci(P["action_inagent"]))
    mac("GEightTransfer", ci(P["action_transfer"]))
    mac("GEightFuture", ci(P["future_inagent"]))
    mac("GEightQwen", ci([A["qwen38"]["action"]["inagent"]["cot_minus_act"]] + A["qwen38"]["action"]["inagent"]["ci"]))
    mac("GEightTransferEach", ", ".join(("%+.4f" if abs(A[a]["action"]["transfer"]["cot_minus_act"]) < 0.001 else "%+.3f") % A[a]["action"]["transfer"]["cot_minus_act"] for a in ("gptoss20b", "qwen38", "museglimmer")))
    lv = [A[a]["future"]["inagent"][v] for a in A for v in ("act", "cot")]
    mac("GEightFutureLevel", "%.2f--%.2f" % (min(lv), max(lv)))
else:
    for n in ("GEightInagent", "GEightTransfer", "GEightFuture", "GEightQwen", "GEightTransferEach", "GEightFutureLevel"):
        mac(n, r"\PENDING{}")
out[m0:] = [o.split("}{", 1)[0] + "}{" + o.split("}{", 1)[1].replace("-", "$-$").replace("$-$$-$", "--") for o in out[m0:]]
(Path(__file__).parent / "geometry_numbers.tex").write_text("\n".join(out) + "\n")
print("wrote", len(out), "macros")
