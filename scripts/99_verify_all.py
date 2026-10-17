#!/usr/bin/env python
"""Verify every preregistration lock and print every headline number from the saved result files.

1. For each results/prereg_*_lock.json: recompute the sha256 of each locked file. A mismatch is reported with its
   documented reason (docs/PREREGISTRATION_RECORD.md), or as UNEXPLAINED.
2. Print the confirmatory and key development numbers straight from results/*.json (nothing is recomputed).
python scripts/99_verify_all.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
EXPLAINED = {   # (lock, file) -> documented reason for an intended later change
    ("prereg_T1_agents_lock.json", "scripts/25_t1_confirm.py"): "addendum 1 added F4/F5 (re-locked in prereg_T1_agents_addendum1_lock.json)",
    ("prereg_L3b_claude_livebench_lock.json", "scripts/41_claude_livebench.py"): "L3b deviation 1: checkpoint/resume after a rate-limit crash",
    ("prereg_L3c_claude_batch_lock.json", "src/graybox/batch.py"): "append-only extension for L3d; the L3c part hashes to the lock (verified)",
    ("prereg_L2_lock.json", "scripts/30_l2_confirm.py"): "L2 deviation 1: clip standardized features at prediction (as in development)",
}



def _monitor_locked_part(text):
    """monitor.py minus the post-lock additions (docstring thresholds note, QuantileGate, TextMonitor)."""
    i, k = text.index("\n\nThresholds: do not cut"), text.index('\n"""', text.index("\n\nThresholds: do not cut"))
    text = text[:i] + text[k:]
    i, k = text.index("class QuantileGate:"), text.index("class ReaderMonitor:")
    return text[:i] + text[k:]


ADDITIVE = {   # (lock, file) -> strip the documented additions; the rest must hash to the lock (checked on every run)
    ("prereg_T7_agentdojo_lock.json", "src/graybox/monitor.py"): _monitor_locked_part,
    ("prereg_T8_claude_agent_lock.json", "src/graybox/monitor.py"): _monitor_locked_part,
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_locks():
    print("== locks")
    bad = 0
    for lk in sorted(R.glob("prereg_*_lock.json")):
        L = json.loads(lk.read_text())
        for f, h in L["sha256"].items():
            p = Path(f) if Path(f).is_absolute() else ROOT / f
            if not p.exists():
                print(f"  MISSING   {lk.name}: {f}"); bad += 1; continue
            if sha(p) == h:
                continue
            add = ADDITIVE.get((lk.name, f))
            if add and hashlib.sha256(add(p.read_text()).encode()).hexdigest() == h:
                print(f"  ADDITIVE  {lk.name}: {f} -> post-lock additions only (QuantileGate, TextMonitor, docstring); locked part verified")
                continue
            why = EXPLAINED.get((lk.name, f))
            print(f"  CHANGED   {lk.name}: {f} -> {why or 'UNEXPLAINED'}")
            bad += why is None
    print(f"  unexplained mismatches: {bad}")


def j(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def numbers():
    print("== headline numbers (from saved results)")
    rows = []
    t1 = j("t1/t1_agents_confirm.json")
    if t1:
        rows += [("T1 agents fresh: AUROC R16 / WB / R4+PSI", f"{t1['auroc']['all']['R16']} / {t1['auroc']['all']['WB']} / {t1['auroc']['all']['R4+PSI']}"),
                 ("T1 F2 R16-WB [CI]", t1["F2_R16_minus_WB"]), ("T1 F4 R4+PSI-WB [CI]", t1["F4_R4PSI_minus_WB"])]
    for name, f, key in (("L3b D2 Claude harmful-only (per-question)", "l3/claude_livebench.json", "D2_harmful_only"),
                         ("L3d F1 Claude harmful-only (transcript)", "l3/claude_transcript.json", "F1_harmful_refusal"),
                         ("L3c E1 Claude harmful-only (in-conversation)", "l3/claude_batch.json", "E1_harmful_refusal")):
        d = j(f)
        if d and key in d:
            e = d[key]
            rows.append((name, f"greybox {e.get('auroc', e.get('greybox_auroc'))} vs moderation {e.get('baseline_auroc', e.get('moderation_auroc'))}; "
                               f"diff {e.get('minus_baseline', [e.get('diff'), e.get('ci95')])}; pass {e.get('pass')}"))
    g = j("t1/guard_heldout.json")
    if g:
        x = g["K"]["16"]["target_esc_safe_0.05"]
        rows.append(("Guard K=16, tau from dev: caught / escalated (fresh)", f"{x['fresh_caught']} {x['fresh_caught_ci95']} / {x['fresh_esc_safe']}"))
    t2 = j("t1/t2_time.json")
    if t2:
        rows.append(("T2 AUROC (I_dec, R16_dec, R16_ts, WB_ts)", {k: t2["auroc"].get(k) for k in ("I_dec", "R16_dec", "R16_ts", "WB_ts")}))
        rows.append(("T2 G1/G2/G3 pass", (t2.get("G1_pass"), t2.get("G2_pass"), t2.get("G3_pass"))))
    e = j("l3/l3e_tool.json")
    if e:
        P = e["primary_qwen3guard_judge"]
        rows.append(("L3e A1 pass / A2 (vs graded QG) / A3", (P["A1"].get("pass"), P["A2"]["qg_p"], P["A3"].get("pass") if isinstance(P.get("A3"), dict) else P.get("A3"))))
    for name, f, keys in (("T3 deployable", "t1/t3_deployable.json", None), ("T4 reader", "t1/t4_reader.json", ("auroc_fresh", "M1_READERact_minus_Idec", "M2_READERact_minus_Jact")),
                          ("T5 families", "t1/t5_family.json", ("auroc", "P1_READER_minus_AGENT_family", "P1_pass")),
                          ("T6 cross-agent", "t1/t6_cross_agent.json", ("auroc_gptoss", "Q1_READER_minus_Jact_gptoss", "Q1_pass")),
                          ("T7 AgentDojo", "t1/t7_agentdojo.json", ("per_actor", "R1_READER_minus_Jact", "R1_pass", "R2_READER_minus_CAPE")),
                          ("T8 Claude post-reveal", "t1/t8_claude_agent.json", ("n_unsafe", "S1_not_tested")),
                          ("T8b Claude all actions", "t1/t8b_claude_all_actions.json", ("auroc", "S1_READER_minus_Jact", "S1_pass", "caught_at_5pct_safe_escalated"))):
        d = j(f)
        if d:
            rows.append((name, {k: d.get(k) for k in keys} if keys else {k: v for k, v in d.items() if "auroc" in k or "pass" in k}))
    d = j("t1/t3_caught_exploratory.json")
    if d:
        rows.append(("T3 caught at 5% safe escalated (exploratory)", d["caught"]))
    d = j("t1/t9_text_baseline.json")
    if d:
        rows.append(("T9 reader vs TF-IDF: B1 (Claude) / B2 (families)", (d["B1_pass"], d["T8b_claude"]["READER_minus_TXT"], d["B2_pass"], d["T5_unseen_families"]["READER_minus_TXT"])))
        rows.append(("T9 AUROC reader/TXT per test", {k: (d[k]["auroc"]["READER"], d[k]["auroc"]["TXT"]) for k in ("T4_same_scenarios", "T5_unseen_families", "T6_gptoss", "T7_agentdojo", "T8b_claude")}))
    d = j("t1/t10_claude_unseen_families.json")
    if d:
        rows.append(("T10 Claude unseen families: C1 / C2 pass; pooled, within AUROC", (d["C1_pass"], d["C2_pass"], d["auroc_pooled"], d["auroc_within"])))
        rows.append(("T10 caught at label-free cut (READER_lofo)", d["label_free_cut"]["READER_lofo"]))
    d = j("t1/t10_new_policy_labels_exploratory.json")
    if d:
        rows.append(("T10 follow-up: restart caught vs labels added (exploratory)", {n: v["cons_service_restart"] for n, v in d["by_n"].items()}))
    d = j("l3/l3c_graded_reanalysis.json")
    if d:
        rows.append(("L3c graded re-analysis: E1 diff vs QG / LG", (d["qg_p"]["E1_refusal"]["diff"], d["qg_p"]["E1_refusal"]["ci95"], d["lg_p"]["E1_refusal"]["diff"], d["lg_p"]["E1_refusal"]["ci95"])))
    d = j("t1/t8b_calibration_exploratory.json")
    if d:
        rows.append(("T8b calibration (exploratory): 0.5 cut / online gate", (d["READER"]["at_0.5"], d["online_QuantileGate_budget0.05"]["run_order"])))
    for k, v in rows:
        print(f"  {k:60s} {v}")


if __name__ == "__main__":
    check_locks()
    numbers()
