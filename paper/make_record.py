#!/usr/bin/env python
"""Write paper/appendix_record.tex from the table in docs/PREREGISTRATION_RECORD.md, so the paper's appendix can never
go stale (reviewer finding M5). Run before building the paper: python paper/make_record.py"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNI = {"−": "$-$", "→": "$\\rightarrow$", "≥": "$\\geq$", "≤": "$\\leq$", "×": "$\\times$", "≈": "$\\approx$", "∪": "$\\cup$",
       "²": "$^2$", "τ": "$\\tau$", "α": "$\\alpha$", "σ": "$\\sigma$", "Ψ": "$\\Psi$", "ψ": "$\\psi$", "θ": "$\\theta$", "Δ": "$\\Delta$",
       "·": "$\\cdot$", "…": "\\ldots{}", "–": "--", "—": "---", "‑": "-", "≠": "$\\neq$", "±": "$\\pm$", "μ": "$\\mu$", "ρ": "$\\rho$",
       "✓": "yes", "’": "'", "“": "``", "”": "''", "é": "\\'e", "≪": "$\\ll$", "⊥": "$\\perp$", "∈": "$\\in$", "β": "$\\beta$", "λ": "$\\lambda$", "κ": "$\\kappa$"}


def tex(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("$", "\\$"), ("{", "\\{"), ("}", "\\}"), ("~", "\\~{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    s = s.replace("\\textbackslash\\{\\}", "\\textbackslash{}")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    out = "".join(UNI.get(ch, ch) for ch in s)
    bad = sorted({ch for ch in out if ord(ch) > 127})
    assert not bad, f"unmapped characters: {bad}"
    return out


NAMES = {  # descriptive names for the paper (the released record keeps its ids)
    "L1": "Interrogation state, Othello", "L2": "Interrogation state, open chat models", "L3": "Claude Haiku 5.5 chat readouts",
    "L3b": "Claude Haiku 5.5 refusals vs moderation", "L3c": "Claude Haiku 5.5, one-call grey box", "L3d": "Claude Haiku 5.5, transcript readout",
    "L3e": "Claude Haiku 5.5 vs graded moderators",
    "T1": "Resampling vs a probe", "T2": "When the internals are read", "T3": "Resampling with a deployable judge",
    "T4": "Reader vs the agent's internals", "T5": "Unseen task families", "T6": "Another open agent",
    "T7": "AgentDojo, no labels", "T8": "Claude Haiku 5.5, after a reveal", "T8b": "Closed agent: Claude Haiku 5.5",
    "T9": "Reader vs TF-IDF", "T10": "A policy removed from training", "T11": "Closed agent: GPT-6-luna",
    "T12": "Labels for a new policy", "T13": "AgentDojo with a defense", "T13b": "AgentDojo, closed agent",
    "T14": "Open-agent labels for a closed agent", "T15": "Replication (Gemma-4)", "T16": "Live prevention (GPT)",
    "T16b": "Live prevention (Claude Haiku 5.5)", "T16c": "Hold for review", "T17": "Replication, three architectures",
    "T18": "tau2-bench, Claude Haiku 5.5", "T19": "Browser-size readers", "T20": "tau2-bench, two closed agents",
    "T21": "tau2-bench, unseen tasks", "T22": "tau2-bench, reasoning judge", "T23": "Reasoning judge, frontier agents",
    "T24": "Monitor-aware agent", "T25": "Adaptive attacker", "T26": "Canonical-action monitor",
    "T27": "Training against the attack", "T28": "Cut set again after a prompt change"}
GNAMES = {"G": "Observable geometry (G1--G3)", "G5": "Answers vs replies (G5)", "G6": "Shared geometry (G6)",
          "G7": "What each API channel reveals (G7)",
          "G8": "Reading the reasoning (G8)",
          "G9": "Live self-policing (G9)",
          "G10": "A closed agent polices itself (G10)",
          "G11": "The prevention law across agents (G11, stopped)",
          "G11b": "The prevention law across agents (G11b)",
          "G12": "The composition-weighted law (G12)",
          "G13": "Self-policing against prompt injection (G13)",
          "G14": "Recreating the decision state, trained comparison (G14)",
          "G15": "The agent's own answers, training-free (G15)",
          "G16": "Own answers or another model's (G16)",
          "G17": "Training-free read-out, prompt injection (G17)",
          "G18": "Training-free read-out, a second benchmark for Claude Haiku 5.5 (G18)",
          "G19": "Own conversation or outside reviewer, same closed model (G19)",
          "G20": "Questionnaire or reasoning judge (G20)",
          "G21": "An open model answers about a closed agent (G21)",
          "G22": "One universal questionnaire on two benchmarks (G22)",
          "G23": "The theory predicts which questions monitor (G23)",
          "G24": "The closed agent's own answers with reasoning (G24)",
          "G25": "The theory predicts which of the agent's own questions monitor (G25)",
          "G26": "The closed model reviews its own calls, with reasoning (G26)",
          "G27": "State tomography and forecasting (G27)",
          "G28": "The estimated state against the true hidden state (G28)",
          "G27b": "State tomography replicated on three agents (G27b)",
          "G29": "Live prevention by the estimated state (G29)",
          "G29b": "Live prevention by the state before the call (G29b)",
          "G30": "The state before a call, beyond position, on three agents (G30)",
          "G31": "Repeats against dither: is the within-step effect measurement-limited? (G31)",
          "G31b": "Early warning and the horizon of the state (G31b)",
          "G31c": "The memory of the state and the filter's noise (G31c)",
          "G32": "A label-free signal says where the read-out works (G32)",
          "G33": "The closed agent's answers, dithered (G33)",
          "G34": "Triangulation: the closed model's own hidden geometry (G34)",
          "G35": "Does the monitor come from the reconstructed state? (G35)",
          "G35b": "With dither, does the read-out follow the hidden state? (G35b)",
          "G36": "Triangulation at the agent's decision (G36)",
          "G37": "The closed agent's state before its call (G37)",
          "G38": "A second closed agent, at its call and before it (G38)",
          "G39": "Live prevention on a closed agent before the call (G39)",
          "G40": "A closed agent's pre-call state beyond position, confirmed (G40)",
          "G39b": "The pre-call gate against a matched position gate (G39b)",
          "G41": "The relevant directions of the call (G41)",
          "G42": "The prevention law for pre-call gates (G42)",
          "G43": "A broader battery at the decision (G43)",
          "G44": "Self against other forecast (G44)"}
TID = re.compile(r"\b(T\d+[a-z]?|L\d[a-z]?)\b('s)?")


def names(s: str, lead: bool = False) -> str:
    """Test ids replaced by descriptive names (quoted inside running text)."""
    if lead:
        m = re.match(r"(T\d+[a-z]?|L\d[a-z]?)\b\s*", s)
        if m and m.group(1) in NAMES:
            return "\x00" + NAMES[m.group(1)] + "\x01" + names(s[m.end():])
    return TID.sub(lambda m: ("\x02" + NAMES[m.group(1)] + "\x03" + (m.group(2) or "")) if m.group(1) in NAMES else m.group(0), s)


ROMAN = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v", "6": "vi"}
EP = re.compile(r"\b([A-HJKMN-SU-Z])([1-6])(b?)\b")          # not L (test ids L1-L3e, named below) nor T or I          # endpoint ids (H1, C2, K1b, ...); T and I are test/feature ids
ARMS = {"R16_ts": "resampling at the start of the turn", "R16_dec": "resampling at the decision", "WB_ts": "white-box probe at the start of the turn",
        "R16": "resampling (16 rollouts)", "WB": "white-box probe", "I_dec": "internals at the decision", "READER_act": "reader",
        "J_act": "judge", "TXT_cv": "TF-IDF (cross-validated)", "READER_cv": "reader (cross-validated)", "TXT": "TF-IDF",
        "TOOL_cv": "tool identity (cross-validated)", "TOOL": "tool identity", "READER": "reader", "JUDGE": "judge"}
ARM = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ARMS), key=len, reverse=True)) + r")\b")


def endpoints(s: str, lead_id: str) -> str:
    """Endpoint and arm ids replaced by roman numerals and plain names (the released record keeps the ids)."""
    if lead_id == "T14":                                     # T14's endpoints are named L1/L2, like two test ids
        s = re.sub(r"\bL([12])\b", lambda m: "(" + ROMAN[m.group(1)] + ")", s)
    s = ARM.sub(lambda m: ARMS[m.group(1)], s)
    return EP.sub(lambda m: "(" + ROMAN[m.group(2)] + ("'" if m.group(3) else "") + ")", s)


PATH = re.compile(r"\b(?:src|scripts|tests|results|docs|paper|site|deploy)/[\w./*\-]+|\b[\w\-]+\.(?:json|jsonl|parquet|npz|csv\.gz|csv|py|md)\b")
JOB = re.compile(r"\bjob \d+\b")
COMMIT = re.compile(r"\b(?:in |commit )?[0-9a-f]{7}\b")


def anon(s: str) -> str:
    """Double-blind: no repository paths, job ids or commit hashes in the paper (the released record keeps them)."""
    s = PATH.sub("[released file]", s)
    s = JOB.sub("a GPU job", s)
    for x, y in (("Claude Haiku 5.5's", "Claude Haiku 5.5's"), ("Claude Haiku 5.5", "Claude Haiku 5.5")):   # the closed model is named (user, 2026-10-22)
        s = s.replace(x, y)
    return COMMIT.sub(lambda m: "[a public commit]" if m.group(0).strip()[:6] != "commit" else "[a public commit]", s)


def main():
    rows = []
    for line in (ROOT / "docs/PREREGISTRATION_RECORD.md").read_text().split("\n"):
        if not line.startswith("| ") or line.startswith("| Test") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 4:
            continue
        gm = re.match(r"(G\d*b?)\s+", cells[0])                  # the geometry tests: named before endpoint ids are rewritten
        if gm and gm.group(1) in GNAMES:
            if gm.group(1) == "G15":                              # G15's endpoints are named T1-T3, like test ids
                cells = [re.sub(r"\bT([1-3])\b", lambda m: "(" + ROMAN[m.group(1)] + ")", c) for c in cells]
            ep = {"G16": "O", "G17": "V", "G18": "R", "G19": "O", "G20": "J", "G21": "A", "G22": "U", "G23": "F", "G24": "E", "G25": "F", "G26": "R", "G27": "K", "G28": "T", "G27b": "K", "G29": "P", "G29b": "P", "G30": "W", "G31": "M", "G31b": "E", "G31c": "A", "G32": "C", "G33": "Q", "G34": "V", "G35": "L", "G35b": "K", "G36": "V", "G37": "P", "G38": "A", "G39": "P", "G40": "C", "G39b": "P", "G41": "R", "G42": "L", "G43": "S", "G44": "F"}.get(gm.group(1))   # their endpoints are named O1/V1/R1 ...
            if ep:
                cells = [re.sub(rf"\b{ep}([1-3])\b", lambda m: "(" + ROMAN[m.group(1)] + ")", c) for c in cells]
            if gm.group(1) == "G9":                               # G9's endpoints are named L1-L3, like test ids
                cells = [re.sub(r"\bL([1-3])\b", lambda m: "(" + ROMAN[m.group(1)] + ")", c) for c in cells]
                cells = [c.replace("S4@", "resampling@").replace("S4 ", "resampling ") for c in cells]
            rows.append(("\x00" + GNAMES[gm.group(1)] + "\x01" + names(endpoints(anon(cells[0][gm.end():]), None)),
                         names(endpoints(anon(cells[-1]), None))))
            continue
        lid = (re.match(r"(T\d+[a-z]?|L\d[a-z]?)\b", cells[0]) or [None])[0]
        rows.append((names(endpoints(anon(cells[0]), lid), lead=True), names(endpoints(anon(cells[-1]), lid))))
    body = "\n".join(f"{tex(t)} & {tex(o)}\\\\[2pt]".replace("\x00", "\\textbf{").replace("\x01", "}\\newline ")
                      .replace("\x02", "``").replace("\x03", "''") for t, o in rows)
    def _cls(line):
        u = line.rstrip().rstrip("|").split(" | ")[-1].upper()
        if "PENDING" in u:
            return "pending"
        if "STOPPED" in u:
            return "stopped"
        hp, hf = "PASS" in u, ("FAIL" in u or "NO DETECTABLE" in u)
        return "mixed" if hp and hf else "pass" if hp else "fail" if hf else "reported"
    rec = [l for l in (ROOT / "docs/PREREGISTRATION_RECORD.md").read_text().split("\n") if re.match(r"\| [GT]\d", l)]
    tally = []
    for pre, name in (("G", "method (G)"), ("T", "trained comparisons (T)")):
        c = {}
        for l in rec:
            if l.startswith("| " + pre):
                k = _cls(l)
                c[k] = c.get(k, 0) + 1
        tally.append(f"{name}: {sum(c.values())} tests; {c.get('pass', 0)} passed every primary endpoint, {c.get('mixed', 0)} passed "
                     f"some endpoints and failed others, {c.get('fail', 0)} failed, {c.get('stopped', 0)} stopped before any "
                     f"outcome, {c.get('reported', 0)} reported without a pass/fail bar, {c.get('pending', 0)} pending")
    tally_txt = ("Tally, as recorded: " + "; ".join(tally) + ". No correction for multiple tests is applied; each test answers its own "
                 "registered question, and every outcome, failed or passed, is listed below.\n\n")
    (ROOT / "paper/appendix_record.tex").write_text(
        tally_txt +
        "Every confirmatory test, as in the released record (generated from it by a script; file paths and commit ids are "
        "replaced for review). Locks (SHA-256 of the preregistration and its code) and outcomes are in the released record; lock "
        "times are self-reported. Endpoints are numbered (i), (ii), ... in the order of each preregistration. Later tests changed two "
        "readings fixed in advance. First, the unseen-task test showed that labels do not transfer on tau2-bench through word counts: the "
        "test tasks had been in training. Second, ``checks rules'' in the frontier-agent judge test means only that the judge separates "
        "violating from allowed writes of the same tool.\n\n"
        "{\\small\\begin{longtable}{p{0.24\\textwidth}p{0.71\\textwidth}}\n\\toprule Test & Outcome (verbatim from the record)\\\\\\midrule\\endhead\n"
        + body + "\n\\bottomrule\n\\end{longtable}}\n")
    md = (ROOT / "docs/PREREGISTRATION_RECORD.md").read_text()
    if "## Correction to the T26 cause" in md:                 # a dated correction of one recorded cause, as in the record
        sec = md[md.index("## Correction to the T26 cause"):].split("\n## ")[0].split("\n", 1)[1]
        items, cur = [], None
        for line in sec.splitlines():
            if line.startswith("- "):
                cur = [line[2:].strip()]; items.append(cur)
            elif line.startswith("  ") and cur is not None:
                cur.append(line.strip())
            elif line.strip() and (not items or (len(items) == 1 and not sec.lstrip().startswith("- ") and cur is None)):
                (items[0].append(line.strip()) if items else items.append([line.strip()]))
        body_c = "\n".join("\\item " + tex(names(endpoints(anon(" ".join(i)), "T26"))).replace("\x02", "``").replace("\x03", "''") for i in items)
        with open(ROOT / "paper/appendix_record.tex", "a") as f:
            f.write("\n\\paragraph{Correction to one recorded cause.} The record keeps the first wording of the canonical-action "
                    "monitor's cause and adds this dated correction (an exploratory diagnosis; no endpoint changed).\n"
                    "\\begin{itemize}\n" + body_c + "\n\\end{itemize}\n")
    print("appendix_record.tex:", len(rows), "tests")


if __name__ == "__main__":
    main()
