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
    "L1": "Interrogation state, open chat models", "L2": "Interrogation state, confirmation", "L3": "Claude chat readouts",
    "L3b": "Claude refusals vs moderation", "L3c": "Claude, one-call grey box", "L3d": "Claude, transcript readout",
    "L3e": "Claude vs graded moderators",
    "T1": "Resampling vs a probe", "T2": "When the internals are read", "T3": "Resampling with a deployable judge",
    "T4": "Reader vs the agent's internals", "T5": "Unseen task families", "T6": "Another open agent",
    "T7": "AgentDojo, no labels", "T8": "Claude, after a reveal", "T8b": "Closed agent: Claude",
    "T9": "Reader vs TF-IDF", "T10": "A policy removed from training", "T11": "Closed agent: GPT-6-luna",
    "T12": "Labels for a new policy", "T13": "AgentDojo with a defense", "T13b": "AgentDojo, closed agent",
    "T14": "Open-agent labels for a closed agent", "T15": "Replication (Gemma-4)", "T16": "Live prevention (GPT)",
    "T16b": "Live prevention (Claude)", "T16c": "Hold for review", "T17": "Replication, three architectures",
    "T18": "tau2-bench, Claude", "T19": "Browser-size readers", "T20": "tau2-bench, two closed agents",
    "T21": "tau2-bench, unseen tasks", "T22": "tau2-bench, reasoning judge", "T23": "Reasoning judge, frontier agents",
    "T24": "Monitor-aware agent", "T25": "Adaptive attacker", "T26": "Canonical-action monitor"}
TID = re.compile(r"\b(T\d+[a-z]?|L\d[a-z]?)\b('s)?")


def names(s: str, lead: bool = False) -> str:
    """Test ids replaced by descriptive names (quoted inside running text)."""
    if lead:
        m = re.match(r"(T\d+[a-z]?|L\d[a-z]?)\b\s*", s)
        if m and m.group(1) in NAMES:
            return "\x00" + NAMES[m.group(1)] + "\x01" + names(s[m.end():])
    return TID.sub(lambda m: ("\x02" + NAMES[m.group(1)] + "\x03" + (m.group(2) or "")) if m.group(1) in NAMES else m.group(0), s)


ROMAN = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v", "6": "vi"}
EP = re.compile(r"\b([A-HJ-SU-Z])([1-6])(b?)\b")          # endpoint ids (H1, C2, K1b, ...); T and I are test/feature ids
ARMS = {"R16": "resampling (16 rollouts)", "WB": "white-box probe", "I_dec": "internals at the decision", "READER_act": "reader",
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
    return COMMIT.sub(lambda m: "[a public commit]" if m.group(0).strip()[:6] != "commit" else "[a public commit]", s)


def main():
    rows = []
    for line in (ROOT / "docs/PREREGISTRATION_RECORD.md").read_text().split("\n"):
        if not line.startswith("| ") or line.startswith("| Test") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 4:
            continue
        lid = (re.match(r"(T\d+[a-z]?|L\d[a-z]?)\b", cells[0]) or [None])[0]
        rows.append((names(endpoints(anon(cells[0]), lid), lead=True), names(endpoints(anon(cells[-1]), lid))))
    body = "\n".join(f"{tex(t)} & {tex(o)}\\\\[2pt]".replace("\x00", "\\textbf{").replace("\x01", "}\\newline ")
                      .replace("\x02", "``").replace("\x03", "''") for t, o in rows)
    (ROOT / "paper/appendix_record.tex").write_text(
        "Every confirmatory test, as in the released record (generated from it by a script; file paths and commit ids are "
        "replaced for review). Locks (SHA-256 of the preregistration and its code) and outcomes are in the released record; lock "
        "times are self-reported. Endpoints are numbered (i), (ii), ... in the order of each preregistration. Later tests changed two "
        "readings fixed in advance. First, the unseen-task test showed that labels do not transfer on tau2-bench through word counts: the "
        "test tasks had been in training. Second, ``checks rules'' in the frontier-agent judge test means only that the judge separates "
        "violating from allowed writes of the same tool.\n\n"
        "{\\small\\begin{longtable}{p{0.24\\textwidth}p{0.71\\textwidth}}\n\\toprule Test & Outcome (verbatim from the record)\\\\\\midrule\\endhead\n"
        + body + "\n\\bottomrule\n\\end{longtable}}\n")
    print("appendix_record.tex:", len(rows), "tests")


if __name__ == "__main__":
    main()
