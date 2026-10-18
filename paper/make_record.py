#!/usr/bin/env python
"""Write paper/appendix_record.tex from the table in docs/PREREGISTRATION_RECORD.md, so the paper's appendix can never
go stale (reviewer finding M5). Run before building the paper: python paper/make_record.py"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNI = {"−": "$-$", "→": "$\\rightarrow$", "≥": "$\\geq$", "≤": "$\\leq$", "×": "$\\times$", "≈": "$\\approx$", "∪": "$\\cup$",
       "²": "$^2$", "τ": "$\\tau$", "α": "$\\alpha$", "σ": "$\\sigma$", "Ψ": "$\\Psi$", "ψ": "$\\psi$", "θ": "$\\theta$", "Δ": "$\\Delta$",
       "·": "$\\cdot$", "…": "\\ldots{}", "–": "--", "—": "---", "‑": "-", "≠": "$\\neq$", "±": "$\\pm$", "μ": "$\\mu$", "ρ": "$\\rho$",
       "✓": "yes", "’": "'", "“": "``", "”": "''", "é": "\\'e", "≪": "$\\ll$", "⊥": "$\\perp$", "∈": "$\\in$", "β": "$\\beta$", "λ": "$\\lambda$"}


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


PATH = re.compile(r"\b(?:src|scripts|tests|results|docs|paper|site|deploy)/[\w./*\-]+")
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
        rows.append((anon(cells[0]), anon(cells[-1])))
    body = "\n".join(f"{tex(t)} & {tex(o)}\\\\[2pt]" for t, o in rows)
    (ROOT / "paper/appendix_record.tex").write_text(
        "Every confirmatory test, as in the released record (generated from it by a script; file paths and commit ids are "
        "replaced for review). Locks (SHA-256 of the preregistration and its code) and outcomes are in the released record; lock "
        "times are self-reported. Later tests changed two readings fixed in advance: T21 refuted T20's reading that the labels "
        "transfer through word counts (the test tasks were in training); T23's ``checks rules'' means that the judge separates "
        "violating from allowed writes of the same tool.\n\n"
        "{\\small\\begin{longtable}{p{0.24\\textwidth}p{0.71\\textwidth}}\n\\toprule Test & Outcome (verbatim from the record)\\\\\\midrule\\endhead\n"
        + body + "\n\\bottomrule\n\\end{longtable}}\n")
    print("appendix_record.tex:", len(rows), "tests")


if __name__ == "__main__":
    main()
