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


def main():
    rows = []
    for line in (ROOT / "docs/PREREGISTRATION_RECORD.md").read_text().split("\n"):
        if not line.startswith("| ") or line.startswith("| Test") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 4:
            continue
        rows.append((cells[0], cells[-1]))
    body = "\n".join(f"{tex(t)} & {tex(o)}\\\\[2pt]" for t, o in rows)
    (ROOT / "paper/appendix_record.tex").write_text(
        "Every confirmatory test, as in the record (generated from docs/PREREGISTRATION\\_RECORD.md by paper/make\\_record.py). "
        "Locks (SHA-256 of the preregistration and its code) and outcomes are in the released record; lock times are self-reported.\n\n"
        "{\\small\\begin{longtable}{p{0.24\\textwidth}p{0.71\\textwidth}}\n\\toprule Test & Outcome (verbatim from the record)\\\\\\midrule\\endhead\n"
        + body + "\n\\bottomrule\n\\end{longtable}}\n")
    print("appendix_record.tex:", len(rows), "tests")


if __name__ == "__main__":
    main()
