"""Per-section word count for main.tex with explicit exclusions.

Reports words per section so the title page, references, and any appendix
can be excluded transparently.

Run: python scripts/verify/wordcount_breakdown.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "assets" / "CSC3094_Dissertation" / "main.tex"

LANDMARK_PATTERNS = {
    "maketitle":       r"\\maketitle",
    "exec_start":      r"\\section\*\{Executive Summary\}",
    "abstract_start":  r"\\begin\{abstract\}",
    "abstract_end":    r"\\end\{abstract\}",
    "toc":             r"\\tableofcontents",
    "acronyms_start":  r"\\section\*\{List of Acronyms\}",
    "intro":           r"\\chapter\{Introduction\}",
    "bg":              r"\\chapter\{Background Review\}",
    "method":          r"\\chapter\{Methodology\}",
    "results":         r"\\chapter\{Results and Evaluation\}",
    "concl":           r"\\chapter\{Conclusions\}",
    "appendix":        r"\\appendix\b",
    "biblio":          r"\\bibliography\{",
}


def count_words(text: str) -> int:
    text = re.sub(r"(?<!\\)%.*", "", text)
    for cmd in ("textbf", "textit", "emph", "texttt", "textsc"):
        text = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(cite|ref|autoref|eqref|label|cref|Cref|url)\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", "", text)
    text = re.sub(r"\$[^$]*\$", "X", text)
    text = re.sub(r"[{}]", " ", text)
    return len(re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", text))


def main() -> int:
    lines = TEX.read_text(encoding="utf-8").splitlines()

    landmarks: dict[str, int] = {}
    for key, pat in LANDMARK_PATTERNS.items():
        rx = re.compile(pat)
        for i, line in enumerate(lines):
            if rx.search(line):
                landmarks.setdefault(key, i)
                break

    print("Landmarks (1-indexed line numbers):")
    for key in sorted(landmarks, key=landmarks.get):
        print(f"  {key:18s} L{landmarks[key] + 1}")
    print()

    def slice_count(start: int, end: int, label: str) -> int:
        block = "\n".join(lines[start:end])
        n = count_words(block)
        print(f"  {label:55s}  L{start + 1:5d}..{end:5d}  {n:6d} words")
        return n

    intro = landmarks["intro"]
    bibl = landmarks["biblio"]
    appendix = landmarks.get("appendix")
    body_end = appendix if appendix is not None else bibl
    acronyms_start = landmarks.get("acronyms_start", intro)
    abs_start = landmarks["abstract_start"]
    abs_end_inclusive = landmarks["abstract_end"] + 1
    exec_start = landmarks.get("exec_start")

    # The executive summary lives either in the front matter (before the
    # abstract) or at the end of Chapter 5 (after the closing reflection).
    # If it sits within Chapter 5, the chapter's word count already covers
    # it; we still report its size for visibility.
    if exec_start is not None and exec_start < abs_start:
        exec_in_front = True
    else:
        exec_in_front = False

    print("Per-section word counts:")
    if exec_in_front:
        exec_n = slice_count(exec_start, abs_start, "Executive Summary (front matter)")
    else:
        exec_n = 0
    abs_n   = slice_count(abs_start, abs_end_inclusive, "Abstract")
    acro_n  = slice_count(acronyms_start, intro, "List of Acronyms (table)")
    intro_n = slice_count(intro, landmarks["bg"], "Chapter 1 Introduction")
    bg_n    = slice_count(landmarks["bg"], landmarks["method"], "Chapter 2 Background Review")
    meth_n  = slice_count(landmarks["method"], landmarks["results"], "Chapter 3 Methodology")
    res_n   = slice_count(landmarks["results"], landmarks["concl"], "Chapter 4 Results and Evaluation")
    con_n   = slice_count(landmarks["concl"], body_end, "Chapter 5 Conclusions")
    if not exec_in_front and exec_start is not None and exec_start < body_end:
        # Sub-count: portion of Chapter 5 that is the executive summary.
        exec_n_subcount = count_words(
            "\n".join(lines[exec_start:body_end])
        )
        print(f"    of which Executive Summary (in Ch.~5): "
              f"{'':10s} L{exec_start + 1:5d}..{body_end:5d}  {exec_n_subcount:6d} words")
    if appendix is not None:
        slice_count(appendix, bibl, "Appendix (excluded from totals)")
    else:
        print("  (no \\appendix directive found)")
    print()

    chapters = intro_n + bg_n + meth_n + res_n + con_n
    body_with_acronyms = exec_n + abs_n + acro_n + chapters
    body_without_acronyms = exec_n + abs_n + chapters
    print("Totals (excluding title page, bibliography, and appendix):")
    print(f"  Abstract + Acronyms + 5 Chapters: {body_with_acronyms:6d} words")
    print(f"  Abstract + 5 Chapters (no acronyms): {body_without_acronyms:6d} words")
    print(f"  Five chapters only (no abstract, no acronyms): {chapters:6d} words")
    return 0


if __name__ == "__main__":
    main()
