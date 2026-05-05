"""Find paragraphs in main.tex that are a single sentence.

A "paragraph" here is a non-empty block of text between blank lines that
isn't a LaTeX environment header, comment, or list item. A "single
sentence" is a paragraph whose stripped content contains exactly one
sentence-terminating period (excluding abbreviations and citation periods).

This is a heuristic and is meant to flag suspect paragraphs for manual
review, not to provide a strict count.

Run: python scripts/verify/audit_one_sentence_paragraphs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "assets" / "CSC3094_Dissertation" / "main.tex"

SKIP_PREFIXES = (
    "%", "\\", "}", "\\end", "\\item", "\\begin", "\\caption",
)


def is_skip(line: str) -> bool:
    s = line.lstrip()
    if not s:
        return True
    if s.startswith("%"):
        return True
    if s.startswith("\\"):
        return True
    return False


def main() -> int:
    text = TEX.read_text(encoding="utf-8")
    lines = text.splitlines()

    paragraphs: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = None
    in_table = False
    in_itemize = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if "\\begin{tabular" in line or "\\begin{table" in line:
            in_table = True
        if "\\end{tabular" in line or "\\end{table" in line:
            in_table = False
            continue
        if "\\begin{itemize" in line or "\\begin{enumerate" in line:
            in_itemize = True
        if "\\end{itemize" in line or "\\end{enumerate" in line:
            in_itemize = False
            continue
        if in_table or in_itemize:
            continue

        if not stripped:
            if buf:
                paragraphs.append((buf_start, " ".join(buf)))
                buf = []
                buf_start = None
            continue

        if is_skip(line):
            if buf:
                paragraphs.append((buf_start, " ".join(buf)))
                buf = []
                buf_start = None
            continue

        if not buf:
            buf_start = i
        buf.append(stripped)

    if buf:
        paragraphs.append((buf_start, " ".join(buf)))

    sentence_terminators = re.compile(r"(?<![A-Z])\.(\s|$)")

    flagged = 0
    for line_no, content in paragraphs:
        # Strip simple math and citations to reduce false positives.
        cleaned = re.sub(r"\\cite\{[^}]+\}", "", content)
        cleaned = re.sub(r"\\ref\{[^}]+\}", "", cleaned)
        cleaned = re.sub(r"\$[^$]+\$", "", cleaned)
        sentences = sentence_terminators.findall(cleaned)
        if len(sentences) == 1 and len(content.split()) > 5:
            flagged += 1
            preview = content[:140] + ("..." if len(content) > 140 else "")
            print(f"L{line_no:5d}  {preview}")

    print()
    print(f"Flagged {flagged} candidate one-sentence paragraphs.")
    return 0 if flagged == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
