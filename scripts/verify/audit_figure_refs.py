"""Audit that every figure label is referenced in text BEFORE its figure environment.

Run: python scripts/verify/audit_figure_refs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "assets" / "CSC3094_Dissertation" / "main.tex"

LABEL_RE = re.compile(r"\\label\{(fig:[^}]+)\}")


def main() -> int:
    text = TEX.read_text(encoding="utf-8")
    lines = text.splitlines()

    fig_labels: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("%"):
            continue
        m = LABEL_RE.search(line)
        if m:
            fig_labels.append((i, m.group(1)))

    failures = 0
    for line_no, label in fig_labels:
        needle = "\\ref{" + label + "}"
        first_ref_line = None
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith("%"):
                continue
            if needle in line:
                first_ref_line = i
                break
        ok = first_ref_line is not None and first_ref_line < line_no
        status = "OK" if ok else "PROBLEM"
        first_ref_str = str(first_ref_line) if first_ref_line is not None else "NONE"
        print(f"{status:8s}  fig L{line_no:5d}  {label:38s}  first ref L{first_ref_str}")
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"{failures} figure(s) without a preceding text reference.")
        return 1
    print("All figures referenced in text before they appear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
