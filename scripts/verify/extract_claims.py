"""Extract all verifiable claims from main.tex.

Produces two lists:
  1. Regulatory citations with article/section numbers
  2. Numerical claims (integers, decimals, ratios, costs, sizes)

Run: python scripts/verify/extract_claims.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "assets" / "CSC3094_Dissertation" / "main.tex"


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def extract_body(text: str) -> str:
    m = re.search(r"\\begin\{document\}", text)
    return text[m.end():] if m else text


def main() -> None:
    raw = TEX.read_text(encoding="utf-8")
    lines = raw.splitlines()
    body = extract_body(strip_comments(raw))

    # --- 1. Regulatory citations ---
    # Look for: GDPR Art/Article, Directive, §, PRA, OSFI, FATF, 6AMLD, PATRIOT, HIPAA, etc.
    reg_patterns = [
        r"GDPR\s+Art(?:icle)?\.?\s*\d+(?:\(\d+\)(?:\([a-z]\))?)?",
        r"Article\s+\d+(?:\(\d+\)(?:\([a-z]\))?)?",
        r"Directive\s+\(EU\)\s+\d{4}/\d+",
        r"PRA\s+SS\d+/\d+",
        r"OSFI\s+(?:Guideline\s+)?E-\d+",
        r"FATF\s+Recommendation\s+\d+",
        r"Recommendation\s+\d+",
        r"Section?\s*~?\\?S?\s*314\(b\)",
        r"\\\$\s*314\(b\)",
        r"\§\s*314",
        r"SI\s+\d{4}/\d+",
        r"Regulations\s+\d{4}",
        r"in force\s+since\s+[\w~\s]+\d{4}",
        r"effective\s+\d+\s+\w+\s+\d{4}",
        r"in force\s+\d+",
    ]
    combined_reg = re.compile("|".join(f"(?:{p})" for p in reg_patterns), re.IGNORECASE)

    print("=" * 70)
    print("REGULATORY CLAIMS")
    print("=" * 70)
    for i, line in enumerate(lines, 1):
        clean = re.sub(r"(?<!\\)%.*", "", line)
        matches = combined_reg.findall(clean)
        if matches:
            for m in matches:
                print(f"  L{i:5d}  {m.strip()[:80]}")
                print(f"         context: {clean.strip()[:100]}")

    # --- 2. Key numerical claims ---
    print()
    print("=" * 70)
    print("NUMERICAL CLAIMS (key metrics, ratios, costs, sizes)")
    print("=" * 70)

    # Patterns for specific kinds of numbers
    num_patterns = {
        "F1/AUC scores":     r"[Ff]1\s*(?:score\s*)?[=~]\s*\$?[\d\.]+|AUC\s+(?:of\s+)?\$?[\d\.]+|0\.[78]\d{2}|0\.4[0-9]",
        "GB/MB sizes":       r"\$[\d\.]+\$~?(?:GB|MB|GiB|MiB|Mbps|Mb)\b|[\d\.]+~(?:GB|MB|GiB|MiB|Mbps)",
        "Times (x)":         r"\$[\d,]+\$~?\\times|\d+[\d,]*\s*\\times\s+smaller",
        "A100-hours/cost":   r"[\d\.]+~?A100-hours?|USD~?\\\$[\d,\.]+|\\\\$[\d,\.]+\s*(?:per|per\s+session)|USD~\$[\d,\.]+",
        "Per-round seconds": r"[\d\.]+~seconds?|[\d\.]+\s+seconds?\s+per",
        "Parameters":        r"[\d\.]+~?(?:million|billion)\s+parameters?|\$[\d\.]+\$~?(?:million|billion)",
        "Ratios/percent":    r"\$[\d\.]+\\%|\d+\.?\d*\s*\\%|\d+\.?\d*\s*per\s+cent",
        "Costs USD/GBP":     r"(?:USD|GBP)~?\\\$[\d,\.]+(?:~billion)?|GBP~\\\$[\d,\.]+",
        "FP/TP counts":      r"\$[\d,]+\$\s*(?:false|per\s+million)|[\d,]+\s+false\s+positives",
    }

    for label, pat in num_patterns.items():
        rx = re.compile(pat)
        hits = []
        for i, line in enumerate(lines, 1):
            clean = re.sub(r"(?<!\\)%.*", "", line)
            if rx.search(clean):
                hits.append((i, clean.strip()[:110]))
        if hits:
            print(f"\n--- {label} ---")
            for lineno, ctx in hits:
                print(f"  L{lineno:5d}  {ctx}")


if __name__ == "__main__":
    main()
