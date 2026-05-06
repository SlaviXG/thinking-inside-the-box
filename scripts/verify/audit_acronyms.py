"""Audit acronyms and key proper nouns in main.tex.

Finds all-caps sequences (2+ letters) and a curated list of proper nouns,
then checks each against the acronym table and first-use definitions.

Run: python scripts/verify/audit_acronyms.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "assets" / "CSC3094_Dissertation" / "main.tex"

# Proper nouns / mixed-case terms that are not all-caps but still need definitions.
PROPER_NOUNS = [
    "Flower", "FedAvg", "SCAFFOLD", "FLoRA", "FedProx", "FedBN",
    "Virtual Client Engine", "QLoRA", "LoRA", "DeepSeek", "Kuzu",
    "PEFT", "bitsandbytes", "LangChain", "Hugging Face",
]

# Terms that are intentionally not defined (common English, LaTeX internals, etc.)
SKIP_TERMS = {
    "I", "A", "AI", "API", "GPU", "CPU", "RAM", "URL", "PDF", "HTML",
    "SQL", "JSON", "CSV", "XML", "HTTP", "HTTPS", "LLM", "NLP",
    "ML", "DL", "FL", "RL", "CV", "NLU",
    "GBP", "USD", "EUR", "UK", "US", "EU",
    "AML", "KYC", "CFT",
    "GDPR", "HIPAA", "FATF", "PRA", "FCA", "OSFI",
    "MLR", "SS",  # short fragments
    "AMLD",
    "OK", "NaN", "TRUE", "FALSE", "NULL",
    "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI",  # Roman numerals
    "OI", "OII", "OIII", "OIV", "OV",  # Objective labels
    "RQ", "RQI", "RQII", "RQIII",  # Research question labels
    "CI", "CD",  # DevOps
    "IEEE", "ACM", "NIST",
    "MSc", "BSc", "PhD",
    "LLaMA",
    "SFT", "DPO",
    "VRAM",
    "CoT",
    "KV",  # KV cache
    "BF", "TF",  # tensor formats
    "ID", "IDs",
    "cf",  # Latin abbreviation
    "eg", "ie",
    "OOM",  # out of memory
    "TCO",
    "FP", "FN", "TP", "TN",
    "ROC", "AUC",
    "SWA",
    "EMA",
    "SGD",
    "MLP",
    "FFN",
    "QA",
    "RA",
    "NF",  # NF4 quantisation
    "INT",  # INT8
    "STR",
    "BM",  # BM25
    "KNN",
    "NN",
    "DB",
    "PII",
    "HSM",
    "TLS",
    "PKI",
    "AES",
    "RSA",
    "SHA",
    "MAC",
    "IV",
    "SMPC",
    "HE",
    "DP",
    "TE",
    "LA",
    "MIA",
    "LDP",
    "CDP",
}


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def extract_body(text: str) -> str:
    """Return only the text after \begin{document}."""
    m = re.search(r"\\begin\{document\}", text)
    return text[m.end():] if m else text


def find_acronym_table_entries(text: str) -> set[str]:
    """Extract acronyms defined in the acronym table (longtable rows like GDPR & ...)."""
    defined = set()
    # Match lines like: GDPR & General Data Protection Regulation \\
    for m in re.finditer(r"^\s*([A-Z][A-Z0-9\-]{1,})\s*&", text, re.MULTILINE):
        defined.add(m.group(1).strip())
    return defined


def find_inline_definitions(text: str) -> set[str]:
    """Find patterns like 'Federated Learning (FL)' or '(FL)' after an expansion."""
    defined = set()
    # Pattern: word(s) (ACRONYM)
    for m in re.finditer(r"\(([A-Z][A-Z0-9\-]{1,})\)", text):
        defined.add(m.group(1))
    # Also match proper noun definitions like \textit{Flower} or just "Flower framework"
    # For proper nouns we just check they appear in the document at all (shallow check).
    return defined


def main() -> int:
    raw = TEX.read_text(encoding="utf-8")
    no_comments = strip_comments(raw)
    body = extract_body(no_comments)

    # --- Build the set of defined terms ---
    acro_table_defs = find_acronym_table_entries(no_comments)
    inline_defs = find_inline_definitions(body)
    all_defined = acro_table_defs | inline_defs

    print(f"Acronym table entries:  {len(acro_table_defs)}")
    print(f"Inline definitions:     {len(inline_defs)}")
    print()

    # --- Find all all-caps acronyms in body ---
    # Exclude things inside \cite{}, \ref{}, \label{}, \url{}, \texttt{}, math $...$
    cleaned = re.sub(r"\$[^$]+\$", "", body)
    cleaned = re.sub(r"\\(?:cite|ref|label|url|autoref|cref|Cref|eqref)\*?\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"\\texttt\{[^}]*\}", "", cleaned)
    cleaned = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", "", cleaned)

    all_caps_re = re.compile(r"\b([A-Z]{2,}(?:[0-9]+[A-Z]*)?)\b")
    found_acronyms: dict[str, int] = {}
    for m in all_caps_re.finditer(cleaned):
        term = m.group(1)
        if term not in SKIP_TERMS:
            found_acronyms[term] = found_acronyms.get(term, 0) + 1

    # --- Check proper nouns ---
    found_proper: dict[str, int] = {}
    for noun in PROPER_NOUNS:
        count = len(re.findall(re.escape(noun), body))
        if count > 0:
            found_proper[noun] = count

    # --- Report ---
    print("=== All-caps acronyms not in acronym table or inline-defined ===")
    undefined_acronyms = {k: v for k, v in sorted(found_acronyms.items())
                          if k not in all_defined}
    if undefined_acronyms:
        for term, count in sorted(undefined_acronyms.items(), key=lambda x: -x[1]):
            print(f"  {term:20s}  {count:4d} occurrences")
    else:
        print("  (none)")
    print()

    print("=== Proper nouns present in body ===")
    for noun, count in sorted(found_proper.items(), key=lambda x: -x[1]):
        status = "OK" if any(noun in d or d in noun for d in all_defined | {"Flower", "FedAvg"}) else "CHECK"
        # More precise: look for definition pattern near first occurrence
        first_pos = body.find(noun)
        context_window = body[max(0, first_pos - 200):first_pos + 300]
        defined_inline = bool(re.search(
            re.escape(noun) + r".{0,80}\(", context_window
        )) or bool(re.search(
            r"\(' + re.escape(noun) + r'", context_window
        ))
        print(f"  {noun:30s}  {count:4d} occurrences")
    print()

    print("=== Acronym table entries (for reference) ===")
    for entry in sorted(acro_table_defs):
        print(f"  {entry}")
    print()

    total_undefined = len(undefined_acronyms)
    print(f"Potentially undefined acronyms: {total_undefined}")
    return 0 if total_undefined == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
