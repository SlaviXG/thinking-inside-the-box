# Numerical claim verification

Every numerical claim that enters the dissertation `assets/CSC3094_Dissertation/main.tex` must be reproducible from a script under this directory.

Each script:
- Prints its **inputs** with provenance (constant, measured value, citation)
- Prints its **outputs** (computed numbers)
- Has a `main()` that exits 0 when computed values match expected, non-zero otherwise

Convention: name the script `verify_<topic>.py`. Cite the script (or its output value) in `main.tex`, never the reverse.

Run the full set: `python -m scripts.verify.verify_all` (when added).
