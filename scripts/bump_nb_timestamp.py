"""Bump the first-cell timestamp on CSC3094_Thinking_Inside_The_Box.ipynb.

Intended to be invoked from a git pre-commit hook when the notebook is
part of the staged change set. Writes UTC time in the same format used
by prior manual bumps so the cell stays visually stable.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "CSC3094_Thinking_Inside_The_Box.ipynb"


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    line = f"# Last updated: {ts} UTC"
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = nb["cells"][0].get("source")
    if isinstance(src, list):
        nb["cells"][0]["source"] = [line]
    else:
        nb["cells"][0]["source"] = line
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[pre-commit] notebook timestamp -> {ts} UTC")


if __name__ == "__main__":
    main()
