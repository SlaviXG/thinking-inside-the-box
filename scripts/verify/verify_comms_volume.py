"""Verify the communication-volume ratio claimed in the dissertation.

Inputs:
- Backbone parameter count for DeepSeek-R1-Distill-Llama-8B (constant from _style.py).
- Measured FLoRA per-client per-round encrypted adapter payload (from the
  archived 2026-04-02 benchmark, comm_bytes_flora field).

Outputs:
- FedAvg fp16 reference payload per client per round (bytes, MB, GB).
- FLoRA measured payload per client per round (bytes, MB).
- Ratio FedAvg_fp16 / FLoRA (the "approximately 880x" claim in main.tex).

Reproduce: python scripts/verify/verify_comms_volume.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "2026-04-02"

# DeepSeek-R1-Distill-Llama-8B has the Llama-3.1-8B architecture (untied
# embeddings), 8,030,261,248 parameters total. Constant referenced in
# scripts/figures/_style.py (BACKBONE_PARAMS).
BACKBONE_PARAMS = 8_030_261_248

# Expected ratio reported in main.tex: "approximately three orders of
# magnitude (approximately 880x)". This script asserts the computed value
# is within +/- 10x of that headline.
EXPECTED_RATIO = 880
RATIO_TOLERANCE = 10


def load_flora_payload_bytes() -> int:
    with (DATA_DIR / "history_flora_graph.json").open("r", encoding="utf-8") as fh:
        history = json.load(fh)
    rows = history["comm_bytes_flora"]
    per_round_per_client = []
    for row in rows:
        if isinstance(row, list):
            per_round_per_client.extend(row)
        else:
            per_round_per_client.append(row)
    seen = set(per_round_per_client)
    if len(seen) != 1:
        raise RuntimeError(
            f"Expected a single FLoRA payload size across rounds and clients; "
            f"saw {len(seen)} distinct values: {sorted(seen)}"
        )
    return per_round_per_client[0]


def main() -> int:
    fedavg_fp16_bytes = BACKBONE_PARAMS * 2
    flora_bytes = load_flora_payload_bytes()
    ratio = fedavg_fp16_bytes / flora_bytes

    print("Inputs:")
    print(f"  BACKBONE_PARAMS                = {BACKBONE_PARAMS:,} (constant, _style.py)")
    print(f"  fp16 bytes per parameter       = 2 (constant)")
    print(f"  FLoRA payload (bytes)          = {flora_bytes:,} (measured, history_flora_graph.json)")
    print()
    print("Computed:")
    print(f"  FedAvg fp16 per client/round   = {fedavg_fp16_bytes:,} bytes")
    print(f"                                 = {fedavg_fp16_bytes / 2**20:,.1f} MB")
    print(f"                                 = {fedavg_fp16_bytes / 2**30:,.2f} GB")
    print(f"  FLoRA per client/round         = {flora_bytes / 2**20:,.2f} MB")
    print(f"  ratio FedAvg_fp16 / FLoRA      = {ratio:,.1f}x")
    print()
    print(f"Expected ratio in main.tex: ~{EXPECTED_RATIO}x (+/- {RATIO_TOLERANCE}x)")

    if abs(ratio - EXPECTED_RATIO) > RATIO_TOLERANCE:
        print(f"FAIL: computed ratio {ratio:.1f} differs from expected {EXPECTED_RATIO} by more than {RATIO_TOLERANCE}.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
