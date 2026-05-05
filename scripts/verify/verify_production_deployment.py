"""Verify production-deployment numbers used in the Methodology chapter.

Recasts the benchmark's measured per-round bandwidth and wall-clock time
into figures that a deployment engineer can reason about: upload time at
realistic WAN speeds, GPU-hour budget for one full federated training
session, and dollar-cost ranges at on-demand A100 cloud pricing.

Inputs:
- Adapter payload per client per round (measured from the archived JSON,
  Fernet-wrapped LoRA weights at r=8, q_proj+v_proj).
- FLoRA fit-stage wall-clock per client per round (measured, mean across
  the 10 archived rounds).
- WAN connection speeds (100 Mbps, 1 Gbps, 10 Gbps) representing typical
  inter-datacentre links available to a Tier-1 bank.
- On-demand A100 cloud pricing range (citation: 2025-2026 cloud pricing
  surveys; Spheron/SynpixCloud at the low end, AWS p4d at the high end).

Outputs:
- Upload time per round per client at each WAN speed.
- Compute-to-comms ratio (fit_time / upload_time): demonstrates that the
  bandwidth saving makes the upload time operationally negligible.
- 10-round 3-client session: GPU-hours and USD cost range.
- Annual cost projection at weekly retraining cadence.

Run: python scripts/verify/verify_production_deployment.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "2026-04-02"

ADAPTER_BYTES_EXPECTED = 18_197_432  # cross-checked by verify_comms_volume.py

# WAN connection speeds in megabits per second. Representative of typical
# inter-datacentre links used by Tier-1 banks for cross-site replication.
WAN_SPEEDS_MBPS = [100, 1_000, 10_000]

# On-demand A100 hourly USD pricing, sourced from publicly listed rates
# in 2026:
#   - Specialised providers (Spheron, SynpixCloud A100 40GB SXM4): ~USD 1.07/hr
#   - AWS p4d.24xlarge per-A100 effective rate: ~USD 4.09/hr
# The script reports cost ranges at these low/high bounds.
A100_USD_PER_HOUR_LOW = 1.07
A100_USD_PER_HOUR_HIGH = 4.09

# Benchmark dimensions (matches main.tex Section 4.1 setup).
ROUNDS_PER_SESSION = 10
CLIENTS_PER_SESSION = 3
SESSIONS_PER_YEAR_WEEKLY = 52


def _mean_of_rows(rows):
    flat = []
    for row in rows:
        if isinstance(row, list):
            flat.extend(row)
        else:
            flat.append(float(row))
    return statistics.mean(flat)


def load_flora_inputs() -> tuple[int, float]:
    with (DATA_DIR / "history_flora_graph.json").open("r", encoding="utf-8") as fh:
        history = json.load(fh)
    payload = set()
    for row in history["comm_bytes_flora"]:
        if isinstance(row, list):
            payload.update(row)
        else:
            payload.add(row)
    if len(payload) != 1:
        raise RuntimeError(f"non-uniform adapter payload sizes: {payload}")
    fit_seconds = _mean_of_rows(history["fit_latency_s"])
    return next(iter(payload)), fit_seconds


def main() -> int:
    payload_bytes, fit_seconds = load_flora_inputs()

    if payload_bytes != ADAPTER_BYTES_EXPECTED:
        raise RuntimeError(
            f"Adapter payload disagreement: expected {ADAPTER_BYTES_EXPECTED}, "
            f"got {payload_bytes}. verify_comms_volume.py must be re-run."
        )

    payload_megabits = payload_bytes * 8 / 1e6  # 1 megabit = 1e6 bits
    print("Inputs:")
    print(f"  Adapter payload (bytes)        = {payload_bytes:,}")
    print(f"  Adapter payload (megabits)     = {payload_megabits:.2f}")
    print(f"  FLoRA fit time per round/client= {fit_seconds:.1f} s "
          f"(mean across {ROUNDS_PER_SESSION} rounds, history_flora_graph.json)")
    print()

    print("Upload time per client per round at WAN speed:")
    print(f"  {'speed':>10s}  {'upload_s':>10s}  {'compute/comms ratio':>20s}")
    for speed_mbps in WAN_SPEEDS_MBPS:
        upload_s = payload_megabits / speed_mbps
        ratio = fit_seconds / upload_s if upload_s > 0 else float("inf")
        speed_str = f"{speed_mbps} Mbps" if speed_mbps < 1000 else f"{speed_mbps // 1000} Gbps"
        print(f"  {speed_str:>10s}  {upload_s:>10.2f}  {ratio:>20,.0f}x")
    print()

    gpu_seconds_per_session = ROUNDS_PER_SESSION * CLIENTS_PER_SESSION * fit_seconds
    gpu_hours_per_session = gpu_seconds_per_session / 3600
    cost_low = gpu_hours_per_session * A100_USD_PER_HOUR_LOW
    cost_high = gpu_hours_per_session * A100_USD_PER_HOUR_HIGH

    annual_hours_weekly = gpu_hours_per_session * SESSIONS_PER_YEAR_WEEKLY
    annual_cost_low = annual_hours_weekly * A100_USD_PER_HOUR_LOW
    annual_cost_high = annual_hours_weekly * A100_USD_PER_HOUR_HIGH

    print("Per-session compute (10 rounds, 3 clients, fit-stage only):")
    print(f"  GPU-hours                      = {gpu_hours_per_session:.2f}")
    print(f"  USD at A100 ${A100_USD_PER_HOUR_LOW}/hr (low)     "
          f"= {cost_low:.2f}")
    print(f"  USD at A100 ${A100_USD_PER_HOUR_HIGH}/hr (high)   "
          f"= {cost_high:.2f}")
    print()
    print(f"Annual cost at weekly retraining cadence ({SESSIONS_PER_YEAR_WEEKLY} sessions/yr):")
    print(f"  GPU-hours/yr                   = {annual_hours_weekly:.0f}")
    print(f"  USD/yr (low - high)            = {annual_cost_low:,.0f} - {annual_cost_high:,.0f}")
    print()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
