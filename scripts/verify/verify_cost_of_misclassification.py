"""Verify the cost-of-misclassification figures used in the Results chapter.

The dissertation does not claim a £/year saving for any specific bank. It
projects, per million transactions monitored, the false-positive count and
the implied investigation cost, using:

- Per-condition precision and recall at peak F1 (measured, archived JSON).
- The IBM AML positive-class rate p_true (measured, archived dataset).
- An industry-standard per-alert investigation cost range cited in the
  LexisNexis True Cost of Financial Crime Compliance reports and aligned
  industry sources (USD 30-70 per alert).

Outputs:
- For each of the six benchmark conditions, at the peak F1 round:
    * mean precision and recall across clients
    * implied false positive count per 1,000,000 monitored transactions
    * implied false negative count per 1,000,000 monitored transactions
    * annualised investigation cost low/high bound
- A summary delta of FLoRA-with-RAG vs FedAvg-flat (the practical baseline).

Run: python scripts/verify/verify_cost_of_misclassification.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "2026-04-02"

# Empirical positive-class rate in the IBM AML small-scale partitions used
# in the benchmark. The dissertation states ~1-2% in main.tex; the script
# uses the midpoint as a single representative value.
P_TRUE = 0.015

# Per-alert investigation cost in USD. Range citation:
#   LexisNexis Risk Solutions, True Cost of Financial Crime Compliance
#   (UK 2024) and aligned industry analyses report per-alert investigation
#   costs in the USD 30-70 band, with 85-95% of alerts ultimately false
#   positive. This script uses the cited range as low/high bounds.
COST_PER_ALERT_USD_LOW = 30
COST_PER_ALERT_USD_HIGH = 70

# Reference monitored-transaction volume. The dissertation reports figures
# per 1,000,000 transactions monitored, leaving the reader free to scale to
# their own institution's volume.
REFERENCE_TXN_VOLUME = 1_000_000

CONDITIONS = [
    "centralised_flat",
    "centralised_graph",
    "fedavg_flat",
    "fedavg_graph",
    "flora_flat",
    "flora_graph",
]


def _mean_per_round(rows):
    out = []
    for row in rows:
        if isinstance(row, list):
            out.append(sum(row) / len(row))
        else:
            out.append(float(row))
    return out


def peak_round_pr(condition: str) -> tuple[int, float, float, float]:
    with (DATA_DIR / f"history_{condition}.json").open("r", encoding="utf-8") as fh:
        history = json.load(fh)
    f1 = _mean_per_round(history["f1"])
    p = _mean_per_round(history["precision"])
    r = _mean_per_round(history["recall"])
    peak_idx = f1.index(max(f1))
    return peak_idx + 1, f1[peak_idx], p[peak_idx], r[peak_idx]


def fp_fn_per_volume(precision: float, recall: float, txn_volume: int,
                     p_true: float) -> tuple[float, float]:
    """Project false-positive and false-negative counts at a fixed volume.

    Assumes the model's precision and recall on the IBM AML test set
    transfer to a real monitoring stream at the same positive-class rate.
    This is a projection, not a measurement; the dissertation labels it
    as such.
    """
    true_positives = txn_volume * p_true * recall
    if precision <= 0:
        return float("inf"), txn_volume * p_true
    false_positives = true_positives * (1 - precision) / precision
    false_negatives = txn_volume * p_true * (1 - recall)
    return false_positives, false_negatives


def main() -> int:
    print("Inputs:")
    print(f"  P_TRUE                        = {P_TRUE} (positive-class rate, IBM AML small-scale)")
    print(f"  Cost per alert (USD)          = [{COST_PER_ALERT_USD_LOW}, {COST_PER_ALERT_USD_HIGH}] "
          f"(LexisNexis True Cost of FC Compliance UK 2024 and aligned sources)")
    print(f"  Reference monitored volume    = {REFERENCE_TXN_VOLUME:,} transactions")
    print()

    rows = []
    for cond in CONDITIONS:
        peak_round, f1, p, r = peak_round_pr(cond)
        fp, fn = fp_fn_per_volume(p, r, REFERENCE_TXN_VOLUME, P_TRUE)
        cost_low = fp * COST_PER_ALERT_USD_LOW
        cost_high = fp * COST_PER_ALERT_USD_HIGH
        rows.append((cond, peak_round, f1, p, r, fp, fn, cost_low, cost_high))

    header = (
        f"{'condition':22s} {'peak_rnd':>9s} {'F1':>6s} {'P':>6s} {'R':>6s} "
        f"{'FP/1M':>10s} {'FN/1M':>8s} {'cost_low_USD':>14s} {'cost_high_USD':>15s}"
    )
    print(header)
    print("-" * len(header))
    for cond, rnd, f1, p, r, fp, fn, cl, ch in rows:
        print(f"{cond:22s} {rnd:>9d} {f1:>6.3f} {p:>6.3f} {r:>6.3f} "
              f"{fp:>10,.0f} {fn:>8,.0f} {cl:>14,.0f} {ch:>15,.0f}")
    print()

    by_name = {row[0]: row for row in rows}
    fl_rag = by_name["flora_graph"]
    fa_flat = by_name["fedavg_flat"]
    fp_fl = fl_rag[5]
    fp_fa = fa_flat[5]
    cost_high_fl = fl_rag[8]
    cost_high_fa = fa_flat[8]
    print("Headline comparison (peak F1 round):")
    print(f"  flora_graph  FP/1M = {fp_fl:,.0f}, fedavg_flat FP/1M = {fp_fa:,.0f}")
    if fp_fa > 0:
        delta_pct = (fp_fl - fp_fa) / fp_fa * 100
        print(f"  FLoRA-with-RAG vs FedAvg-flat: FP delta = {delta_pct:+.1f}%")
    print(f"  Investigation cost upper bound (USD per 1M txn): "
          f"flora_graph = {cost_high_fl:,.0f}, fedavg_flat = {cost_high_fa:,.0f}")
    print()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
