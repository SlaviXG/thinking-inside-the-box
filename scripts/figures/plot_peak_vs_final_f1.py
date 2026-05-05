"""Headline summary: peak and final F1 per condition.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_peak_vs_final_f1.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    CONDITIONS,
    LABEL,
    PALETTE,
    apply_rc,
    ensure_res_dir,
    load_history,
    mean_per_round,
)


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    peaks, finals, peak_rounds = [], [], []
    for cond in CONDITIONS:
        f1 = mean_per_round(load_history(cond), "f1")
        peaks.append(max(f1))
        peak_rounds.append(f1.index(max(f1)) + 1)
        finals.append(f1[-1])

    x = np.arange(len(CONDITIONS))
    w = 0.38

    ax.bar(x - w / 2, peaks, w,
           color=[PALETTE[c] for c in CONDITIONS],
           edgecolor="none", label="Peak F1")
    ax.bar(x + w / 2, finals, w,
           color=[PALETTE[c] for c in CONDITIONS],
           edgecolor="none", alpha=0.55, label="Final F1")

    for i, (p, r) in enumerate(zip(peaks, peak_rounds)):
        ax.text(i - w / 2, p + 0.02, f"{p:.2f}\nr{r}", ha="center", fontsize=7.5)
    for i, f in enumerate(finals):
        ax.text(i + w / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in CONDITIONS], rotation=20, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.1)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#555555", edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#555555", edgecolor="none", alpha=0.55),
    ]
    ax.legend(handles, ["Peak F1 (with best-round marker)", "Final F1 (round 10)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.32)

    out = ensure_res_dir() / "fig-peak-vs-final-f1.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
