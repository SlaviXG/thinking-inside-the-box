"""Objectives-vs-achievement radar for the Conclusions chapter.

Reproduce: python scripts/figures/plot_objectives_radar.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_rc, ensure_res_dir


OBJECTIVES = [
    "OI Architecture\n& backends",
    "OII Six-condition\nevaluation",
    "OIII FL within 94%\nof centralised",
    "OIV Privacy (MIA\naround chance)",
    "OV Reproducible\nrelease",
]
ACHIEVEMENT = [1.00, 1.00, 0.94, 0.95, 1.00]
TARGET = [1.0] * len(OBJECTIVES)


def main() -> None:
    apply_rc()
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111, projection="polar")
    # OI starts at the top and objectives progress clockwise.
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    n = len(OBJECTIVES)
    angles = [i * 2 * math.pi / n for i in range(n)] + [0.0]

    target = TARGET + [TARGET[0]]
    achieved = ACHIEVEMENT + [ACHIEVEMENT[0]]

    ax.plot(angles, target, color="#4A5568", linestyle="--", linewidth=1.3,
            label="Target (100%)")
    ax.fill(angles, achieved, color="#2A9D8F", alpha=0.25)
    ax.plot(angles, achieved, color="#2A9D8F", linewidth=1.8,
            marker="o", markersize=6, label="Achieved")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(OBJECTIVES, fontsize=9, color="#1F3A68")
    # Push labels outward to keep them clear of the outer ring.
    ax.tick_params(axis="x", which="major", pad=18)

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7.5,
                       color="#4A5568")
    ax.set_rlabel_position(36)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=9, frameon=False)

    fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.14)

    out = ensure_res_dir() / "fig-concept-objectives-radar.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
