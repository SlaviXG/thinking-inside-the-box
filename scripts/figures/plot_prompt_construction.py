"""Flat vs RAG prompt construction: raw transactions + pattern detector -> fused prompt.

Reproduce: python scripts/figures/plot_prompt_construction.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import apply_rc, ensure_res_dir


NAVY = "#1F3A68"
SLATE = "#4A5568"
TEAL = "#2A9D8F"
ORANGE = "#E07A1F"


def _box(ax, xy, w, h, text, facecolor, fontcolour="white",
         fontsize=8.5, ha="center", va="center"):
    ax.add_patch(FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=facecolor, linewidth=0,
    ))
    ax.text(xy[0], xy[1], text, ha=ha, va=va,
            fontsize=fontsize, color=fontcolour)


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(11.2, 4.6))

    # Source box on the left (taller, shows sample records).
    _box(ax, (1.6, 2.5), 2.6, 2.6,
         "Raw transactions\n(bank subgraph)\n\n"
         "- acct_A -> acct_B  12k\n"
         "- acct_B -> acct_C   9k\n"
         "- acct_C -> acct_A  11k",
         NAVY, fontsize=7.5)

    # Flat-path box (top). Wider so label fits.
    _box(ax, (5.6, 4.0), 3.8, 0.7,
         "Flat: copy transactions verbatim",
         SLATE, fontsize=9)

    # Pattern detector (middle).
    _box(ax, (5.6, 2.5), 3.8, 0.9,
         "Pattern detector\n(src/graph/patterns.py)",
         TEAL, fontsize=9)

    # Named signals (bottom). Split across two lines so the string fits
    # inside the box with padding.
    _box(ax, (5.6, 1.0), 3.8, 0.9,
         "Named signals:\npass-through, cycle, burst",
         ORANGE, fontsize=8.5)

    # Fused prompt on the right. Wider so no line is clipped.
    _box(ax, (9.8, 2.5), 2.8, 3.2,
         "Fused prompt\n\n"
         "Transactions: ...\n"
         "Signals: structuring,\n"
         "  intra-bank cycle\n\n"
         "Verdict:",
         NAVY, fontsize=7.5)

    # Arrows.
    # Source -> flat-path
    ax.add_patch(FancyArrowPatch((2.9, 2.9), (3.7, 3.95),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=SLATE, linewidth=1.2))
    # Source -> pattern detector
    ax.add_patch(FancyArrowPatch((2.9, 2.5), (3.7, 2.5),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=SLATE, linewidth=1.2))
    # Pattern detector -> named signals
    ax.add_patch(FancyArrowPatch((5.6, 2.05), (5.6, 1.35),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=SLATE, linewidth=1.2))
    # Flat-path -> fused prompt
    ax.add_patch(FancyArrowPatch((7.5, 4.0), (8.4, 3.3),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=SLATE, linewidth=1.2))
    # Named signals -> fused prompt
    ax.add_patch(FancyArrowPatch((7.5, 1.0), (8.4, 1.7),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=ORANGE, linewidth=1.4))

    # Lane labels - positioned well above/below boxes so they don't touch anything.
    ax.text(5.6, 4.75, "Flat path", ha="center", fontsize=10, color=SLATE,
            fontweight="bold")
    ax.text(5.6, 0.25, "RAG path (graph-augmented)", ha="center",
            fontsize=10, color=ORANGE, fontweight="bold")

    ax.set_xlim(-0.2, 11.5)
    ax.set_ylim(-0.4, 5.3)
    ax.set_aspect("equal")
    ax.set_axis_off()

    out = ensure_res_dir() / "fig-arch-prompt-construction.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
