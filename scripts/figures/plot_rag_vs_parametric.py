"""RAG vs parametric knowledge, two-panel contrast.

Reproduce: python scripts/figures/plot_rag_vs_parametric.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from _style import apply_rc, ensure_res_dir


NAVY = "#1F3A68"
SLATE = "#4A5568"
TEAL = "#2A9D8F"
ORANGE = "#E07A1F"


def _box(ax, xy, w, h, label, colour, alpha=1.0, fontsize=9, fontcolour="white"):
    patch = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0, facecolor=colour, alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=fontsize, color=fontcolour)


def _panel_parametric(ax):
    ax.set_title("Parametric knowledge", fontsize=11, fontweight="bold", color=NAVY, pad=8)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.2, 2.0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    _box(ax, (0, 0.1), 2.6, 1.5,
         "LLM weights\n(facts live inside)", NAVY, fontsize=10)

    # Fact "dots" inside the model, arranged as a ring around the label so
    # they read as scattered facts without overlapping the text.
    import math
    ring = [
        (math.cos(a) * rx, 0.1 + math.sin(a) * ry)
        for a, rx, ry in (
            (math.radians(deg), 1.05, 0.52)
            for deg in (20, 55, 90, 125, 160, 200, 235, 270, 305, 340)
        )
    ]
    for x, y in ring:
        ax.plot(x, y, "o", color="white", markersize=3)

    _box(ax, (-2.0, -1.55), 1.0, 0.45, "Query", SLATE, fontsize=9)
    _box(ax, ( 2.0, -1.55), 1.0, 0.45, "Answer", TEAL, fontsize=9)
    ax.add_patch(FancyArrowPatch((-1.5, -1.55), (-0.9, -0.7),
                                 arrowstyle="-|>", mutation_scale=10, color=SLATE))
    ax.add_patch(FancyArrowPatch((0.9, -0.7), (1.5, -1.55),
                                 arrowstyle="-|>", mutation_scale=10, color=TEAL))
    ax.text(0, -2.05, "knowledge entangled with parameters",
            ha="center", fontsize=8.5, color=SLATE, style="italic")


def _panel_rag(ax):
    ax.set_title("Retrieval-augmented (this project)", fontsize=11,
                 fontweight="bold", color=NAVY, pad=8)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.2, 2.0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    _box(ax, (0.4, 0.7), 2.3, 1.0, "LLM weights\n(reasoning only)", NAVY, fontsize=10)
    _box(ax, (-1.8, -0.2), 1.5, 1.2, "Graph store\n+ pattern\ndetector", ORANGE, fontsize=9)

    _box(ax, (-2.5, -1.55), 1.0, 0.45, "Query", SLATE, fontsize=9)
    _box(ax, ( 2.5, -1.55), 1.0, 0.45, "Answer", TEAL, fontsize=9)

    ax.add_patch(FancyArrowPatch((-2.0, -1.33), (-1.8, -0.85),
                                 arrowstyle="-|>", mutation_scale=10, color=SLATE))
    ax.add_patch(FancyArrowPatch((-1.0, 0.1), (-0.4, 0.55),
                                 arrowstyle="-|>", mutation_scale=10, color=ORANGE))
    ax.add_patch(FancyArrowPatch((1.3, 0.4), (2.2, -1.33),
                                 arrowstyle="-|>", mutation_scale=10, color=TEAL))

    ax.text(0, -2.05, "knowledge stays local, reasoning stays shared",
            ha="center", fontsize=8.5, color="#B45A10", style="italic")


def main() -> None:
    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    _panel_parametric(axes[0])
    _panel_rag(axes[1])

    out = ensure_res_dir() / "fig-concept-rag-vs-parametric.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
