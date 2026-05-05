"""Per-node privacy boundary cartoon: what stays inside, what crosses the wire.

Reproduce: python scripts/figures/plot_privacy_boundary.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from _style import apply_rc, ensure_res_dir


NAVY = "#1F3A68"
SLATE = "#4A5568"
TEAL = "#2A9D8F"
ORANGE = "#E07A1F"


def _box(ax, xy, w, h, label, colour, alpha=1.0, fontcolour="white", fontsize=9.5):
    patch = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0, facecolor=colour, alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=fontsize, color=fontcolour)


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.set_xlim(-4.5, 7.5)
    ax.set_ylim(-4.1, 3.2)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Privacy boundary.
    boundary_left, boundary_right = -4.0, 4.2
    boundary_bot, boundary_top = -2.7, 2.7
    ax.add_patch(Rectangle(
        (boundary_left, boundary_bot),
        boundary_right - boundary_left, boundary_top - boundary_bot,
        linewidth=1.8, edgecolor=ORANGE, linestyle="--",
        facecolor="#FFF4E8", alpha=0.35,
    ))
    ax.text((boundary_left + boundary_right) / 2, boundary_top - 0.3,
            "Client privacy boundary (bank node)",
            ha="center", fontsize=10, color="#B45A10", fontweight="bold")

    _box(ax, (-2.8,  1.2), 1.8, 0.7, "Transactions\n(IBM AML)", NAVY, fontsize=8.5)
    _box(ax, (-0.6,  1.2), 1.8, 0.7, "Graph store\n(Kuzu)", NAVY, fontsize=8.5)
    _box(ax, ( 1.6,  1.2), 2.2, 0.7, "Pattern\ndetector", NAVY, fontsize=8.5)
    _box(ax, (-2.2, -0.5), 1.9, 0.7, "Prompt\nbuilder", TEAL, fontsize=8.5)
    _box(ax, ( 0.4, -0.5), 2.3, 0.7, "DeepSeek R1 8B\n(frozen)", SLATE, fontsize=8.5)
    _box(ax, ( 2.7, -0.5), 1.1, 0.7, "Local\nadapter", TEAL, fontsize=8.5)
    _box(ax, ( 0.0, -1.85), 2.0, 0.6, "Verdict + MIA", NAVY, fontsize=8.5)

    arrows = [
        ((-2.8, 0.85), (-2.2, -0.15)),
        ((-0.6, 0.85), (-2.2, -0.15)),
        ((1.6, 0.85), (-2.2, -0.15)),
        ((-1.25, -0.5), (-0.75, -0.5)),
        ((1.55, -0.5), (2.15, -0.5)),
        ((0.4, -0.85), (0.0, -1.55)),
    ]
    for src, dst in arrows:
        ax.add_patch(FancyArrowPatch(src, dst, arrowstyle="-|>",
                                     mutation_scale=10, color=SLATE, linewidth=1.1))

    # Server outside the boundary (clearly to the right of the dashed box).
    _box(ax, (6.1, -0.5), 1.8, 0.7, "Aggregation\nserver", SLATE, fontsize=8.5)

    # Encrypted delta crossing the boundary with the label sitting well
    # above the arrow, clear of both the Local adapter and the server.
    ax.add_patch(FancyArrowPatch(
        (3.28, -0.5), (5.32, -0.5),
        arrowstyle="-|>", mutation_scale=14,
        color=TEAL, linewidth=1.8,
        connectionstyle="arc3,rad=-0.25",
    ))
    ax.text(4.3, 0.35, "encrypted\nadapter delta",
            fontsize=8.5, color="#155A53", ha="center", va="center",
            fontweight="bold")

    # "Never leaves" placed as a caption OUTSIDE the boundary box so it
    # acts like a legend and cannot collide with the Verdict + MIA block.
    ax.text((boundary_left + boundary_right) / 2, boundary_bot - 0.55,
            "Never leaves the boundary: raw records, account IDs, "
            "per-record features, client-local embeddings, aggregate statistics.",
            ha="center", va="top", fontsize=8.5, color="#B45A10", style="italic")

    out = ensure_res_dir() / "fig-concept-privacy-boundary.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
