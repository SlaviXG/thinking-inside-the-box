"""UML-style class diagram for the Strategy pattern around GraphStore.

Reproduce: python scripts/figures/plot_strategy_pattern.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import apply_rc, ensure_res_dir


NAVY = "#1F3A68"
SLATE = "#4A5568"
TEAL = "#2A9D8F"
ORANGE = "#E07A1F"


def _class_box(ax, xy, w, h, name, methods, colour=NAVY, italic=False):
    x, y = xy
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                           facecolor="white", edgecolor=colour, linewidth=1.3))
    # Header strip
    ax.add_patch(Rectangle((x - w / 2, y + h / 2 - 0.35), w, 0.35,
                           facecolor=colour, edgecolor=colour, linewidth=0))
    style = "italic" if italic else "normal"
    ax.text(x, y + h / 2 - 0.17, name, ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", style=style)

    for i, m in enumerate(methods):
        ax.text(x - w / 2 + 0.08, y + h / 2 - 0.5 - i * 0.22,
                m, ha="left", va="center", fontsize=8.3, color="#111111",
                family="monospace")


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(10.6, 5.0))

    iface_methods = [
        "+ ingest(transactions)",
        "+ retrieve_context(acct, mode)",
        "+ structural_signals(acct)",
        "+ close()",
    ]

    _class_box(ax, (0, 2.2), 4.6, 1.8, "GraphStore <<interface>>",
               iface_methods, colour=NAVY, italic=True)

    _class_box(ax, (-4.6, -1.4), 3.8, 1.3, "KuzuGraphStore",
               ["+ ingest(...)", "+ retrieve_context(...)",
                "+ structural_signals(...)"], colour=ORANGE)
    _class_box(ax, (0.0, -1.4), 3.8, 1.3, "NetworkXGraphStore",
               ["+ ingest(...)", "+ retrieve_context(...)",
                "+ structural_signals(...)"], colour=TEAL)
    _class_box(ax, (4.6, -1.4), 3.8, 1.3, "Neo4jGraphStore",
               ["+ ingest(...)", "+ retrieve_context(...)",
                "+ structural_signals(...)"], colour=SLATE)

    # Inheritance arrows
    for cx in (-4.6, 0.0, 4.6):
        ax.add_patch(FancyArrowPatch(
            (cx, -0.75), (cx * 0.35, 1.35),
            arrowstyle="-|>", mutation_scale=14, color=NAVY, linewidth=1.1))

    # Client / Factory
    _class_box(ax, (-6.0, 2.6), 4.0, 1.1, "Factory",
               ["+ make(cfg) -> GraphStore"], colour=SLATE)
    ax.add_patch(FancyArrowPatch((-4.0, 2.6), (-2.45, 2.3),
                                 arrowstyle="->", mutation_scale=10, color=SLATE,
                                 linestyle="--"))
    ax.text(-3.2, 2.85, "uses", fontsize=8.5, color=SLATE, style="italic",
            ha="center")

    _class_box(ax, (6.0, 2.6), 4.0, 1.1, "Federation layer",
               ["depends only on GraphStore"], colour=NAVY)
    ax.add_patch(FancyArrowPatch((4.0, 2.6), (2.45, 2.3),
                                 arrowstyle="->", mutation_scale=10, color=NAVY,
                                 linestyle="--"))
    ax.text(3.2, 2.85, "uses", fontsize=8.5, color=NAVY, style="italic",
            ha="center")

    ax.set_xlim(-8.3, 8.3)
    ax.set_ylim(-2.5, 3.5)
    ax.set_aspect("equal")
    ax.set_axis_off()

    out = ensure_res_dir() / "fig-arch-strategy-pattern.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
