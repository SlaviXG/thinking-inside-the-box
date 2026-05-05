"""Privacy-Utility-Efficiency trilemma triangle, annotated with this project's
approximate operating region.

Reproduce: python scripts/figures/plot_trilemma.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

from _style import apply_rc, ensure_res_dir


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.set_aspect("equal")

    # Equilateral triangle vertices.
    r = 1.0
    verts = {
        "Utility":    (0.0, r),
        "Privacy":    (-r * math.sin(math.radians(60)),  -r * math.cos(math.radians(60))),
        "Efficiency": (+r * math.sin(math.radians(60)),  -r * math.cos(math.radians(60))),
    }
    tri = Polygon(list(verts.values()), closed=True,
                  facecolor="#F5F7FA", edgecolor="#1F3A68", linewidth=1.4)
    ax.add_patch(tri)

    # Vertex labels with metrics.
    metrics = {
        "Utility":    "F1 score",
        "Privacy":    "MIA ROC AUC",
        "Efficiency": "MB per round",
    }
    offsets = {"Utility": (0, 0.12),
               "Privacy": (-0.12, -0.08),
               "Efficiency": (0.12, -0.08)}
    for name, (x, y) in verts.items():
        dx, dy = offsets[name]
        ax.text(x + dx, y + dy, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#1F3A68")
        ax.text(x + dx, y + dy - 0.08,
                f"({metrics[name]})",
                ha="center", va="center", fontsize=9, color="#4A5568",
                style="italic")

    def blend(a, b, t):
        return (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)

    u, p, e = verts["Utility"], verts["Privacy"], verts["Efficiency"]
    centre = ((u[0] + p[0] + e[0]) / 3, (u[1] + p[1] + e[1]) / 3)

    # This project's operating region: a concentric equilateral triangle
    # centred on the centroid, same orientation as the outer frame. The
    # concentric placement is deliberate: the thesis is that the three
    # trilemma axes can be decoupled (not traded), so the marker must not
    # bias toward any vertex.
    def concentric(scale):
        return [
            (centre[0] + (u[0] - centre[0]) * scale, centre[1] + (u[1] - centre[1]) * scale),
            (centre[0] + (p[0] - centre[0]) * scale, centre[1] + (p[1] - centre[1]) * scale),
            (centre[0] + (e[0] - centre[0]) * scale, centre[1] + (e[1] - centre[1]) * scale),
        ]

    ax.add_patch(Polygon(concentric(0.38), closed=True,
                         facecolor="#E07A1F", alpha=0.40,
                         edgecolor="#E07A1F", linewidth=1.4))
    ax.text(centre[0], centre[1],
            "This\nproject", ha="center", va="center",
            fontsize=9.5, color="#6B2E04", fontweight="bold")

    # Single-axis baselines as dots pulled well inside the triangle so they
    # sit clear of the border and vertex labels.
    baselines = [
        ("FedAvg",      blend(e, centre, 0.30), "#4A5568", (-0.02, 0.11), "right"),
        ("RAG-only",    blend(p, centre, 0.30), "#4F6DAE", (0.02, 0.11), "left"),
        ("Vanilla LLM", blend(u, centre, 0.30), "#2A9D8F", (0.0, -0.12), "center"),
    ]
    for name, (x, y), colour, (dx, dy), halign in baselines:
        ax.plot(x, y, marker="o", markersize=9, color=colour, markeredgecolor="white",
                markeredgewidth=1.2)
        ax.text(x + dx, y + dy, name, fontsize=8.5, color=colour,
                ha=halign, va="center", fontweight="bold")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.9, 1.3)
    ax.set_axis_off()

    out = ensure_res_dir() / "fig-concept-trilemma.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
