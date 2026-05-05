"""Sequence-style diagram of a single federation round.

Reproduce: python scripts/figures/plot_federation_round.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import apply_rc, ensure_res_dir


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(10.4, 7.4))

    # Actors along the top.  Per-actor box widths so the longer "Aggregation
    # server" label fits with the same 0.15-unit padding the bank labels enjoy.
    actors = ["Bank 20", "Bank 11", "Bank 12", "Aggregation server"]
    xs = [3.2, 4.9, 6.6, 9.6]
    actor_widths = [1.9, 1.9, 1.9, 2.8]
    colours = ["#1F3A68", "#2A9D8F", "#E07A1F", "#4A5568"]

    y_top = 13.0
    y_bot = 0.5

    for x, name, c, w in zip(xs, actors, colours, actor_widths):
        ax.add_patch(Rectangle((x - w / 2, y_top), w, 0.6,
                               facecolor=c, edgecolor="none"))
        ax.text(x, y_top + 0.3, name, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.plot([x, x], [y_top, y_bot], color=c, linewidth=1.0,
                alpha=0.5, linestyle="--")

    # Each step is a vertical band with ~1.6 units of space so the step
    # label never collides with the step's arrows or boxes.
    band_height = 1.8

    def step_band(top, num, heading, subheading):
        centre = top - band_height / 2
        # Left-gutter step tag, anchored well to the left of all lifelines.
        ax.text(0.15, centre + 0.3, f"Step {num}", ha="left", va="center",
                fontsize=10, color="#1F3A68", fontweight="bold")
        ax.text(0.15, centre + 0.0, heading, ha="left", va="center",
                fontsize=8.5, color="#1F3A68")
        ax.text(0.15, centre - 0.35, subheading, ha="left", va="center",
                fontsize=7.8, color="#4A5568")
        return centre  # vertical centre of the band where visuals live

    # Step 1: broadcast global adapter.
    c1 = step_band(12.8, 1, "Broadcast global adapter", "(encrypted transport)")
    for i, (x, c) in enumerate(zip(xs[:3], colours[:3])):
        y = c1 + 0.45 - i * 0.30
        ax.add_patch(FancyArrowPatch((9.6, y), (x, y),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#4A5568", linewidth=1.1))
    ax.text((9.6 + xs[0]) / 2, c1 + 0.9, "global adapter (~18 MB)",
            ha="center", fontsize=8, color="#4A5568", style="italic")

    # Step 2: local fit.
    c2 = step_band(11.0, 2, "Local fit on", "bank partition")
    for x, c in zip(xs[:3], colours[:3]):
        ax.add_patch(Rectangle((x - 0.55, c2 - 0.22), 1.1, 0.5,
                               facecolor=c, alpha=0.3, edgecolor=c,
                               linewidth=0.9))
        ax.text(x, c2 + 0.03, "fit", ha="center", va="center",
                fontsize=9, color=c)

    # Step 3: eval + MIA.
    c3 = step_band(9.2, 3, "Local evaluation (F1)", "and MIA ROC AUC")
    for x, c in zip(xs[:3], colours[:3]):
        ax.add_patch(Rectangle((x - 0.65, c3 - 0.22), 1.3, 0.5,
                               facecolor=c, alpha=0.3, edgecolor=c,
                               linewidth=0.9))
        ax.text(x, c3 + 0.03, "eval + MIA", ha="center", va="center",
                fontsize=8, color=c)

    # Step 4: encrypted upload.
    c4 = step_band(7.4, 4, "Upload encrypted", "adapter delta")
    for i, (x, c) in enumerate(zip(xs[:3], colours[:3])):
        y = c4 + 0.45 - i * 0.30
        ax.add_patch(FancyArrowPatch((x, y), (9.6, y),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#2A9D8F", linewidth=1.3))
    ax.text((9.6 + xs[0]) / 2, c4 + 0.9, "encrypted delta (~18 MB)",
            ha="center", fontsize=8, color="#2A9D8F", style="italic")

    # Step 5: server aggregation.
    c5 = step_band(5.6, 5, "Server aggregates", "(FLoRA SVD or FedAvg)")
    ax.add_patch(Rectangle((8.95, c5 - 0.28), 1.3, 0.65,
                           facecolor="#4A5568", alpha=0.3,
                           edgecolor="#4A5568", linewidth=0.9))
    ax.text(9.6, c5 + 0.04, "aggregate", ha="center", va="center",
            fontsize=8.5, color="#4A5568")

    # Step 6: broadcast next round.
    c6 = step_band(3.8, 6, "Broadcast new adapter", "(loop to Step 1)")
    for i, (x, c) in enumerate(zip(xs[:3], colours[:3])):
        y = c6 + 0.45 - i * 0.30
        ax.add_patch(FancyArrowPatch((9.6, y), (x, y),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#1F3A68", linewidth=1.1))
    ax.text((9.6 + xs[0]) / 2, c6 + 0.9, "updated adapter",
            ha="center", fontsize=8, color="#1F3A68", style="italic")

    ax.text((9.6 + xs[0]) / 2, 1.3,
            "Round repeats 10 times. No raw records leave any bank.",
            ha="center", fontsize=9, color="#E07A1F", style="italic")

    ax.set_xlim(-0.2, 11.3)
    ax.set_ylim(0.3, 13.9)
    ax.set_axis_off()

    out = ensure_res_dir() / "fig-arch-federation-round.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
