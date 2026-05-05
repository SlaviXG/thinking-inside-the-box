"""High-level architecture overview: three banks, local pipelines, aggregation server.

Reproduce: python scripts/figures/plot_architecture_overview.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _style import apply_rc, ensure_res_dir


NAVY = "#1F3A68"
SLATE = "#4A5568"
TEAL = "#2A9D8F"
ORANGE = "#E07A1F"

# Geometry constants. All figure positions derive from these so that the
# privacy boundary, inner pipeline boxes, and arrow corridors stay in sync.
PRIVACY_LEFT = 0.20
PRIVACY_RIGHT = 7.05
PRIVACY_HALF_HEIGHT = 1.30

BOX_W = 2.05
BOX_H = 0.85
IBM_X = 1.25
KUZU_X = 3.55
PROMPT_X = 5.85
PROMPT_RIGHT = PROMPT_X + BOX_W / 2  # 6.875

GATHER_X = 7.95
SERVER_X = 9.95
SERVER_W = 2.80
SERVER_LEFT = SERVER_X - SERVER_W / 2   # 8.575
SERVER_TOP = 4.00
SERVER_BOT = 0.40
SERVER_CY = (SERVER_TOP + SERVER_BOT) / 2
SERVER_H = SERVER_TOP - SERVER_BOT


def _rounded(ax, xy, w, h, facecolor, alpha=1.0, edgecolor="none"):
    ax.add_patch(FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=1.0,
    ))


def _bank(ax, cy, bank_id, colour):
    privacy_w = PRIVACY_RIGHT - PRIVACY_LEFT
    ax.add_patch(Rectangle(
        (PRIVACY_LEFT, cy - PRIVACY_HALF_HEIGHT),
        privacy_w, 2 * PRIVACY_HALF_HEIGHT,
        facecolor="#FFF4E8", edgecolor=ORANGE,
        linestyle="--", linewidth=1.4, alpha=0.45,
    ))
    ax.text(PRIVACY_LEFT + 0.15, cy + 1.10, f"Bank {bank_id}: privacy boundary",
            fontsize=8.5, color="#B45A10", ha="left", fontweight="bold")

    # Pipeline trio. Box widths chosen so the longest label (the prompt box)
    # has ~0.18-unit padding on each side at fontsize 8.
    _rounded(ax, (IBM_X, cy + 0.4), BOX_W, BOX_H, "#E8EEF7")
    ax.text(IBM_X, cy + 0.4, "IBM AML\npartition", ha="center", va="center",
            fontsize=8.5, color=NAVY)

    _rounded(ax, (KUZU_X, cy + 0.4), BOX_W, BOX_H, "#E6F4F1")
    ax.text(KUZU_X, cy + 0.4, "Kuzu graph\n+ patterns", ha="center", va="center",
            fontsize=8.5, color="#155A53")

    _rounded(ax, (PROMPT_X, cy + 0.4), BOX_W, BOX_H, "#FDE8D4")
    ax.text(PROMPT_X, cy + 0.4, "Prompt + adapter\non DeepSeek 8B",
            ha="center", va="center", fontsize=8.0, color="#7A3F0F")

    fit_cx = (IBM_X + PROMPT_X) / 2
    fit_w = (PROMPT_X - IBM_X) + BOX_W * 0.4
    _rounded(ax, (fit_cx, cy - 0.6), fit_w, 0.6, colour, alpha=0.22,
             edgecolor=colour)
    ax.text(fit_cx, cy - 0.6, "Local fit / eval / MIA",
            ha="center", va="center", fontsize=8.5, color=colour)

    half = BOX_W / 2
    for (src_x, dst_x) in [(IBM_X + half, KUZU_X - half),
                           (KUZU_X + half, PROMPT_X - half)]:
        ax.add_patch(FancyArrowPatch(
            (src_x + 0.02, cy + 0.4), (dst_x - 0.02, cy + 0.4),
            arrowstyle="-|>", mutation_scale=9, color=SLATE, linewidth=1.0))


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(10.8, 6.6))

    bank_cys = (4.8, 2.2, -0.4)
    _bank(ax, bank_cys[0], 20, NAVY)
    _bank(ax, bank_cys[1], 11, TEAL)
    _bank(ax, bank_cys[2], 12, ORANGE)

    # Aggregation server. Internal text laid out so the title sits well below
    # the upper edge and well above the topmost arrow entry corridor.
    _rounded(ax, (SERVER_X, SERVER_CY), SERVER_W, SERVER_H,
             "#E8ECF3", edgecolor=SLATE)
    ax.text(SERVER_X, 3.40, "Aggregation server", ha="center", fontsize=10,
            color=SLATE, fontweight="bold")
    ax.text(SERVER_X, 2.75, "Aggregation strategy", ha="center",
            fontsize=8.5, color=SLATE)
    ax.text(SERVER_X, 2.30, "FLoRA (stacking + SVD)", ha="center",
            fontsize=8, color=ORANGE)
    ax.text(SERVER_X, 1.85, "or FedAvg (param mean)", ha="center",
            fontsize=8, color=SLATE)
    ax.text(SERVER_X, 1.05, "no raw data visible", ha="center",
            fontsize=8, color=ORANGE, style="italic")

    # Two corridors clear of the server title text.
    entry_top_y = SERVER_TOP - 0.20   # 3.80, well above bold title at 3.40
    entry_bot_y = SERVER_BOT + 0.25   # 0.65, well below italic note at 1.05
    entry_top = (SERVER_LEFT, entry_top_y)
    entry_bot = (SERVER_LEFT, entry_bot_y)

    # Encrypted-delta path: prompt-box right edge of each bank -> gather point
    # at top -> horizontal entry into server top-left.
    for cy in bank_cys:
        ax.add_patch(FancyArrowPatch(
            (PROMPT_RIGHT, cy + 0.4), (GATHER_X, entry_top_y),
            arrowstyle="-", mutation_scale=10, color=TEAL, linewidth=1.2,
            connectionstyle="arc3,rad=0",
        ))
    ax.add_patch(FancyArrowPatch(
        (GATHER_X, entry_top_y), entry_top,
        arrowstyle="-|>", mutation_scale=14, color=TEAL, linewidth=1.8,
    ))

    # Broadcast path: server bottom-left -> split point at bottom -> each bank.
    ax.add_patch(FancyArrowPatch(
        entry_bot, (GATHER_X, entry_bot_y),
        arrowstyle="-", mutation_scale=10, color=SLATE, linewidth=1.2,
    ))
    for cy in bank_cys:
        ax.add_patch(FancyArrowPatch(
            (GATHER_X, entry_bot_y), (PROMPT_RIGHT, cy + 0.4),
            arrowstyle="-|>", mutation_scale=10, color=SLATE, linewidth=1.0,
            connectionstyle="arc3,rad=0",
        ))

    # Legend at the bottom, well clear of all other elements.
    legend_y = -2.25
    ax.add_patch(FancyArrowPatch(
        (2.2, legend_y), (3.0, legend_y),
        arrowstyle="-|>", mutation_scale=12, color=TEAL, linewidth=1.8,
    ))
    ax.text(3.15, legend_y, "encrypted adapter delta (client to server)",
            ha="left", va="center", fontsize=8.5, color=SLATE)
    ax.add_patch(FancyArrowPatch(
        (2.2, legend_y - 0.50), (3.0, legend_y - 0.50),
        arrowstyle="-|>", mutation_scale=12, color=SLATE, linewidth=1.2,
    ))
    ax.text(3.15, legend_y - 0.50, "broadcast global adapter (server to client)",
            ha="left", va="center", fontsize=8.5, color=SLATE)

    ax.set_xlim(-0.2, 11.4)
    ax.set_ylim(-3.3, 6.4)
    ax.set_aspect("equal")
    ax.set_axis_off()

    out = ensure_res_dir() / "fig-arch-system.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
