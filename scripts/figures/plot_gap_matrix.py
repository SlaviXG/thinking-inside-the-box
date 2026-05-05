"""Approach-vs-axis gap matrix: which existing approaches honour which trilemma axis.

Reproduce: python scripts/figures/plot_gap_matrix.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_rc, ensure_res_dir


APPROACHES = ["Centralised LLM", "FedAvg", "FLoRA only", "RAG only", "This project"]
AXES = ["Privacy", "Utility", "Efficiency"]

# 0 = no, 0.5 = partial, 1 = yes
MATRIX = np.array([
    [0.0, 1.0, 0.5],  # Centralised LLM
    [1.0, 0.8, 0.0],  # FedAvg
    [1.0, 0.7, 1.0],  # FLoRA only
    [1.0, 0.5, 1.0],  # RAG only
    [1.0, 0.9, 1.0],  # This project
])


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(MATRIX, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i, row in enumerate(MATRIX):
        for j, v in enumerate(row):
            symbol = {0.0: "x", 0.5: "~", 1.0: "v"}.get(round(v * 2) / 2, f"{v:.1f}")
            if v >= 0.9:
                symbol = "v"
            elif v <= 0.1:
                symbol = "x"
            else:
                symbol = "~"
            # RdYlGn goes red -> yellow -> green. Yellow middle is light;
            # only the deep-red and deep-green ends can support white text.
            if v <= 0.15 or v >= 0.85:
                colour = "white"
            else:
                colour = "#111111"
            ax.text(j, i, symbol, ha="center", va="center",
                    fontsize=13, color=colour, fontweight="bold")

    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels(AXES)
    ax.set_yticks(range(len(APPROACHES)))
    ax.set_yticklabels(APPROACHES)
    ax.set_xlim(-0.5, len(AXES) - 0.5)
    ax.set_ylim(len(APPROACHES) - 0.5, -0.5)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)

    cax = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cax.set_ticks([0.0, 0.5, 1.0])
    cax.set_ticklabels(["breaks", "partial", "honours"])
    cax.ax.tick_params(labelsize=8)

    out = ensure_res_dir() / "fig-concept-gap-matrix.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
