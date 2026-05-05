"""Flat vs RAG ablation, paired bars showing peak and final F1 per aggregator.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_flat_vs_rag.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    AGGREGATORS,
    apply_rc,
    ensure_res_dir,
    load_history,
    mean_per_round,
)


AGG_LABEL = {"centralised": "Centralised", "fedavg": "FedAvg", "flora": "FLoRA"}


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    x = np.arange(len(AGGREGATORS))
    width = 0.2

    peak_flat, peak_graph, final_flat, final_graph = [], [], [], []
    for agg in AGGREGATORS:
        f1_flat = mean_per_round(load_history(f"{agg}_flat"), "f1")
        f1_graph = mean_per_round(load_history(f"{agg}_graph"), "f1")
        peak_flat.append(max(f1_flat))
        peak_graph.append(max(f1_graph))
        final_flat.append(f1_flat[-1])
        final_graph.append(f1_graph[-1])

    b1 = ax.bar(x - 1.5 * width, peak_flat, width, label="Peak, flat",
                color="#4F6DAE", edgecolor="none")
    b2 = ax.bar(x - 0.5 * width, peak_graph, width, label="Peak, RAG",
                color="#E07A1F", edgecolor="none", hatch="//")
    b3 = ax.bar(x + 0.5 * width, final_flat, width, label="Final, flat",
                color="#4F6DAE", alpha=0.55, edgecolor="none")
    b4 = ax.bar(x + 1.5 * width, final_graph, width, label="Final, RAG",
                color="#E07A1F", alpha=0.55, edgecolor="none", hatch="//")

    for bars in (b1, b2, b3, b4):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.01,
                    f"{rect.get_height():.2f}",
                    ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([AGG_LABEL[a] for a in AGGREGATORS])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, fontsize=8)

    out = ensure_res_dir() / "fig-flat-vs-rag.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
