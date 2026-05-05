"""Per-round precision, recall, and F1 side-by-side.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_precision_recall.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    CONDITIONS,
    apply_rc,
    condition_kwargs,
    ensure_res_dir,
    load_history,
    mean_per_round,
)


METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]


def main() -> None:
    apply_rc()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)

    handles, labels = None, None
    for ax, (key, title) in zip(axes, METRICS):
        for cond in CONDITIONS:
            hist = load_history(cond)
            vals = mean_per_round(hist, key)
            rounds = list(range(1, len(vals) + 1))
            kw = condition_kwargs(cond)
            kw["markersize"] = 4
            kw["linewidth"] = 1.3
            ax.plot(rounds, vals, **kw)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0.0, 1.05)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Score")

    # Shared legend below all three panels.
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), fontsize=8.5,
               handlelength=2.2, frameon=False)
    fig.subplots_adjust(bottom=0.24, wspace=0.08)

    out = ensure_res_dir() / "fig-precision-recall-f1.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
