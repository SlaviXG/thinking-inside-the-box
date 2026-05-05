"""Per-round F1 curves across all six conditions.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_f1_curves.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    CONDITIONS,
    ensure_res_dir,
    apply_rc,
    condition_kwargs,
    load_history,
    mean_per_round,
    PALETTE,
)


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    peak_point = None  # (round, f1) for flora_graph
    centralised_peak = 0.0

    for cond in CONDITIONS:
        hist = load_history(cond)
        f1 = mean_per_round(hist, "f1")
        rounds = list(range(1, len(f1) + 1))
        ax.plot(rounds, f1, **condition_kwargs(cond))
        if cond == "flora_graph":
            best = max(f1)
            peak_point = (rounds[f1.index(best)], best)
        if cond == "centralised_flat":
            centralised_peak = max(f1)

    ax.axhline(centralised_peak, color=PALETTE["centralised_flat"],
               linestyle=":", linewidth=1.1, alpha=0.7)
    ax.text(10.1, centralised_peak, f" upper\n bound {centralised_peak:.3f}",
            color=PALETTE["centralised_flat"], fontsize=8, va="center")

    if peak_point is not None:
        ax.annotate(f"FLoRA+RAG peak\n{peak_point[1]:.3f} at round {peak_point[0]}",
                    xy=peak_point, xytext=(peak_point[0] - 3.0, peak_point[1] + 0.10),
                    fontsize=8, color=PALETTE["flora_graph"], ha="center",
                    arrowprops=dict(arrowstyle="->", color=PALETTE["flora_graph"],
                                    linewidth=1, shrinkA=2, shrinkB=2))

    ax.set_xlabel("Federation round")
    ax.set_ylabel("F1 score (mean across clients)")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.7, 11.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35),
              ncol=3, fontsize=8.5, handlelength=2.5, frameon=False)

    out = ensure_res_dir() / "fig-f1-rounds.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
