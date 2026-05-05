"""Membership inference AUC across federated conditions.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_mia_auc.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    LABEL,
    PALETTE,
    apply_rc,
    ensure_res_dir,
    flatten_client_rounds,
    load_history,
)


FEDERATED = ["fedavg_flat", "fedavg_graph", "flora_flat", "flora_graph"]


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    data = [flatten_client_rounds(load_history(c), "mia_auc") for c in FEDERATED]
    positions = list(range(1, len(FEDERATED) + 1))

    bp = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        medianprops=dict(color="#111111", linewidth=1.2),
        whiskerprops=dict(color="#333333"),
        capprops=dict(color="#333333"),
        flierprops=dict(markerfacecolor="#999", markeredgecolor="none", markersize=4),
    )
    for patch, cond in zip(bp["boxes"], FEDERATED):
        patch.set_facecolor(PALETTE[cond])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#222222")

    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.text(0.55, 0.505, "chance (0.5)", color="#555555", fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABEL[c] for c in FEDERATED], rotation=15, ha="right")
    ax.set_ylabel("MIA ROC AUC (per client x round)")
    ax.set_ylim(0.30, 0.65)

    out = ensure_res_dir() / "fig-mia-auc.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
