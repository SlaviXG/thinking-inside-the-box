"""Communication volume per round: FLoRA vs FedAvg, log scale.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_comms_volume.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    FEDAVG_BYTES_PER_CLIENT_FP16,
    PALETTE,
    apply_rc,
    ensure_res_dir,
    load_history,
)


def main() -> None:
    apply_rc()

    # FLoRA: measured encrypted payload from the archive (decimal MB).
    hist = load_history("flora_graph")
    flora_bytes = hist["comm_bytes_flora"][0][0]
    # FedAvg reference: 8B fp16 weights per client per round (McMahan 2017).
    # The archived comm_bytes_fedavg_per_round is an aggregate-across-clients
    # figure computed on the quantised numel; the per-client fp16 reference
    # used here is the honest comparison.
    fedavg_bytes = FEDAVG_BYTES_PER_CLIENT_FP16

    flora_mb = flora_bytes / 1_000_000
    fedavg_mb = fedavg_bytes / 1_000_000
    ratio = fedavg_mb / flora_mb

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    bars = ax.bar(
        [0, 1],
        [flora_mb, fedavg_mb],
        color=[PALETTE["flora_graph"], PALETTE["fedavg_flat"]],
        width=0.55,
        edgecolor="none",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FLoRA adapter deltas", "FedAvg full weights"])
    ax.set_ylabel("MB per client per round (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(5, fedavg_mb * 3)

    for bar, mb in zip(bars, [flora_mb, fedavg_mb]):
        label = f"{mb:,.1f} MB" if mb < 1000 else f"{mb/1000:,.1f} GB"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                label, ha="center", fontsize=9)

    ax.annotate(f"{ratio:,.0f}x reduction",
                xy=(0.5, flora_mb * 12), xycoords="data",
                ha="center", fontsize=10, color="#333333")

    out = ensure_res_dir() / "fig-comms-volume.pdf"
    fig.savefig(out)
    print(f"Wrote {out} (FLoRA {flora_mb:.2f} MB, FedAvg {fedavg_mb/1000:.2f} GB, ratio {ratio:.0f}x)")


if __name__ == "__main__":
    main()
