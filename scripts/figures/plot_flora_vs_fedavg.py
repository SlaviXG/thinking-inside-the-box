"""Two-panel visual: FedAvg parcel vs FLoRA envelope size, with byte-size labels.

Reproduce: python scripts/figures/plot_flora_vs_fedavg.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _style import FEDAVG_BYTES_PER_CLIENT_FP16, apply_rc, ensure_res_dir, load_history


def _node(ax, xy, label, colour, width=1.2, height=0.5, fontsize=10, fontcolour="white"):
    patch = FancyBboxPatch(
        (xy[0] - width / 2, xy[1] - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=colour,
    )
    ax.add_patch(patch)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=fontsize, color=fontcolour, fontweight="bold")


def _panel(ax, title, payload_size, size_label, payload_colour):
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.8)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1F3A68", pad=10)

    _node(ax, (-1.8, 0), "Client", "#1F3A68")
    _node(ax, (1.8, 0), "Server", "#4A5568")

    # Payload between the nodes, sized by payload_size (0..1 scale).
    # Floor the min width so small-payload labels are not clipped, but keep
    # the max well under the client-to-server span so arrows remain visible.
    w = max(1.05, 0.35 + payload_size * 1.55)
    h = max(0.55, 0.3 + payload_size * 0.5)
    patch = FancyBboxPatch(
        (-w / 2, -h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0, facecolor=payload_colour,
    )
    ax.add_patch(patch)
    ax.text(0, 0, size_label, ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold")

    ax.add_patch(FancyArrowPatch(
        (-1.8 + 0.65, 0), (-w / 2 - 0.08, 0),
        arrowstyle="-|>", mutation_scale=12,
        color="#333333", linewidth=1.2))
    ax.add_patch(FancyArrowPatch(
        (w / 2 + 0.08, 0), (1.8 - 0.65, 0),
        arrowstyle="-|>", mutation_scale=12,
        color="#333333", linewidth=1.2))


def main() -> None:
    apply_rc()

    # FLoRA: real encrypted payload from the archive (Fernet/base64-wrapped
    # fp32 adapter; decimal MB to match the dissertation prose).
    hist = load_history("flora_graph")
    flora_mb = hist["comm_bytes_flora"][0][0] / 1_000_000
    # FedAvg reference: 8B fp16 weights per client per round (McMahan 2017).
    # Computed from a constant so the plot does NOT inherit the archived
    # value, which was derived from a 4-bit-quantised numel multiplied by
    # num_clients and therefore misrepresents the per-client cost.
    fedavg_gb = FEDAVG_BYTES_PER_CLIENT_FP16 / 1_000_000_000
    ratio = (fedavg_gb * 1000) / flora_mb

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    _panel(axes[0], "FedAvg: full model weights",
           payload_size=1.0,
           size_label=f"~{fedavg_gb:.1f} GB",
           payload_colour="#4A5568")
    _panel(axes[1], "FLoRA: adapter deltas only",
           payload_size=0.12,
           size_label=f"~{flora_mb:.0f} MB",
           payload_colour="#E07A1F")

    fig.suptitle(f"Per-client per-round payload: ~{ratio:,.0f}x reduction",
                 fontsize=11, y=1.02)

    out = ensure_res_dir() / "fig-concept-flora-vs-fedavg.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
