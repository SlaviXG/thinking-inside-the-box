"""Per-client F1 trajectory for the best federated condition (flora_graph).

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_per_client_f1.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    apply_rc,
    ensure_res_dir,
    load_config,
    load_history,
)


CLIENT_COLOURS = ["#1F3A68", "#2A9D8F", "#E07A1F"]


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    cond = "flora_graph"
    hist = load_history(cond)
    cfg = load_config(cond)
    bank_ids = cfg.get("bank_ids", [0, 1, 2])

    f1_matrix = hist["f1"]  # rounds x clients
    rounds = list(range(1, len(f1_matrix) + 1))
    n_clients = len(f1_matrix[0])

    for c in range(n_clients):
        series = [r[c] for r in f1_matrix]
        ax.plot(rounds, series, marker="o", color=CLIENT_COLOURS[c % 3],
                linewidth=1.4, markersize=4,
                label=f"Bank {bank_ids[c]}")

    mean_series = [sum(r) / len(r) for r in f1_matrix]
    ax.plot(rounds, mean_series, color="#333333", linestyle="--",
            linewidth=1.2, label="Mean")

    ax.set_xlabel("Federation round")
    ax.set_ylabel("F1 score")
    ax.set_title("FLoRA + RAG: per-client F1 trajectory")
    ax.set_xticks(rounds)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=8)

    out = ensure_res_dir() / "fig-per-client-f1.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
