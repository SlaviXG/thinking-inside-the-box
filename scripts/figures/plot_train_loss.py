"""Training loss per round across all conditions.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_train_loss.py
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


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    for cond in CONDITIONS:
        hist = load_history(cond)
        losses = mean_per_round(hist, "train_loss")
        rounds = list(range(1, len(losses) + 1))
        ax.plot(rounds, losses, **condition_kwargs(cond))

    ax.set_xlabel("Federation round")
    ax.set_ylabel("Training loss (mean across clients)")
    ax.set_xticks(range(1, 11))
    ax.set_yscale("log")
    ax.legend(loc="upper right", ncol=2, fontsize=8, handlelength=2.5)

    out = ensure_res_dir() / "fig-train-loss.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
