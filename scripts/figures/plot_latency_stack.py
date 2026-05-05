"""Wall-clock latency per round, broken down into fit / eval / MIA components.

Centralised conditions have no MIA pass, so their MIA contribution is zero.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/plot_latency_stack.py
"""

from __future__ import annotations

from statistics import mean

import matplotlib.pyplot as plt

from _style import (
    CONDITIONS,
    LABEL,
    PALETTE,
    apply_rc,
    ensure_res_dir,
    load_history,
)


def main() -> None:
    apply_rc()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    fit_vals = []
    eval_vals = []
    mia_vals = []
    totals = []
    for cond in CONDITIONS:
        hist = load_history(cond)
        fit = mean(hist["fit_latency_s"])
        ev = mean(hist["eval_latency_s"])
        mia = mean(hist["mia_latency_s"]) if "mia_latency_s" in hist else 0.0
        fit_vals.append(fit)
        eval_vals.append(ev)
        mia_vals.append(mia)
        totals.append(fit + ev + mia)

    xs = list(range(len(CONDITIONS)))
    bar_kw = dict(width=0.6, edgecolor="none")
    ax.bar(xs, fit_vals, color="#2A9D8F", label="Fit", **bar_kw)
    ax.bar(xs, eval_vals, bottom=fit_vals, color="#4F6DAE", label="Eval", **bar_kw)
    ax.bar(xs, mia_vals,
           bottom=[f + e for f, e in zip(fit_vals, eval_vals)],
           color="#E07A1F", label="MIA", **bar_kw)

    for i, total in enumerate(totals):
        ax.text(i, total + max(totals) * 0.02,
                f"{total:.0f}s", ha="center", fontsize=8.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[c] for c in CONDITIONS], rotation=20, ha="right")
    ax.set_ylabel("Seconds per round (mean of 10)")
    ax.legend(loc="upper left", fontsize=9)

    out = ensure_res_dir() / "fig-latency-stack.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
