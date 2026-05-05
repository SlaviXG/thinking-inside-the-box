"""Shared style module for dissertation figures.

Every script under scripts/figures/ should import from this module so the
palette, typography, and grid treatment stay identical across all plots.

Reproduce: git checkout benchmark-2026-04-02 && python scripts/figures/_style.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Repo-root-relative paths.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "2026-04-02"
RES_DIR = ROOT / "assets" / "CSC3094_Dissertation" / "res"

PALETTE = {
    "centralised_flat":  "#1F3A68",
    "centralised_graph": "#4F6DAE",
    "fedavg_flat":       "#4A5568",
    "fedavg_graph":      "#8390A3",
    "flora_flat":        "#2A9D8F",
    "flora_graph":       "#E07A1F",
}

LABEL = {
    "centralised_flat":  "Centralised, flat",
    "centralised_graph": "Centralised, RAG",
    "fedavg_flat":       "FedAvg, flat",
    "fedavg_graph":      "FedAvg, RAG",
    "flora_flat":        "FLoRA, flat",
    "flora_graph":       "FLoRA, RAG",
}

MARKERS = {
    "centralised_flat":  "o",
    "centralised_graph": "o",
    "fedavg_flat":       "s",
    "fedavg_graph":      "s",
    "flora_flat":        "^",
    "flora_graph":       "^",
}

LINESTYLE_BY_RETRIEVAL = {"flat": "-", "graph": "--"}

AGGREGATORS = ["centralised", "fedavg", "flora"]
RETRIEVALS = ["flat", "graph"]
CONDITIONS = [f"{a}_{r}" for a in AGGREGATORS for r in RETRIEVALS]

# DeepSeek-R1-Distill-Llama-8B has the Llama-3.1-8B architecture (untied
# embeddings), total 8,030,261,248 parameters. A classical McMahan-style
# FedAvg round transmits the full backbone per client per round; at fp16
# precision this is ~16 GB per client, the honest reference against which
# FLoRA's encrypted ~18 MB adapter delta is compared. The archived history
# field `comm_bytes_fedavg_per_round` underestimates this because it uses
# `sum(p.numel())` on the 4-bit quantised model (packed int4 tensors have
# roughly half the numel of the nominal fp16 model) and multiplies by
# num_clients; both are corrected for in the plots below.
BACKBONE_PARAMS = 8_030_261_248
FEDAVG_BYTES_PER_CLIENT_FP16 = BACKBONE_PARAMS * 2


def apply_rc() -> None:
    """Apply the shared matplotlib rcParams. Call once per script."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Palatino", "Times New Roman", "serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.figsize": (6.0, 3.5),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_history(condition: str) -> dict:
    with (DATA_DIR / f"history_{condition}.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_config(condition: str) -> dict:
    with (DATA_DIR / f"config_{condition}.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_res_dir() -> Path:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    return RES_DIR


def condition_kwargs(condition: str) -> dict:
    retrieval = "graph" if condition.endswith("_graph") else "flat"
    return dict(
        color=PALETTE[condition],
        marker=MARKERS[condition],
        linestyle=LINESTYLE_BY_RETRIEVAL[retrieval],
        label=LABEL[condition],
        linewidth=1.6,
        markersize=5,
    )


def mean_per_round(history: dict, key: str) -> list[float]:
    """Average metric across clients for each round."""
    rows = history[key]
    out = []
    for row in rows:
        if isinstance(row, list):
            out.append(sum(row) / len(row))
        else:
            out.append(float(row))
    return out


def flatten_client_rounds(history: dict, key: str) -> list[float]:
    """Flatten per-round per-client values into a single list."""
    out = []
    for row in history[key]:
        if isinstance(row, list):
            out.extend(row)
        else:
            out.append(float(row))
    return out


def _self_test() -> None:
    print("Palette:")
    for k, v in PALETTE.items():
        print(f"  {k:22s} {v}")
    print("Data dir:", DATA_DIR)
    print("Res dir:", RES_DIR)
    assert DATA_DIR.exists(), f"missing {DATA_DIR}"
    for c in CONDITIONS:
        h = load_history(c)
        assert "f1" in h
    print(f"Loaded {len(CONDITIONS)} histories OK")


if __name__ == "__main__":
    apply_rc()
    _self_test()
    ensure_res_dir()
    fig, ax = plt.subplots(figsize=(6, 1.2))
    for i, c in enumerate(CONDITIONS):
        ax.barh(i, 1, color=PALETTE[c])
        ax.text(0.5, i, LABEL[c], ha="center", va="center", color="white", fontsize=9)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.suptitle("Palette preview", fontsize=10)
    out = ensure_res_dir() / "_palette_preview.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")
