"""Fig 1: ACE OOB AUROC on ANLI R1, 9 models with 95% CIs + 0.50 threshold line.

Primary E_A1 figure. Pass = OOB CI_lo > 0.50 (colored). Fail = grey.

Usage:
    PYTHONUNBUFFERED=1 .venv/bin/python -u paper/t0/figures/fig1_anli_auroc.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_ace_profiles import by_dataset

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    records = by_dataset("anli")
    models = [r.model_short for r in records]
    medians = np.array([r.oob_median for r in records])
    ci_lo = np.array([r.oob_ci_lo for r in records])
    ci_hi = np.array([r.oob_ci_hi for r in records])
    err_lo = medians - ci_lo
    err_hi = ci_hi - medians
    colors = [r.color if r.passes_e_a1 else "#bbbbbb" for r in records]
    edge = ["#222222" if r.passes_e_a1 else "#888888" for r in records]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(models))
    ax.bar(x, medians, color=colors, edgecolor=edge, linewidth=0.8, zorder=2)
    ax.errorbar(
        x, medians, yerr=[err_lo, err_hi], fmt="none",
        ecolor="#222222", capsize=4, capthick=1.0, linewidth=1.0, zorder=3,
    )
    ax.axhline(0.50, color="#cc0000", linestyle="--", linewidth=1.0, zorder=1, label="E_A1 threshold (CI_lo > 0.50)")

    pass_count = sum(1 for r in records if r.passes_e_a1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("OOB AUROC (median, 95% CI)")
    ax.set_title(f"ACE on ANLI R1 (n=200): {pass_count}/9 pass E_A1")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.legend(loc="lower right", frameon=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig1_anli_auroc.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close()


if __name__ == "__main__":
    main()
