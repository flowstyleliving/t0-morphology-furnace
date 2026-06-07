"""Fig 2: ANLI vs TriviaQA paired OOB AUROC per model (slope graph).

Surfaces task dependence + TriviaQA descriptive strength. Highlights
Llama-3.2-3B as the task-flip case (fails ANLI, passes TriviaQA cleanly).

Usage:
    .venv/bin/python paper/t0/figures/fig2_cross_task.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_ace_profiles import paired_by_model

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    pairs = paired_by_model()
    n = len(pairs)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    x_anli, x_triviaqa = 0.0, 1.0
    for anli, triviaqa in pairs:
        color = anli.color
        # Line connecting the pair
        ax.plot(
            [x_anli, x_triviaqa],
            [anli.oob_median, triviaqa.oob_median],
            color=color, alpha=0.7, linewidth=1.6, zorder=2,
        )
        # Endpoints (open circle for fail, filled for pass)
        for x, rec in ((x_anli, anli), (x_triviaqa, triviaqa)):
            marker = "o" if rec.passes_e_a1 else "x"
            facecolor = color if rec.passes_e_a1 else "white"
            edgecolor = color if rec.passes_e_a1 else "#888888"
            ax.scatter(
                x, rec.oob_median, s=80, marker=marker,
                facecolor=facecolor, edgecolor=edgecolor,
                linewidth=1.4, zorder=3,
            )
        # Right-side label
        ax.annotate(
            anli.model_short,
            xy=(x_triviaqa, triviaqa.oob_median),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8.5, color=color, va="center",
        )

    ax.axhline(0.50, color="#cc0000", linestyle="--", linewidth=0.9, zorder=1, alpha=0.7)
    ax.text(
        x_anli - 0.05, 0.50, "  E_A1\n  threshold",
        color="#cc0000", fontsize=7.5, va="center", ha="right",
    )

    ax.set_xticks([x_anli, x_triviaqa])
    ax.set_xticklabels(["ANLI R1 (n=200)", "TriviaQA paired (n=100)"], fontsize=10)
    ax.set_xlim(-0.35, 1.45)
    ax.set_ylim(0.35, 1.02)
    ax.set_ylabel("OOB AUROC (median)")
    ax.set_title("ACE per-model AUROC: ANLI R1 → TriviaQA")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend (manual: marker = pass/fail)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor="#444444", markeredgecolor="#444444",
               label="passes E_A1 (CI_lo > 0.50)"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=9,
               markeredgecolor="#888888", markerfacecolor="white",
               label="fails E_A1"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig2_cross_task.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close()


if __name__ == "__main__":
    main()
