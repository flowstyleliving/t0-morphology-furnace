"""Fig 3: Cell-transfer matrix — ANLI winner vs TriviaQA winner per model.

Visual carrier for E_A2 (3/9 exact transfer, 6/9 block-stable). Each row =
one model. Columns: ANLI winner cell · TriviaQA winner cell · exact-transfer
flag · block-stable flag.

Usage:
    .venv/bin/python paper/t0/figures/fig3_transfer_matrix.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from load_ace_profiles import paired_by_model

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def _fmt_cell(rec) -> str:
    sign = "+" if rec.sign > 0 else "−"
    return f"{rec.block_prefix} · {rec.metric_name} · {sign}"


def main() -> None:
    pairs = paired_by_model()
    n = len(pairs)

    # Columns: model | ANLI winner | TriviaQA winner | exact | block-stable
    col_labels = ["Model", "ANLI winner (block · metric · sign)",
                  "TriviaQA winner (block · metric · sign)", "Exact", "Block-stable"]
    col_x = [0.00, 0.18, 0.56, 0.86, 0.94]  # column left edges (in axis fraction)

    fig, ax = plt.subplots(figsize=(11.0, 0.45 * (n + 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1)
    ax.axis("off")

    # Header row (top, y = n + 0.5)
    y_header = n + 0.5
    for x, label in zip(col_x, col_labels):
        ax.text(x, y_header, label, fontsize=10, fontweight="bold",
                va="center", ha="left")
    ax.add_patch(Rectangle((0, n + 0.1), 1.0, 0.02, color="#222222", transform=ax.transData))

    exact_count = 0
    block_count = 0

    # Data rows (top-to-bottom = canonical order)
    for i, (anli, triviaqa) in enumerate(pairs):
        y = n - i  # row position
        exact = anli.winner_triple == triviaqa.winner_triple
        block_stable = anli.block_prefix == triviaqa.block_prefix
        if exact:
            exact_count += 1
        if block_stable:
            block_count += 1

        # Soft row background for exact-transfer rows
        if exact:
            ax.add_patch(Rectangle((0, y - 0.45), 1.0, 0.9,
                                   color="#e6f4ea", zorder=0,
                                   transform=ax.transData))

        ax.text(col_x[0], y, anli.model_short, fontsize=9.5,
                color=anli.color, fontweight="bold", va="center")
        ax.text(col_x[1], y, _fmt_cell(anli), fontsize=9, va="center",
                family="monospace")
        ax.text(col_x[2], y, _fmt_cell(triviaqa), fontsize=9, va="center",
                family="monospace")
        ax.text(col_x[3], y, "✓" if exact else "—", fontsize=12,
                va="center", ha="left",
                color="#177245" if exact else "#999999",
                fontweight="bold")
        ax.text(col_x[4], y, "✓" if block_stable else "—", fontsize=12,
                va="center", ha="left",
                color="#177245" if block_stable else "#999999",
                fontweight="bold")

    # Footer summary
    y_footer = -0.1
    ax.add_patch(Rectangle((0, 0.4), 1.0, 0.02, color="#222222", transform=ax.transData))
    ax.text(0.0, y_footer,
            f"Exact transfer: {exact_count}/{n}  ·  Block-prefix stable: {block_count}/{n}  ·  "
            f"E_A2 verdict: PARTIAL TRANSFER (≥3/9)",
            fontsize=10, va="center", ha="left", fontweight="bold")

    plt.title("Cell transfer: ANLI R1 → TriviaQA (sealed ACE winners)",
              fontsize=12, pad=12)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"fig3_transfer_matrix.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close()


if __name__ == "__main__":
    main()
