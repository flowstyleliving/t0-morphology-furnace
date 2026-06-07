"""Plotting helpers for the PRI experiment pipeline.

Separated from `pri_runtime` so hot-path subprocesses do not import
Matplotlib/Seaborn unless they actually render figures.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from pri_runtime import Config, hedges_g, print_header, safe_auroc


def make_figures(results: pd.DataFrame, config: Config) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)

    print_header("GENERATING FIGURES")
    if results.empty:
        print("  No results; skipping figures.")
        return

    save_dir = config.save_dir
    models = sorted(results["model"].unique())
    pri_cols = sorted([c for c in results.columns if c.startswith("pri_")])
    n_models = len(models)

    # Figure 1: AUROC bar chart
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
        ]
        labels = s1["contradiction"].astype(int).values
        auc_pairs = [(col, safe_auroc(labels, s1[col].values)) for col in pri_cols]
        auc_pairs = [x for x in auc_pairs if not np.isnan(x[1])]
        auc_pairs.sort(key=lambda x: x[1])

        names = [x[0].replace("pri_", "") for x in auc_pairs]
        vals = [x[1] for x in auc_pairs]
        colors = ["#1E88E5" if "v1" in x[0] else "#E53935" for x in auc_pairs]

        ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0.45, 1.01)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("AUROC")
        ax.set_title(model_name.split("/")[-1], fontweight="bold")

    fig.suptitle("PRI v1 vs v2 AUROC (Step 1, Final Layer)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_v1_vs_v2_auroc.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "fig1_v1_vs_v2_auroc.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Trajectory
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8), sharey="row")
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for ci, model_name in enumerate(models):
        md = results[
            (results["model"] == model_name)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
            & (results["gen_step"] <= 5)
        ]
        for ri, variant in enumerate(["pri_v1_cosine", "pri_v2_full"]):
            ax = axes[ri, ci]
            for cond, color, label in [
                (False, "#1E88E5", "Control"),
                (True, "#E53935", "Contradiction"),
            ]:
                sub = md[md["contradiction"] == cond]
                if sub.empty:
                    continue
                means = sub.groupby("gen_step")[variant].mean()
                sems = sub.groupby("gen_step")[variant].sem()
                ax.errorbar(
                    means.index,
                    means.values,
                    yerr=sems.values,
                    color=color,
                    marker="o",
                    capsize=3,
                    label=label,
                )
            ax.set_xlabel("Generation Step")
            ax.set_ylabel(variant.replace("pri_", ""))
            if ri == 0:
                ax.set_title(model_name.split("/")[-1], fontweight="bold")
            ax.legend(fontsize=8)

    fig.suptitle("PRI Trajectory: v1 (top) vs v2_full (bottom)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig2_step_trajectory.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Violin step-1
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8))
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for ci, model_name in enumerate(models):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
        ]
        for ri, variant in enumerate(["pri_v1_cosine", "pri_v2_full"]):
            ax = axes[ri, ci]
            pdv = s1[["contradiction", variant]].copy()
            pdv["condition"] = pdv["contradiction"].map(
                {False: "Control", True: "Contradiction"}
            )
            sns.violinplot(
                data=pdv,
                x="condition",
                y=variant,
                palette={"Control": "#1E88E5", "Contradiction": "#E53935"},
                cut=0,
                ax=ax,
            )
            g, _ = hedges_g(
                s1[s1["contradiction"]][variant].values,
                s1[~s1["contradiction"]][variant].values,
            )
            ax.set_title(f"{model_name.split('/')[-1]} | g={g:.2f}", fontsize=10)
            ax.set_ylabel(variant.replace("pri_", ""))

    fig.suptitle("Step-1 Distribution: v1 (top) vs v2_full (bottom)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_violins.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: Outcome independence
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
        ]
        v = "pri_v2_full"
        groups = {
            "Control": s1[~s1["contradiction"]][v].values,
            "Contr-Correct": s1[s1["contradiction"] & s1["is_correct"]][v].values,
            "Contr-Incorrect": s1[s1["contradiction"] & ~s1["is_correct"]][v].values,
        }
        means = [np.nanmean(vv) if len(vv) else np.nan for vv in groups.values()]
        sems = [
            np.nanstd(vv) / np.sqrt(len(vv)) if len(vv) > 1 else 0.0
            for vv in groups.values()
        ]

        ax.bar(
            range(3),
            means,
            yerr=sems,
            capsize=5,
            color=["#1E88E5", "#43A047", "#E53935"],
            alpha=0.85,
        )
        ax.set_xticks(range(3))
        ax.set_xticklabels(list(groups.keys()), fontsize=8)
        ax.set_ylabel(v.replace("pri_", ""))
        ax.set_title(model_name.split("/")[-1], fontweight="bold")

    fig.suptitle("Outcome Independence (Step 1, v2_full)", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig4_outcome_independence.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: Alpha sweep
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    variants = ["pri_v1_cosine", "pri_v2_diag", "pri_v2_full", "pri_v2_topk64"]
    styles = ["--", "-", "-", "-."]

    for ax, model_name in zip(axes, models):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
        ]
        for variant, style in zip(variants, styles):
            if variant not in s1.columns:
                continue
            aucs: List[float] = []
            for a in config.alpha_values:
                sub = s1[s1["alpha"] == a]
                aucs.append(
                    safe_auroc(sub["contradiction"].astype(int).values, sub[variant].values)
                )
            ax.plot(
                config.alpha_values,
                aucs,
                style,
                marker="o",
                label=variant.replace("pri_", ""),
                markersize=4,
            )

        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.set_ylabel("AUROC")
        ax.set_title(model_name.split("/")[-1], fontweight="bold")
        ax.legend(fontsize=7)

    fig.suptitle("Alpha Sweep", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_alpha_sweep.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures written to: {save_dir}")
