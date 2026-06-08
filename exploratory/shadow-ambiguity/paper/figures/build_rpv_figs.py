#!/usr/bin/env python3
"""Build RPV (Readout Pseudo-Volume) workshop figures from comprehensive_outputs/.

Constrained, high-signal set sized for an 8pp workshop -- three figures, one claim
each, plus a compact verdict table:

  fig1_rpv_vs_confidence    forest: RPV increment over {surprise} per (model x benchmark)
                            with the random-effects meta diamond.  -> "beats confidence"
  fig2_redundancy_ladder    the three meta diamonds as v3's null_ratio enters the base
                            (+0.102 -> +0.027 -> +0.011).          -> "redundant with v3 (H1 NO-GO)"
  fig3_collapse_complement  H2: RPV increment over {surprise,null_ratio} vs v3 weakness,
                            weighted fit.                          -> "earns its keep where v3 dies"
  table1_summary.tex        compact meta-level verdict table.

Reads JSON only -- no model re-traces. Run via build_all.sh (uses the t0 .venv).
The number plumbing is faithful to comprehensive_run.py's meta aggregation; fig3
re-derives the weighted slope and prints it next to the stored value as a self-check.
"""
import json
import os
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "out")
DATADIR = os.path.normpath(os.path.join(HERE, "..", "..", "comprehensive_outputs"))
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
})

FAMILY_COLORS = {
    "llama": "#1f77b4",
    "mistral": "#d62728",
    "qwen": "#2ca02c",
    "phi": "#9467bd",
    "gemma": "#ff7f0e",
    "deepseek": "#17becf",
}
POS = "#2ca02c"   # clears the bar
NEG = "#8c8c8c"   # below the bar


def fam_color(f):
    return FAMILY_COLORS.get(str(f).lower(), "#7f7f7f")


BENCH_SHORT = {"anli_r1": "ANLI", "triviaqa_paired": "TriviaQA"}


def short_model(m):
    s = m.split("/")[-1]
    for suf in ("-4bit", "-MXFP4-Q4", "-Instruct", "-instruct", "-it"):
        s = s.replace(suf, "")
    return s


def pair_label(rec):
    return f"{short_model(rec['model'])} · {BENCH_SHORT.get(rec['benchmark'], rec['benchmark'])}"


def save(fig, name, aliases=()):
    names = (name, *aliases)
    for ext in ("pdf", "png"):
        for out_name in names:
            fig.savefig(os.path.join(OUTDIR, f"{out_name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    alias_note = "" if not aliases else f" (+ aliases: {', '.join(aliases)})"
    print(f"[ok] {name}.pdf / .png{alias_note}", file=sys.stderr)


def load():
    meta = json.load(open(os.path.join(DATADIR, "shadow_v2_meta.json")))
    pairs = {}
    for p in glob.glob(os.path.join(DATADIR, "*.json")):
        if "meta" in os.path.basename(p):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if not j.get("analysis"):
            continue
        pairs[(j["model"], j["benchmark"])] = j
    return meta, pairs


def fig1(meta):
    rows = [r for r in meta["per_pair"] if r.get("primary_diff") is not None]
    rows.sort(key=lambda r: r["primary_diff"])
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    for i, r in enumerate(rows):
        d = r["primary_diff"]
        lo, hi = r.get("primary_ci", [None, None])
        c = fam_color(r["family"])
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [i, i], color=c, lw=1.6, alpha=0.85, zorder=2)
        ax.plot(d, i, "o", color=c, ms=5, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([pair_label(r) for r in rows], fontsize=7)
    ax.axvline(0.0, color="0.4", lw=1.0, zorder=1)
    ax.axvline(0.02, color="0.4", lw=1.0, ls="--", zorder=1)

    m = meta["primary_random_effects"]
    mm, mlo, mhi = m["mean"], m["ci_lo"], m["ci_hi"]
    ax.plot([mlo, mhi], [-1.8, -1.8], color="k", lw=2.6, zorder=4)
    ax.plot(mm, -1.8, "D", color="k", ms=10, zorder=5)
    ax.text(mm, -3.0,
            f"random-effects meta  {mm:+.3f}  [{mlo:+.3f}, {mhi:+.3f}]   p≈{m['p_one_sided_le_zero']:.0e}",
            ha="center", fontsize=8.5, fontweight="bold")
    ax.text(0.021, len(rows) - 0.6, "+0.02 min effect", fontsize=7.5, color="0.35", va="top")

    ax.set_ylim(-3.6, len(rows) - 0.3)
    ax.set_xlabel("Incremental AUROC of RPV over plain confidence {surprise}")
    ax.set_title("RPV beats plain confidence across 26 (model × benchmark) pairs")
    fams = sorted({str(r["family"]).lower() for r in rows})
    ax.legend(handles=[Patch(color=fam_color(f), label=f) for f in fams],
              loc="lower right", fontsize=8, title="family")
    fig.tight_layout()
    save(fig, "fig1_rpv_vs_confidence", aliases=("fig1_forest_rpv_vs_confidence",))


def fig2(meta):
    bases = [
        ("over {surprise}", meta["primary_random_effects"]),
        ("over {surprise, null_ratio}", meta["primary_base_null_ratio_random_effects"]),
        ("over {surprise, null_ratio, p_max}", meta["primary_base_b_random_effects"]),
    ]
    ypos = [2, 1, 0]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for (label, m), yp in zip(bases, ypos):
        mm, lo, hi = m["mean"], m["ci_lo"], m["ci_hi"]
        c = POS if mm >= 0.02 else NEG
        ax.plot([lo, hi], [yp, yp], color=c, lw=2.6, zorder=3)
        ax.plot(mm, yp, "D", color=c, ms=12, zorder=4)
        kh = m.get("modified_knapp_hartung_t") or {}
        if kh.get("available"):
            ax.plot([kh["ci_lo"], kh["ci_hi"]], [yp - 0.17, yp - 0.17],
                    color=c, lw=1.1, alpha=0.55, zorder=2)
        ax.text(hi + 0.004, yp, f"{mm:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
                va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels([b[0] for b in bases], fontsize=9)
    ax.axvline(0.0, color="0.4", lw=1.0)
    ax.axvline(0.02, color="0.4", lw=1.3, ls="--")
    ax.text(0.0205, 2.55, "+0.02 min effect", fontsize=8, color="0.35")
    ax.set_xlim(-0.01, 0.165)
    ax.set_ylim(-0.7, 2.9)
    ax.set_xlabel("RPV incremental AUROC  (random-effects meta, k=26; thin bar = small-k Knapp–Hartung)")
    ax.set_title("v3's null_ratio absorbs RPV:  +0.102 → +0.011  (below the bar → H1 NO-GO)")
    fig.tight_layout()
    save(fig, "fig2_redundancy_ladder")


def fig3(meta, pairs):
    xs, ys, ws, cs, labs = [], [], [], [], []
    for (model, bench), j in pairs.items():
        a = j["analysis"]
        ep = (a.get("incremental_logistic_repeated_cv", {})
              .get("fisher_eff_rank", {})
              .get("over_surprise_null_ratio", {}))
        diff = ep.get("diff")
        marg = (a.get("marginal_train_locked_auroc", {})
                .get("null_ratio_post_rank1", {})
                .get("auroc"))
        lo, hi = ep.get("ci_lo"), ep.get("ci_hi")
        if diff is None or marg is None or lo is None or hi is None:
            continue
        se = (hi - lo) / (2 * 1.959964)
        if se <= 0:
            continue
        xs.append(1.0 - float(marg))
        ys.append(float(diff))
        ws.append(1.0 / (se * se))
        cs.append(fam_color(j.get("model_family")))
        labs.append((model, bench))
    xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)

    X = np.column_stack([np.ones(len(xs)), xs])
    W = np.diag(ws)
    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ ys)
    intc, slope = float(beta[0]), float(beta[1])
    stored = meta["h2_regime_interaction"].get("weighted_slope")
    print(f"[fig3] reconstructed weighted slope = {slope:+.5f}  (stored {stored:+.5f})", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sizes = 30 + 340 * (ws / ws.max())
    ax.scatter(xs, ys, s=sizes, c=cs, alpha=0.82, edgecolor="white", lw=0.7, zorder=3)
    xx = np.linspace(float(xs.min()), float(xs.max()), 50)
    ax.plot(xx, intc + slope * xx, color="k", lw=2.0, zorder=2,
            label=f"weighted fit  (slope {slope:+.3f})")
    ax.axhline(0.0, color="0.4", lw=1.0, zorder=1)
    for (model, bench), x, yv in zip(labs, xs, ys):
        if "Qwen3-8B" in model:
            ax.annotate(f"Qwen3-8B·{BENCH_SHORT.get(bench, bench)}", (x, yv),
                        textcoords="offset points", xytext=(7, 4), fontsize=7.5)
    ax.set_xlabel("v3 weakness  =  1 − marginal AUROC(null_ratio)      (→ v3 weaker)")
    ax.set_ylabel("RPV increment over {surprise, null_ratio}")
    ax.set_title("RPV earns its keep where v3 collapses  (H2 slope +0.080)")
    fams = sorted({k for k in FAMILY_COLORS if any(fam_color(j.get("model_family")) == FAMILY_COLORS[k]
                   for j in pairs.values())})
    handles = [Patch(color=FAMILY_COLORS[f], label=f) for f in fams]
    handles.append(plt.Line2D([0], [0], color="k", lw=2.0, label=f"weighted fit (slope {slope:+.3f})"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig3_collapse_complement")


def table1(meta):
    a = meta["primary_random_effects"]
    nr = meta["primary_base_null_ratio_random_effects"]
    b = meta["primary_base_b_random_effects"]
    fam = meta["family_spanning_verdict"]
    h2 = meta["h2_regime_interaction"]
    gate = meta["h1_gate_summary"]

    def row(name, m):
        return (f"{name} & {m['mean']:+.3f} & [{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}] "
                f"& {m['p_one_sided_le_zero']:.1e} \\\\")

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"RPV incremental over base & mean $\Delta$AUROC & 95\% CI (DL) & $p$ \\",
        r"\midrule",
        row(r"\{surprise\} \;(base A)", a),
        row(r"\{surprise, null\_ratio\}", nr),
        row(r"\{surprise, null\_ratio, p\_max\} \;(base B)", b),
        r"\midrule",
        rf"\multicolumn{{4}}{{l}}{{$k={a['k']}$ pairs $\cdot$ min practical effect $=0.02$ $\cdot$ "
        rf"brittleness gate: {'PASS' if gate['passes_brittleness_gate'] else 'FAIL'}}} \\",
        rf"\multicolumn{{4}}{{l}}{{H1 registered gates: \textbf{{{'PASS' if gate['passes_registered_h1_gates'] else 'NO-GO'}}} "
        rf"(base B mean $<0.02$) $\cdot$ family-spanning: {fam['verdict'].replace('_', ' ')}}} \\",
        rf"\multicolumn{{4}}{{l}}{{H2 regime slope $={h2['weighted_slope']:+.3f}$ "
        rf"(RPV adds where null\_ratio is weak)}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(os.path.join(OUTDIR, "table1_summary.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[ok] table1_summary.tex", file=sys.stderr)


if __name__ == "__main__":
    meta, pairs = load()
    print(f"[load] meta k={meta['primary_random_effects']['k']} | per-pair JSONs={len(pairs)}", file=sys.stderr)
    fig1(meta)
    fig2(meta)
    fig3(meta, pairs)
    table1(meta)
    print(f"[done] -> {OUTDIR}", file=sys.stderr)
