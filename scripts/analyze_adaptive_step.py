#!/usr/bin/env python3
"""Adaptive-step rupture analyzer — v3.2 amendment + t0-candidate #3.

Reads `all_results.parquet` (post-2026-05-08 patch — must contain `gen_token_id`
per row). For each (model, sample) finds the answer-commit step by decoding the
gen_token_id sequence and matching standalone YES/NO answer tokens. Then:

  * Computes AUROC for every emitted null_ratio metric at three planes:
      sealed       — gen_step == 1 (the pinned v3.x analysis plane)
      adaptive     — gen_step == commit_step (the answer-emission step)
      adaptive_pre — gen_step == commit_step − 1 (the proximate-cause step)
  * Reports per-model deltas and identifies which metric/plane combination
    dominates per model.
  * Flags samples that never reach a YES/NO in `max_gen_tokens` (COT overflow);
    by default they are excluded from adaptive AUROC.

Usage:
    .venv/bin/python scripts/analyze_adaptive_step.py \\
        --run-dir experiments/v3-main-run/2026-05-08/run-NN

Cross-references:
  - wiki/t0-candidates.md §3 (Adaptive-step rupture detection) — pre-reg
  - wiki/results/v3.2-amendment.md — pre-reg context
  - pipeline patch site: pri_v2_mlx_pipeline.py:row_out (gen_token_id capture)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Commit-step detection
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_PATTERNS = (
    " YES", ":YES", "\nYES", "YES\n", "YES.", "YES,",
    " NO", ":NO", "\nNO", "NO\n", "NO.", "NO,",
)


def find_commit_step(tokenizer, gen_token_ids: List[int]) -> Optional[int]:
    """Return 1-indexed gen_step where the cumulative decoded text first
    contains a standalone YES/NO answer token, or None if never found.

    Heuristic: decode tokens one at a time, accumulate prefix string, scan
    for ANSWER_PATTERNS. BPE may split "YES" across tokens or pad with a
    leading space, so substring match on the cumulative prefix is more
    robust than per-token match.
    """
    prefix = ""
    for i, tid in enumerate(gen_token_ids):
        prefix += tokenizer.decode([int(tid)])
        upper = prefix.upper()
        for ans in ANSWER_PATTERNS:
            if ans in upper:
                return i + 1
        # Edge: very first token IS the answer alone (no prefix space).
        if i == 0 and upper.strip() in ("YES", "NO"):
            return 1
    return None


# ─────────────────────────────────────────────────────────────────────────────
# AUROC helper
# ─────────────────────────────────────────────────────────────────────────────


def auroc_signed(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, int]:
    """Direction-agnostic AUROC: max(auc, 1-auc). Returns (auc, sign in {-1, +1}).
    NaN scores are dropped. NaN auc if fewer than 4 finite samples or fewer than
    2 distinct labels."""
    finite = np.isfinite(scores)
    if finite.sum() < 4 or len(np.unique(labels[finite])) < 2:
        return float("nan"), 0
    auc = roc_auc_score(labels[finite], scores[finite])
    return float(max(auc, 1 - auc)), 1 if auc >= 0.5 else -1


# ─────────────────────────────────────────────────────────────────────────────
# Metric column families
# ─────────────────────────────────────────────────────────────────────────────


def detect_rank_columns(df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    """Group emitted metric columns into Fisher / Raw / Centered families
    plus single-scalar metrics (kl_discharged, d_F_full, …)."""
    families: Dict[str, List[Tuple[str, str]]] = {
        "Fisher":   [],
        "Raw":      [],
        "Centered": [],
        "scalar":   [],
    }
    for c in df.columns:
        if c.startswith("null_ratio_centered_post_rank"):
            r = c.replace("null_ratio_centered_post_rank", "")
            families["Centered"].append((f"r={r}", c))
        elif c.startswith("null_ratio_raw_post_rank"):
            r = c.replace("null_ratio_raw_post_rank", "")
            families["Raw"].append((f"r={r}", c))
        elif c.startswith("null_ratio_post_rank"):
            r = c.replace("null_ratio_post_rank", "")
            families["Fisher"].append((f"r={r}", c))
    if "kl_discharged" in df.columns:
        families["scalar"].append(("kl_discharged", "kl_discharged"))
    if "d_F_full" in df.columns:
        families["scalar"].append(("d_F_full", "d_F_full"))

    # Sort by integer rank within each null_ratio family
    for fam in ("Fisher", "Raw", "Centered"):
        families[fam].sort(key=lambda kv: int(kv[0].split("=")[1]))
    return families


# ─────────────────────────────────────────────────────────────────────────────
# Plane evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_plane(
    df_filt: pd.DataFrame,
    families: Dict[str, List[Tuple[str, str]]],
    plane_label: str,
) -> Dict[Tuple[str, str], Tuple[float, int]]:
    """Compute AUROC per (family, rank_label) on a filtered DataFrame
    (one row per sample). Labels = `contradiction` column, scores = metric col."""
    out: Dict[Tuple[str, str], Tuple[float, int]] = {}
    if df_filt.empty:
        return out
    labels = df_filt["contradiction"].astype(int).to_numpy()
    for fam, cols in families.items():
        for rank_label, col in cols:
            if col not in df_filt.columns:
                continue
            scores = df_filt[col].to_numpy()
            auc, sign = auroc_signed(labels, scores)
            out[(fam, rank_label)] = (auc, sign)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline per run directory
# ─────────────────────────────────────────────────────────────────────────────


def analyze_run(run_dir: Path, fallback: str = "exclude") -> Dict:
    """Run the adaptive-step analysis on one run directory.

    fallback: how to handle samples with no detected commit_step. One of:
        "exclude"  — drop those samples from adaptive AUROC (default)
        "step1"    — fall back to gen_step=1 for those samples
    """
    parquet = run_dir / "all_results.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"missing {parquet}")
    df = pd.read_parquet(parquet)

    if "gen_token_id" not in df.columns:
        raise RuntimeError(
            f"{parquet} has no `gen_token_id` column — this analyzer needs the "
            "post-2026-05-08 pipeline patch. Re-run with the patched pipeline "
            "or use the legacy trace_dumps-only path."
        )

    sealed = df[(df["layer"] == "final") & (df["alpha"] == 1.0)].copy()
    families = detect_rank_columns(sealed)
    models = sealed["model"].unique().tolist()
    print(f"models in run: {len(models)}")
    for m in models:
        n = (sealed["model"] == m).sum()
        print(f"  {m.split('/')[-1]:<35s}  rows={n}")

    # Lazy import — only needed if we'll actually decode tokens
    from transformers import AutoTokenizer

    print("\nLoading tokenizers…")
    tokenizers = {}
    for m in models:
        try:
            tokenizers[m] = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        except Exception as exc:
            print(f"  {m.split('/')[-1]}: tokenizer load FAILED ({exc}); skipping")

    # For each (model, sample_id), determine commit_step from the per-step
    # gen_token_id sequence at layer=='final', alpha==1.0 (one row per step).
    print("\nDetecting commit_step per sample…")
    commit_lookup: Dict[Tuple[str, int], Optional[int]] = {}
    for m in models:
        if m not in tokenizers:
            continue
        tok = tokenizers[m]
        gm = sealed[sealed["model"] == m]
        for sid, gs in gm.groupby("sample_id"):
            gs_sorted = gs.sort_values("gen_step")
            ids = gs_sorted["gen_token_id"].astype(int).tolist()
            cs = find_commit_step(tok, ids)
            commit_lookup[(m, int(sid))] = cs

    # Print commit_step distribution per model
    print("\nCommit-step distribution per model:")
    for m in models:
        cs_list = [v for (mm, _), v in commit_lookup.items() if mm == m and v is not None]
        n_total = sum(1 for (mm, _) in commit_lookup.keys() if mm == m)
        n_found = len(cs_list)
        miss = n_total - n_found
        if cs_list:
            from collections import Counter
            ctr = Counter(cs_list)
            dist = ", ".join(f"step{s}×{ctr[s]}" for s in sorted(ctr))
        else:
            dist = "(none)"
        print(f"  {m.split('/')[-1]:<35s}  found={n_found}/{n_total:<4d}  miss={miss:<3d}  {dist}")

    # Build the three planes per model
    print("\n" + "=" * 110)
    print(f"AUROC at three planes per model (fallback for missing commit_step: {fallback!r})")
    print("=" * 110)

    summary: Dict[str, Dict[str, Dict]] = {}

    for m in models:
        short = m.split("/")[-1]
        gm = sealed[sealed["model"] == m]
        # Plane: sealed step=1
        sealed1 = gm[gm["gen_step"] == 1].copy()
        # Plane: adaptive (one row per sample at commit_step)
        adaptive_rows = []
        adaptive_pre_rows = []
        for sid in gm["sample_id"].unique():
            cs = commit_lookup.get((m, int(sid)))
            if cs is None:
                if fallback == "step1":
                    cs = 1
                else:
                    continue
            r = gm[(gm["sample_id"] == sid) & (gm["gen_step"] == cs)]
            if len(r) == 1:
                adaptive_rows.append(r.iloc[0])
            if cs and cs > 1:
                rp = gm[(gm["sample_id"] == sid) & (gm["gen_step"] == cs - 1)]
                if len(rp) == 1:
                    adaptive_pre_rows.append(rp.iloc[0])
        adaptive = pd.DataFrame(adaptive_rows) if adaptive_rows else pd.DataFrame()
        adaptive_pre = pd.DataFrame(adaptive_pre_rows) if adaptive_pre_rows else pd.DataFrame()

        ev_sealed = evaluate_plane(sealed1, families, "sealed")
        ev_adapt = evaluate_plane(adaptive, families, "adaptive")
        ev_pre = evaluate_plane(adaptive_pre, families, "adaptive_pre")

        print(f"\n  [{short}]  n_sealed={len(sealed1)}  n_adaptive={len(adaptive)}  n_adaptive_pre={len(adaptive_pre)}")
        print(f"    {'metric':<22s}  {'sealed':>10s}  {'adaptive':>10s}  {'Δ_adapt':>10s}  {'pre':>10s}  {'Δ_pre':>10s}")

        # Print every (family, rank) row, sorted: Fisher → Raw → Centered → scalar
        rows_for_summary: Dict[str, Dict] = {}
        for fam in ("Fisher", "Raw", "Centered", "scalar"):
            for rank_label, col in families[fam]:
                key = (fam, rank_label)
                s, _ = ev_sealed.get(key, (float("nan"), 0))
                a, _ = ev_adapt.get(key, (float("nan"), 0))
                p, _ = ev_pre.get(key, (float("nan"), 0))
                d_a = a - s if not (np.isnan(a) or np.isnan(s)) else float("nan")
                d_p = p - s if not (np.isnan(p) or np.isnan(s)) else float("nan")
                metric_name = f"{fam} {rank_label}" if fam != "scalar" else rank_label
                fmt = lambda x: f"{x:>10.4f}" if not np.isnan(x) else f"{'n/a':>10s}"
                fmt_d = lambda x: f"{x:>+10.4f}" if not np.isnan(x) else f"{'n/a':>10s}"
                print(f"    {metric_name:<22s}  {fmt(s)}  {fmt(a)}  {fmt_d(d_a)}  {fmt(p)}  {fmt_d(d_p)}")
                rows_for_summary[metric_name] = {
                    "sealed": s, "adaptive": a, "adaptive_pre": p,
                    "delta_adaptive": d_a, "delta_adaptive_pre": d_p,
                }
        summary[short] = rows_for_summary

    return {
        "run_dir": str(run_dir),
        "n_models": len(models),
        "fallback": fallback,
        "commit_step_summary": {
            m.split("/")[-1]: {
                "found": sum(1 for (mm, _), v in commit_lookup.items() if mm == m and v is not None),
                "total": sum(1 for (mm, _) in commit_lookup.keys() if mm == m),
            }
            for m in models
        },
        "auroc": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="experiments/v3-main-run/<DATE>/run-NN/")
    ap.add_argument("--fallback", choices=("exclude", "step1"), default="exclude",
                    help="how to handle samples with no detected commit_step")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="optional path to dump the per-model AUROC summary as JSON")
    args = ap.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: run dir not found: {args.run_dir}")
        return 1

    summary = analyze_run(args.run_dir, fallback=args.fallback)

    if args.out_json:
        args.out_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        print(f"\nSummary JSON: {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
