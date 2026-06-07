#!/usr/bin/env python3
"""Post-hoc decidedness-floor multiplier sensitivity sweep (read-only, CSV-only).

Sensitivity sweep across the ``CONTROL_FLOOR_MULTIPLIER`` knob (locked at 5.0
in the pre-reg) for ALL 10 step-0 panel models. Reads the locked per-model
``*_belief_readout.csv`` files; runs NO inference; the literal/control mass
fields are already persisted, so the eligibility/coverage/B-AUROC under any
multiplier is recoverable in pure Python.

DOES NOT amend, re-score, or promote any non-5.0x multiplier to a new
operating point. The pre-reg floor stays the operating point of record. This
sweep is descriptive only - it asks "at what floor multiplier does Phi-3.5
reach high coverage, and do the other 9 models stay clean at that floor?"

Output
------
* ``floor_multiplier_sweep.csv``: long-format
  (model, multiplier, n_eligible, eligible_coverage, auroc_at_full_eligible,
   ci_lo, ci_hi, verdict) row per (model x multiplier).
* ``floor_multiplier_sweep.json``: same data + structured-per-model summary
  including "first multiplier at which Phi-3.5 reaches 0.80 coverage".

Notes
-----
* Uses the SAME bootstrap-CI + signed B-AUROC convention as the locked panel
  (re-uses helpers from ``scripts.step0_belief_readout``). Pre-reg lean
  direction (-lean for label B): unchanged.
* The locked verdict logic ``classify_verdict`` is reused verbatim, threaded
  through the same coverage-curve builder.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.step0_belief_readout import (  # noqa: E402
    CONTROL_FLOOR_MULTIPLIER,
    HIGH_COVERAGE_BAR,
    LOCKED_MODEL_PANEL,
    _sanitize_for_json,
    _write_json,
    build_coverage_curve,
    classify_verdict,
    readout_csv_path_for,
    short_model_name,
)

LOCKED_MULTIPLIER = float(CONTROL_FLOOR_MULTIPLIER)
DEFAULT_MULTIPLIERS = (2.0, 3.0, 4.0, LOCKED_MULTIPLIER, 6.0, 8.0, 10.0)
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_SEED = 20260423


def _load_locked_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise SystemExit(f"locked readout csv missing: {csv_path}")
    with csv_path.open(newline="") as f:
        rdr = csv.DictReader(f)
        out = []
        for r in rdr:
            out.append({
                "sample_idx": int(r["sample_idx"]),
                "label_B": int(r["label_B"]),
                "p_yes": float(r["p_yes"]),
                "p_no": float(r["p_no"]),
                "decidedness": float(r["decidedness"]),
                "control_mass": float(r["control_mass"]),
                "lean": float(r["lean"]),
                "semantic_decidedness": float(r["semantic_decidedness"]),
                "semantic_offliteral_mass": float(r["semantic_offliteral_mass"]),
            })
    return out


def _verdict_under_multiplier(
    rows: Sequence[Dict[str, Any]],
    multiplier: float,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    labels = np.asarray([r["label_B"] for r in rows], dtype=np.int32)
    lean = np.asarray([r["lean"] for r in rows], dtype=np.float64)
    decidedness = np.asarray([r["decidedness"] for r in rows], dtype=np.float64)
    control_mass = np.asarray([r["control_mass"] for r in rows], dtype=np.float64)
    sem_dec = np.asarray([r["semantic_decidedness"] for r in rows], dtype=np.float64)
    sem_off = np.asarray([r["semantic_offliteral_mass"] for r in rows], dtype=np.float64)

    floor = multiplier * control_mass
    eligible_mask = decidedness > floor
    n = len(rows)
    eligible_coverage = float(eligible_mask.sum() / n) if n else 0.0
    # Same undetermined-coverage definition the locked panel used.
    undetermined_mask = (sem_off > decidedness) & (sem_dec > floor)
    undetermined_coverage = float(undetermined_mask.sum() / n) if n else 0.0

    # NOTE: Descriptive-sensitivity scope. The locked panel built a full
    # per-prefix coverage curve and picked the highest-cov significant point;
    # under a multiplier sweep, the per-prefix curve at every multiplier is
    # ~200 prefix x 300 bootstrap x roc_auc, prohibitive at 7 x 10 grids.
    # Here we compute the AUROC + bootstrap CI at FULL ELIGIBLE coverage under
    # the swept multiplier (one bootstrap CI per (model,mult), not 200), and
    # use a faithful proxy for "max_significant_coverage" by checking only the
    # full-eligible point. This is more conservative than the locked curve
    # logic (it can miss a high-cov significant interior point if the tail is
    # noisy), but it is correct for the sweep's purpose: "at this multiplier,
    # how strong is the model at full eligible coverage?" The locked 5.0x row
    # therefore re-derives the published eligible_coverage exactly and reports
    # the AUROC at that coverage as the headline number.
    curve: List[Dict[str, Any]] = []
    auroc_full = None
    if eligible_mask.sum() >= 2:
        eligible_idx = np.flatnonzero(eligible_mask)
        y = labels[eligible_idx].astype(int)
        s = (-lean[eligible_idx]).astype(np.float64)
        if len(np.unique(y)) >= 2:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, s))
            from scripts.step0_belief_readout import _bootstrap_ci_fixed_direction
            ci_lo, ci_hi = _bootstrap_ci_fixed_direction(
                s, y, n_resamples=n_bootstrap, seed=bootstrap_seed,
            )
            auroc_full = {
                "n_prefix": int(eligible_mask.sum()),
                "coverage": eligible_coverage,
                "auroc_b_signed": auc,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
            if ci_lo is not None and ci_lo > 0.50:
                curve.append(auroc_full)
    max_sig_cov = auroc_full["coverage"] if (
        auroc_full and auroc_full["ci_lo"] is not None and auroc_full["ci_lo"] > 0.50
    ) else 0.0
    best_point = auroc_full if max_sig_cov > 0.0 else None
    # Same verdict logic as the locked panel, using our (smaller, conservative)
    # curve as the input. Verdict-rule code path is byte-identical.
    verdict = classify_verdict(
        curve,
        eligible_coverage=eligible_coverage,
        undetermined_coverage=undetermined_coverage,
    )
    return {
        "multiplier": float(multiplier),
        "n_eligible": int(eligible_mask.sum()),
        "eligible_coverage": eligible_coverage,
        "undetermined_coverage": undetermined_coverage,
        "max_significant_coverage": max_sig_cov,
        "best_point": best_point,
        "auroc_at_full_eligible": auroc_full,
        "verdict": verdict,
    }


def _first_mult_reaching_cov(
    rows_by_mult: Sequence[Dict[str, Any]], target_cov: float
) -> Dict[str, Any]:
    # Multipliers are descending in eligibility-strictness: smaller mult -> more
    # rows clear floor. Find smallest multiplier (most permissive) at which we
    # achieve target_cov, then also the largest multiplier (strictest) that
    # still keeps cov >= target.
    # We report both for clarity, but the headline ask is "at what multiplier
    # does Phi-3.5 *reach* 0.80?" -- i.e. starting from locked 5.0 and moving
    # to smaller multipliers, when do we first cross 0.80?
    sorted_rows = sorted(rows_by_mult, key=lambda r: r["multiplier"])  # ascending
    crossing = None
    for r in sorted_rows:
        if r["eligible_coverage"] >= target_cov:
            crossing = r
            break
    return {
        "target_coverage": target_cov,
        "smallest_multiplier_reaching_target": (
            crossing["multiplier"] if crossing else None
        ),
        "crossing_row": crossing,
    }


def run_sweep(args: argparse.Namespace) -> int:
    frozen_run = Path(args.frozen_run).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir == frozen_run:
        raise SystemExit("refusing to write into the frozen locked run dir")

    multipliers = tuple(float(m) for m in args.multipliers)
    if LOCKED_MULTIPLIER not in multipliers:
        raise SystemExit(
            f"locked multiplier {LOCKED_MULTIPLIER} must be in --multipliers "
            f"(got {multipliers}) -- the sweep must include the pre-reg "
            f"operating point so readers can see it sits inside the neighborhood"
        )

    long_rows: List[Dict[str, Any]] = []
    per_model: Dict[str, Any] = {}
    for model_slug in LOCKED_MODEL_PANEL:
        csv_path = readout_csv_path_for(frozen_run, model_slug)
        rows = _load_locked_rows(csv_path)
        results = []
        for m in multipliers:
            res = _verdict_under_multiplier(
                rows,
                m,
                n_bootstrap=int(args.n_bootstrap),
                bootstrap_seed=int(args.bootstrap_seed),
            )
            results.append(res)
            af = res["auroc_at_full_eligible"]
            long_rows.append({
                "model": model_slug,
                "model_short": short_model_name(model_slug),
                "multiplier": res["multiplier"],
                "n_total": len(rows),
                "n_eligible": res["n_eligible"],
                "eligible_coverage": round(res["eligible_coverage"], 4),
                "max_significant_coverage": round(res["max_significant_coverage"], 4),
                "auroc_b_signed_at_full_eligible": (
                    round(af["auroc_b_signed"], 4) if af else None
                ),
                "ci_lo_at_full_eligible": round(af["ci_lo"], 4) if af else None,
                "ci_hi_at_full_eligible": round(af["ci_hi"], 4) if af else None,
                "verdict": res["verdict"],
            })
        crossing = _first_mult_reaching_cov(results, HIGH_COVERAGE_BAR)
        # Locked-multiplier reference snapshot.
        locked_snapshot = next(
            (r for r in results if r["multiplier"] == LOCKED_MULTIPLIER), None
        )
        per_model[model_slug] = {
            "model_short": short_model_name(model_slug),
            "n_total": len(rows),
            "results_by_multiplier": results,
            "locked_multiplier": LOCKED_MULTIPLIER,
            "locked_snapshot": locked_snapshot,
            "first_multiplier_reaching_0p80_cov": crossing,
        }

    # Headline: what mult does Phi-3.5 reach 0.80? At that mult, do the other
    # 9 stay clean (i.e. keep their locked verdict)?
    phi_slug = "mlx-community/Phi-3.5-mini-instruct-4bit"
    phi = per_model.get(phi_slug, {})
    phi_crossing = phi.get("first_multiplier_reaching_0p80_cov") or {}
    headline = {
        "phi35_smallest_multiplier_reaching_0p80_cov": (
            phi_crossing.get("smallest_multiplier_reaching_target")
        ),
        "phi35_crossing_row": phi_crossing.get("crossing_row"),
        "herd_response_at_phi35_crossing": None,
    }
    crossing_mult = phi_crossing.get("smallest_multiplier_reaching_target")
    if crossing_mult is not None:
        herd = []
        for slug, payload in per_model.items():
            if slug == phi_slug:
                continue
            at_cross = next(
                (r for r in payload["results_by_multiplier"]
                 if r["multiplier"] == crossing_mult), None
            )
            locked = payload["locked_snapshot"]
            herd.append({
                "model": slug,
                "verdict_at_crossing": at_cross["verdict"] if at_cross else None,
                "verdict_at_locked": locked["verdict"] if locked else None,
                "eligible_cov_at_crossing": (
                    at_cross["eligible_coverage"] if at_cross else None
                ),
                "eligible_cov_at_locked": (
                    locked["eligible_coverage"] if locked else None
                ),
                "verdict_changed_from_locked": bool(
                    at_cross and locked
                    and at_cross["verdict"] != locked["verdict"]
                ),
            })
        headline["herd_response_at_phi35_crossing"] = {
            "phi35_crossing_multiplier": crossing_mult,
            "n_herd_models": len(herd),
            "n_verdict_changed": sum(1 for h in herd if h["verdict_changed_from_locked"]),
            "rows": herd,
        }

    summary = {
        "schema_version": "floor_multiplier_sweep_v1",
        "analysis_kind": "post_hoc_sensitivity_csv_only",
        "locked_multiplier": LOCKED_MULTIPLIER,
        "high_coverage_bar": HIGH_COVERAGE_BAR,
        "multipliers": list(multipliers),
        "n_models": len(LOCKED_MODEL_PANEL),
        "frozen_run_dir": str(frozen_run),
        "per_model": per_model,
        "headline": headline,
        "note": (
            "Descriptive sensitivity sweep around the frozen 5.0x "
            "control_mass decidedness floor. Does NOT amend or promote any "
            "other multiplier. Pre-reg operating point stays 5.0x. Reads "
            "only the locked CSVs; no inference."
        ),
    }

    csv_path = out_dir / "floor_multiplier_sweep.csv"
    json_path = out_dir / "floor_multiplier_sweep.json"
    fields = [
        "model", "model_short", "multiplier", "n_total", "n_eligible",
        "eligible_coverage", "max_significant_coverage",
        "auroc_b_signed_at_full_eligible",
        "ci_lo_at_full_eligible", "ci_hi_at_full_eligible", "verdict",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in long_rows:
            w.writerow(r)
    _write_json(json_path, _sanitize_for_json(summary))

    print(f"[floor-sweep] wrote {csv_path}")
    print(f"[floor-sweep] wrote {json_path}")
    print(f"[floor-sweep] locked multiplier = {LOCKED_MULTIPLIER}")
    print(f"[floor-sweep] sweep multipliers = {list(multipliers)}")
    if crossing_mult is not None:
        cr = phi_crossing["crossing_row"]
        print(
            f"[floor-sweep] Phi-3.5 reaches >={HIGH_COVERAGE_BAR} cov first at "
            f"multiplier={crossing_mult} -> eligible_cov={cr['eligible_coverage']:.3f}, "
            f"verdict={cr['verdict']}"
        )
        herd = headline["herd_response_at_phi35_crossing"]
        if herd:
            print(
                f"[floor-sweep] At that multiplier the other "
                f"{herd['n_herd_models']} herd models had "
                f"{herd['n_verdict_changed']} verdict-change(s) from locked"
            )
    else:
        print(
            f"[floor-sweep] Phi-3.5 does NOT reach >={HIGH_COVERAGE_BAR} cov "
            f"at any swept multiplier"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="post-hoc decidedness-floor multiplier sweep (read-only, CSV-only)"
    )
    p.add_argument(
        "--frozen-run",
        default=str(
            REPO_ROOT / "experiments" / "t0-mech-prep" / "2026-05-17" / "run-02"
        ),
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_MULTIPLIERS),
        help=(
            "decidedness-floor multipliers to sweep (locked 5.0 MUST be "
            "included; default 2/3/4/5/6/8/10)"
        ),
    )
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_N)
    p.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.set_defaults(func=run_sweep)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
