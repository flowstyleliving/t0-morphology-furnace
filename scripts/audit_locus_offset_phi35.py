#!/usr/bin/env python3
"""Post-hoc t=1 locus-offset sensitivity audit for Phi-3.5-mini (read-only).

This is a RETROSPECTIVE sensitivity analysis on the locked step-0 belief panel
([[step0-belief-readout-2026-05-17]]). It does NOT amend, re-score, or
re-interpret the preregistered result. It answers ONE narrow question raised by
the prior 2026-05-18 audit: Phi-3.5-mini's dominant non-literal top-1 at t=0
is the newline token "\n" (38/200 rows, fraction 0.195). Does Phi-3.5 clear the
frozen decidedness floor at t=1 (one position past the formatting prefix) on
the SAME 200 frozen samples?

Contract / guarantees
---------------------
* Inputs are the SAME frozen ``n=200`` slice, the SAME per-model
  ``Phi-3.5-mini-instruct-4bit_belief_spec.json``, and the SAME chat-template
  strategy used by the locked panel. The frozen literal/semantic/control
  buckets are reused verbatim - never re-fit.
* Locked artifacts in the frozen run dir are opened READ-ONLY. All audit
  output goes to a separate audit dir.
* Integrity gate: before reading t=1, recompute the t=0 ``p_yes / p_no / lean``
  for all 200 rows AND the frozen canary top-10, abort on any drift beyond
  fp tolerance. Mirrors ``audit_t0_recoverability.py`` posture exactly.
* t=1 protocol: for each sample, take the model's own greedy top-1 at t=0
  (i.e. the token Phi-3.5 would emit next), append it to the prompt token
  sequence, and read the next-token distribution at the new last position.
  This is the most faithful continuation of the locked measurement -- it asks
  "what does Phi-3.5 believe one token after its own formatting prefix?"
  Choice documented here + in the wiki page.
* Scoring math (``_score_row_from_probs``) imported verbatim from
  ``scripts.step0_belief_readout`` -- the literal / semantic / control
  partition and the 5.0x control_mass decidedness floor are identical to the
  locked run.

Output
------
* Per-row CSV with t=0 + t=1 metrics side by side, plus the conditional
  "t=0 top-1 was newline" flag for the subset analysis.
* JSON summary including: t=1 eligible_coverage, signed B-AUROC + bootstrap
  CI on the t=1 eligible subset, verdict label under the SAME pre-reg rule,
  and the conditional-on-newline subset metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_v2_io_plugins as io_plugins
import pri_v2_mlx_pipeline as pipeline
from pri_calibrator import _load_calibration_jsonl
from scripts.step0_belief_readout import (
    CONTROL_FLOOR_MULTIPLIER,
    EPS,
    HIGH_COVERAGE_BAR,
    _bootstrap_ci_fixed_direction,
    _gold_yes_no,
    _load_json,
    _normalize_literal_token,
    _sanitize_for_json,
    _score_row_from_probs,
    _write_json,
    canary_path_for,
    classify_verdict,
    readout_csv_path_for,
    short_model_name,
    spec_path_for,
)
from sklearn.metrics import roc_auc_score

# Integrity-gate tolerances (per the 2026-05-18 audit's posture: tight, abort
# on breach). Same model + same prompt + same softmax => agreement should be
# fp32-noise scale.
PROB_ABS_TOL = 1e-6
LEAN_ABS_TOL = 1e-5
CANARY_PROB_ABS_TOL = 1e-6


def _topk_records(tokenizer: Any, probs: np.ndarray, *, k: int) -> List[Dict[str, Any]]:
    top_ids = np.argsort(-probs)[:k]
    out: List[Dict[str, Any]] = []
    for rank, token_id in enumerate(top_ids, start=1):
        decoded = pipeline.decode_ids(tokenizer, [int(token_id)])
        out.append({
            "rank": rank,
            "token_id": int(token_id),
            "decoded": decoded,
            "normalized": _normalize_literal_token(decoded),
            "prob": float(probs[int(token_id)]),
        })
    return out


def _forward_last_probs(model: Any, token_ids: Sequence[int]) -> np.ndarray:
    """Run a forward pass on a token-id sequence; return softmax of last position.

    Mirrors ``prefix_readout``'s last-position softmax but accepts a raw
    token-id list (so we can append the t=0 greedy choice without re-encoding).
    """
    input_ids = mx.array([list(token_ids)], dtype=mx.int32)
    prefix_logits = model(input_ids)
    last_logits = pipeline.to_numpy(prefix_logits[0, -1].astype(mx.float32))
    return pipeline.safe_softmax(last_logits)


def _gate_against_locked_t0(
    *,
    model_slug: str,
    t0_rows: Sequence[Dict[str, Any]],
    locked_csv: Path,
    canary_path: Path,
    t0_full_topk: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Abort unless the recomputed t=0 forward pass reproduces locked numbers."""
    with locked_csv.open(newline="") as f:
        locked = {int(r["sample_idx"]): r for r in csv.DictReader(f)}
    if len(locked) != len(t0_rows):
        raise RuntimeError(
            f"[{model_slug}] locked CSV has {len(locked)} rows, audit has "
            f"{len(t0_rows)} -- slice mismatch, refusing to proceed"
        )
    max_dp = 0.0
    max_dl = 0.0
    for row in t0_rows:
        lr = locked[row["sample_idx"]]
        dpy = abs(row["t0_p_yes"] - float(lr["p_yes"]))
        dpn = abs(row["t0_p_no"] - float(lr["p_no"]))
        dln = abs(row["t0_lean"] - float(lr["lean"]))
        max_dp = max(max_dp, dpy, dpn)
        max_dl = max(max_dl, dln)
        if dpy > PROB_ABS_TOL or dpn > PROB_ABS_TOL or dln > LEAN_ABS_TOL:
            raise RuntimeError(
                f"[{model_slug}] integrity gate FAILED at sample "
                f"{row['sample_idx']}: |dp_yes|={dpy:.2e} |dp_no|={dpn:.2e} "
                f"|dlean|={dln:.2e} exceeds tol -- forward pass does not "
                f"reproduce the locked t=0 run; audit aborted"
            )

    canary = _load_json(canary_path)
    max_dc = 0.0
    n_canary = 0
    for s in canary.get("samples", []):
        idx = int(s["sample_idx"])
        recomputed = t0_full_topk.get(idx)
        if recomputed is None:
            raise RuntimeError(
                f"[{model_slug}] canary sample {idx} not in recomputed top-k"
            )
        frozen_top10 = s["top10"]
        recomputed_top10 = recomputed[:10]
        for fr, rc in zip(frozen_top10, recomputed_top10):
            if int(fr["token_id"]) != int(rc["token_id"]):
                raise RuntimeError(
                    f"[{model_slug}] canary sample {idx} rank {fr['rank']} "
                    f"token id drift: frozen={fr['token_id']} "
                    f"recomputed={rc['token_id']} -- audit aborted"
                )
            d = abs(float(fr["prob"]) - float(rc["prob"]))
            max_dc = max(max_dc, d)
            if d > CANARY_PROB_ABS_TOL:
                raise RuntimeError(
                    f"[{model_slug}] canary sample {idx} rank {fr['rank']} "
                    f"prob drift {d:.2e} > {CANARY_PROB_ABS_TOL:.0e} -- aborted"
                )
        n_canary += 1
    if n_canary == 0:
        raise RuntimeError(
            f"[{model_slug}] integrity gate FAILED: frozen canary "
            f"{canary_path} has no samples to check -- refusing to report "
            f"the gate as passed on a corrupt/truncated canary"
        )
    return {
        "passed": True,
        "n_rows_checked": len(t0_rows),
        "n_canary_samples_checked": n_canary,
        "max_abs_prob_drift_vs_locked_csv": max_dp,
        "max_abs_lean_drift_vs_locked_csv": max_dl,
        "max_abs_prob_drift_vs_frozen_canary": max_dc,
        "prob_abs_tol": PROB_ABS_TOL,
        "lean_abs_tol": LEAN_ABS_TOL,
        "canary_prob_abs_tol": CANARY_PROB_ABS_TOL,
    }


def _coverage_curve_and_max(
    *,
    labels: np.ndarray,
    lean: np.ndarray,
    decidedness: np.ndarray,
    eligible_mask: np.ndarray,
    total: int,
    n_bootstrap: int = 1000,
    seed: int = 20260423,
) -> Dict[str, Any]:
    """Build the same prefix-by-decidedness coverage curve the locked run used.

    Returns curve points + (highest-coverage point with CI_lo>0.50) + an
    "AUROC at full eligible coverage" convenience number for the audit page.
    """
    if total == 0 or eligible_mask.sum() == 0:
        return {
            "curve": [],
            "max_significant_coverage": 0.0,
            "best_point": None,
            "auroc_at_full_eligible": None,
            "n_eligible": int(eligible_mask.sum()),
            "eligible_coverage": 0.0,
        }
    eligible_idx = np.flatnonzero(eligible_mask)
    order = eligible_idx[np.argsort(-decidedness[eligible_idx], kind="stable")]
    y = labels[order].astype(int)
    s = (-lean[order]).astype(np.float64)
    decided = decidedness[order].astype(np.float64)
    curve: List[Dict[str, Any]] = []
    for prefix_n in range(2, len(order) + 1):
        yp = y[:prefix_n]
        if len(np.unique(yp)) < 2:
            continue
        sp = s[:prefix_n]
        auc = float(roc_auc_score(yp, sp))
        ci_lo, ci_hi = _bootstrap_ci_fixed_direction(
            sp, yp, n_resamples=n_bootstrap, seed=seed,
        )
        curve.append({
            "n_prefix": int(prefix_n),
            "coverage": float(prefix_n / total),
            "auroc_b_signed": auc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "min_decidedness": float(decided[prefix_n - 1]),
        })
    significant = [p for p in curve if p["ci_lo"] is not None and p["ci_lo"] > 0.50]
    max_sig_cov = max((p["coverage"] for p in significant), default=0.0)
    best_point = max(significant, key=lambda p: p["coverage"], default=None)
    auroc_full = curve[-1] if curve else None
    return {
        "curve": curve,
        "max_significant_coverage": max_sig_cov,
        "best_point": best_point,
        "auroc_at_full_eligible": auroc_full,
        "n_eligible": int(eligible_mask.sum()),
        "eligible_coverage": float(eligible_mask.sum() / total),
    }


def run_audit(args: argparse.Namespace) -> int:
    frozen_run = Path(args.frozen_run).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir == frozen_run:
        raise SystemExit(
            "refusing to write audit output into the frozen locked run dir"
        )

    data_path = Path(args.data).expanduser().resolve()
    prompts, labels, data_hash = _load_calibration_jsonl(str(data_path))

    spec_path = spec_path_for(frozen_run, args.model)
    if not spec_path.exists():
        raise SystemExit(f"frozen spec not found (read-only input): {spec_path}")
    spec = _load_json(spec_path)
    if spec.get("data_hash_sha256") != data_hash:
        raise SystemExit(
            f"slice hash {data_hash} != frozen spec hash "
            f"{spec.get('data_hash_sha256')} -- not the locked slice"
        )

    yes_ids = [int(x) for x in spec["yes_token_ids"]]
    no_ids = [int(x) for x in spec["no_token_ids"]]
    sem_yes_ids = [int(x) for x in spec["semantic_yes_token_ids"]]
    sem_no_ids = [int(x) for x in spec["semantic_no_token_ids"]]
    control_ids = [int(x) for x in spec["control_token_ids"]]
    control_set = set(control_ids)

    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False
    model, tokenizer, _proj, _li = pipeline.load_model(args.model, cfg)
    strategy = io_plugins.get_prompt_strategy(args.model)
    if strategy.__name__ != spec.get("prompt_strategy_name"):
        raise SystemExit(
            f"prompt strategy {strategy.__name__} != frozen spec "
            f"{spec.get('prompt_strategy_name')} -- refusing to proceed"
        )

    # Identify newline token id(s) from the frozen control set + tokenizer.
    # The 2026-05-18 audit observation: Phi-3.5 dominant non-literal top-1 at
    # t=0 is "\n" (38/200). We tag rows where the model's t=0 greedy choice
    # decodes to a newline-class token for the conditional subset.
    newline_normalized = {"", "\n", "\\n"}

    n = len(prompts)
    rows: List[Dict[str, Any]] = []
    t0_full_topk: Dict[int, List[Dict[str, Any]]] = {}

    for i in range(n):
        wrapped = strategy(prompts[i], tokenizer)
        # t=0 forward (locked-equivalent path). Use prefix_readout so it goes
        # through the SAME code path that produced the locked numbers.
        prefix = pipeline.prefix_readout(model, tokenizer, wrapped)
        probs0 = prefix["last_probs"]
        prompt_ids = list(prefix["token_ids"])
        sc0 = _score_row_from_probs(
            probs0, yes_ids, no_ids, sem_yes_ids, sem_no_ids, control_ids
        )
        t0_full_topk[i] = _topk_records(tokenizer, probs0, k=10)

        # Greedy t=0 choice = model's own next token. Most faithful continuation
        # of the locked measurement: ask what the model believes ONE token past
        # the prefix it actually would have emitted.
        t0_top1_id = int(np.argmax(probs0))
        t0_top1_decoded = pipeline.decode_ids(tokenizer, [t0_top1_id])
        t0_top1_normalized = _normalize_literal_token(t0_top1_decoded)
        t0_top1_class = (
            "literal" if (t0_top1_id in set(yes_ids) | set(no_ids))
            else "control" if t0_top1_id in control_set
            else "other"
        )
        # Newline tag: a row counts as "t=0 newline-dominated" if the model's
        # greedy t=0 choice normalizes to empty/whitespace AND lives in the
        # control bucket (the precise condition from the 2026-05-18 finding).
        is_newline_t0 = bool(
            t0_top1_class == "control"
            and t0_top1_normalized in newline_normalized
        )

        # t=1 forward: append t=0 greedy token id, re-run, read last position.
        t1_token_ids = prompt_ids + [t0_top1_id]
        probs1 = _forward_last_probs(model, t1_token_ids)
        sc1 = _score_row_from_probs(
            probs1, yes_ids, no_ids, sem_yes_ids, sem_no_ids, control_ids
        )
        t1_top1_id = int(np.argmax(probs1))
        t1_top1_decoded = pipeline.decode_ids(tokenizer, [t1_top1_id])

        rows.append({
            "sample_idx": i,
            "label_B": int(labels[i]),
            "gold_yes_no": _gold_yes_no(int(labels[i])),
            # t=0 (recomputed; integrity-gated against locked CSV below)
            "t0_p_yes": sc0["p_yes"],
            "t0_p_no": sc0["p_no"],
            "t0_decidedness": sc0["decidedness"],
            "t0_control_mass": sc0["control_mass"],
            "t0_decidedness_floor": sc0["decidedness_floor"],
            "t0_above_floor": sc0["above_floor"],
            "t0_lean": sc0["lean"],
            "t0_top1_token_id": t0_top1_id,
            "t0_top1_decoded": t0_top1_decoded,
            "t0_top1_class": t0_top1_class,
            "is_newline_t0": is_newline_t0,
            # t=1 (the locus-offset measurement)
            "t1_p_yes": sc1["p_yes"],
            "t1_p_no": sc1["p_no"],
            "t1_decidedness": sc1["decidedness"],
            "t1_control_mass": sc1["control_mass"],
            "t1_decidedness_floor": sc1["decidedness_floor"],
            "t1_above_floor": sc1["above_floor"],
            "t1_lean": sc1["lean"],
            "t1_top1_token_id": t1_top1_id,
            "t1_top1_decoded": t1_top1_decoded,
        })

    # Integrity gate (mandatory — license to reinterpret)
    gate = _gate_against_locked_t0(
        model_slug=args.model,
        t0_rows=rows,
        locked_csv=readout_csv_path_for(frozen_run, args.model),
        canary_path=canary_path_for(frozen_run, args.model),
        t0_full_topk=t0_full_topk,
    )

    # ---- Full-panel t=1 verdict (same pre-reg rule, new locus) ----
    labels_arr = np.asarray([r["label_B"] for r in rows], dtype=np.int32)
    t1_lean = np.asarray([r["t1_lean"] for r in rows], dtype=np.float64)
    t1_dec = np.asarray([r["t1_decidedness"] for r in rows], dtype=np.float64)
    t1_eligible = np.asarray([r["t1_above_floor"] for r in rows], dtype=bool)
    n_total = len(rows)

    t1_metrics = _coverage_curve_and_max(
        labels=labels_arr,
        lean=t1_lean,
        decidedness=t1_dec,
        eligible_mask=t1_eligible,
        total=n_total,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.bootstrap_seed),
    )
    # Undetermined-coverage path doesn't change here; locus shift can only move
    # rows in/out of eligible. We compute it for verdict-rule completeness.
    undetermined_cov = 0.0  # semantic mass doesn't dominate literal at t=1 by construction here
    t1_verdict = classify_verdict(
        t1_metrics["curve"],
        eligible_coverage=t1_metrics["eligible_coverage"],
        undetermined_coverage=undetermined_cov,
    )

    # ---- Conditional subset: only rows where t=0 top-1 was newline ----
    newline_mask = np.asarray([r["is_newline_t0"] for r in rows], dtype=bool)
    n_newline = int(newline_mask.sum())
    cond = None
    if n_newline > 0:
        sub_labels = labels_arr[newline_mask]
        sub_t1_lean = t1_lean[newline_mask]
        sub_t1_dec = t1_dec[newline_mask]
        sub_t1_elig = t1_eligible[newline_mask]
        cond_metrics = _coverage_curve_and_max(
            labels=sub_labels,
            lean=sub_t1_lean,
            decidedness=sub_t1_dec,
            eligible_mask=sub_t1_elig,
            total=n_newline,
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.bootstrap_seed),
        )
        cond = {
            "n_newline_t0": n_newline,
            "n_eligible_at_t1": int(sub_t1_elig.sum()),
            "eligible_coverage_within_newline_subset": float(sub_t1_elig.sum() / n_newline),
            "auroc_at_full_eligible": cond_metrics["auroc_at_full_eligible"],
            "max_significant_coverage": cond_metrics["max_significant_coverage"],
            "best_point": cond_metrics["best_point"],
        }

    # ---- Diagnostic: was the locked-row situation actually rescued? ----
    # Counts of rows that newly clear / newly fail the floor when moving t=0 -> t=1
    t0_above = np.asarray([r["t0_above_floor"] for r in rows], dtype=bool)
    newly_above_at_t1 = int(((~t0_above) & t1_eligible).sum())
    newly_below_at_t1 = int((t0_above & (~t1_eligible)).sum())

    summary = {
        "schema_version": "locus_offset_audit_v1",
        "analysis_kind": "post_hoc_sensitivity_readonly",
        "model": args.model,
        "data_hash_sha256": data_hash,
        "frozen_run_dir": str(frozen_run),
        "frozen_spec_path": str(spec_path),
        "prompt_strategy_name": strategy.__name__,
        "n_total": n_total,
        "control_floor_multiplier": CONTROL_FLOOR_MULTIPLIER,
        "high_coverage_bar": HIGH_COVERAGE_BAR,
        "t0_protocol": "prefix_readout last-position softmax (locked-equivalent path)",
        "t1_protocol": (
            "Append model's own greedy t=0 top-1 token id to prompt_ids; "
            "re-run forward; read softmax at the new last position. Most "
            "faithful continuation of the locked measurement."
        ),
        "integrity_gate": gate,
        "t1_full_panel": {
            "verdict": t1_verdict,
            "eligible_coverage": t1_metrics["eligible_coverage"],
            "n_eligible": t1_metrics["n_eligible"],
            "max_significant_coverage": t1_metrics["max_significant_coverage"],
            "best_point": t1_metrics["best_point"],
            "auroc_at_full_eligible": t1_metrics["auroc_at_full_eligible"],
        },
        "t1_conditional_on_newline_t0": cond,
        "locus_shift_diagnostics": {
            "n_newline_dominated_at_t0": n_newline,
            "n_t0_above_floor": int(t0_above.sum()),
            "n_t1_above_floor": int(t1_eligible.sum()),
            "n_newly_above_at_t1": newly_above_at_t1,
            "n_newly_below_at_t1": newly_below_at_t1,
        },
        "note": (
            "Retrospective sensitivity analysis only. Does NOT amend or "
            "re-score the preregistered locked panel. Locked artifacts read "
            "read-only; integrity gate confirms the t=0 forward pass "
            "reproduces the locked p_yes/p_no/lean and the frozen canary "
            "top-10 within tolerance. Pre-reg rule (5.0x control_mass floor, "
            "literal YES/NO partition, -lean B-AUROC) is reused verbatim at "
            "the new locus; nothing about the rule is changed."
        ),
    }

    csv_path = out_dir / f"{short_model_name(args.model)}_t1_locus_offset.csv"
    json_path = out_dir / f"{short_model_name(args.model)}_t1_locus_offset.json"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _write_json(json_path, _sanitize_for_json(summary))

    print(f"[locus-offset] wrote {csv_path}")
    print(f"[locus-offset] wrote {json_path}")
    print(
        f"[locus-offset] gate OK "
        f"max|dp|={gate['max_abs_prob_drift_vs_locked_csv']:.2e} "
        f"max|dlean|={gate['max_abs_lean_drift_vs_locked_csv']:.2e} "
        f"max|dcanary|={gate['max_abs_prob_drift_vs_frozen_canary']:.2e}"
    )
    full = summary["t1_full_panel"]
    auroc_full = full["auroc_at_full_eligible"]
    auroc_s = (
        f"{auroc_full['auroc_b_signed']:.3f} "
        f"[{auroc_full['ci_lo']:.3f}, {auroc_full['ci_hi']:.3f}]"
    ) if auroc_full else "n/a"
    print(
        f"[locus-offset] t=1 verdict={full['verdict']} "
        f"eligible_cov={full['eligible_coverage']:.3f} "
        f"(n_elig={full['n_eligible']}/{n_total}) "
        f"max_sig_cov={full['max_significant_coverage']:.3f} "
        f"AUROC@full_elig={auroc_s}"
    )
    if cond is not None:
        cond_auroc = cond["auroc_at_full_eligible"]
        cond_auroc_s = (
            f"{cond_auroc['auroc_b_signed']:.3f} "
            f"[{cond_auroc['ci_lo']:.3f}, {cond_auroc['ci_hi']:.3f}]"
        ) if cond_auroc else "n/a"
        print(
            f"[locus-offset] conditional (t=0 was newline, n={cond['n_newline_t0']}): "
            f"t=1 eligible_cov_within_subset={cond['eligible_coverage_within_newline_subset']:.3f} "
            f"(n_elig={cond['n_eligible_at_t1']}/{cond['n_newline_t0']}) "
            f"AUROC@full_elig={cond_auroc_s}"
        )
    diag = summary["locus_shift_diagnostics"]
    print(
        f"[locus-offset] locus-shift: t0_above={diag['n_t0_above_floor']} "
        f"t1_above={diag['n_t1_above_floor']} "
        f"newly_above@t1={diag['n_newly_above_at_t1']} "
        f"newly_below@t1={diag['n_newly_below_at_t1']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="post-hoc t=1 locus-offset sensitivity audit for Phi-3.5-mini (read-only)"
    )
    p.add_argument(
        "--model",
        default="mlx-community/Phi-3.5-mini-instruct-4bit",
        help="model slug; scope of this audit is Phi-3.5 only",
    )
    p.add_argument(
        "--data",
        default=str(
            REPO_ROOT
            / "experiments" / "anli-sweep" / "2026-05-15" / "run-02"
            / "anli_R1_seed20260513_n100.jsonl"
        ),
    )
    p.add_argument(
        "--frozen-run",
        default=str(
            REPO_ROOT / "experiments" / "t0-mech-prep" / "2026-05-17" / "run-02"
        ),
        help="locked run dir holding frozen specs/canaries/readouts (read-only)",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=20260423)
    p.set_defaults(func=run_audit)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
