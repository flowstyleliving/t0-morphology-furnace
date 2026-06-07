#!/usr/bin/env python3
"""Post-hoc t=0 answer-recoverability sensitivity audit (read-only).

This is a RETROSPECTIVE sensitivity analysis, not a re-score of the locked
step-0 belief panel. It does not modify, re-derive, or re-interpret the
preregistered result. It answers one question the locked CSVs cannot, because
they only persist ``top1_*`` (not full top-k):

    When the literal YES/NO buckets did not carry the t=0 mass, where did the
    mass go -- to an answer-like non-literal form (Correct / True / Right /
    False / ...) or to non-answer continuation tokens (To / Based / Let ...)?

Contract / guarantees
---------------------
* Inputs are the SAME frozen ``n=200 x 10`` slice and the SAME per-model
  frozen ``*_belief_spec.json`` used by the locked panel. The frozen
  semantic shortlist in the spec IS the answer-alternative shortlist -- it is
  reused verbatim, never re-fit here.
* The locked artifacts in the frozen run dir are opened read-only. Nothing is
  written back into ``run-02``; all output goes to a separate audit dir.
* A new forward pass is required (full top-k at t=0 was never stored). To prove
  the forward pass faithfully reproduces the locked measurement, an integrity
  gate recomputes the 3 frozen canary samples and every locked
  ``p_yes / p_no / lean`` and aborts on any drift beyond tolerance. Only a
  byte-faithful forward pass earns the right to reinterpret the locked slice.

Scoring math is imported verbatim from ``scripts.step0_belief_readout`` so the
literal / semantic / control partition is identical to the locked run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_v2_io_plugins as io_plugins
import pri_v2_mlx_pipeline as pipeline
from pri_calibrator import _load_calibration_jsonl
from scripts.step0_belief_readout import (
    EPS,
    _gold_yes_no,
    _load_json,
    _normalize_literal_token,
    _sanitize_for_json,
    _score_row_from_probs,
    _write_json,
    canary_path_for,
    readout_csv_path_for,
    short_model_name,
    spec_path_for,
)

# Integrity-gate tolerances. The forward pass is the same model + same prompt
# + same softmax, so agreement should be ~fp32 noise; these are deliberately
# tight. Any breach means the audit is measuring something other than the
# locked run and must abort rather than silently "reinterpret".
PROB_ABS_TOL = 1e-6
LEAN_ABS_TOL = 1e-5
CANARY_PROB_ABS_TOL = 1e-6

# "Material" thresholds for the recoverability question. Reported alongside the
# raw masses so a reader can re-bucket; the verdict uses the >1x rule and the
# stricter >=2x rule, both prereg-style fixed before looking at outputs.
MATERIAL_RATIO = 2.0


def _audit_csv_path(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_t0_audit.csv"


def _audit_json_path(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_t0_audit.json"


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


def _first_in(records: Sequence[Dict[str, Any]], id_set: set) -> Dict[str, Any] | None:
    for rec in records:
        if rec["token_id"] in id_set:
            return rec
    return None


def _classify_token(token_id: int, *, literal: set, answerlike: set, control: set) -> str:
    if token_id in literal:
        return "literal"
    if token_id in answerlike:
        return "answerlike_nonliteral"
    if token_id in control:
        return "control"
    return "other"


def _gate_against_locked(
    *,
    model_slug: str,
    audit_rows: Sequence[Dict[str, Any]],
    locked_csv: Path,
    canary_path: Path,
    full_topk: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Abort unless the new forward pass reproduces the locked numbers."""
    with locked_csv.open(newline="") as f:
        locked = {int(r["sample_idx"]): r for r in csv.DictReader(f)}
    if len(locked) != len(audit_rows):
        raise RuntimeError(
            f"[{model_slug}] locked CSV has {len(locked)} rows, audit has "
            f"{len(audit_rows)} -- slice mismatch, refusing to proceed"
        )
    max_dp = 0.0
    max_dl = 0.0
    for row in audit_rows:
        lr = locked[row["sample_idx"]]
        dpy = abs(row["p_yes"] - float(lr["p_yes"]))
        dpn = abs(row["p_no"] - float(lr["p_no"]))
        dln = abs(row["lean"] - float(lr["lean"]))
        max_dp = max(max_dp, dpy, dpn)
        max_dl = max(max_dl, dln)
        if dpy > PROB_ABS_TOL or dpn > PROB_ABS_TOL or dln > LEAN_ABS_TOL:
            raise RuntimeError(
                f"[{model_slug}] integrity gate FAILED at sample "
                f"{row['sample_idx']}: |dp_yes|={dpy:.2e} |dp_no|={dpn:.2e} "
                f"|dlean|={dln:.2e} exceeds tol -- forward pass does not "
                f"reproduce the locked run; audit aborted"
            )

    canary = _load_json(canary_path)
    max_dc = 0.0
    n_canary = 0
    for s in canary.get("samples", []):
        idx = int(s["sample_idx"])
        recomputed = full_topk.get(idx)
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
            f"{canary_path} has no samples to check -- the top-10 "
            f"token-order verification did not run; refusing to report "
            f"the gate as passed on a corrupt/truncated canary"
        )
    return {
        "passed": True,
        "n_rows_checked": len(audit_rows),
        "n_canary_samples_checked": n_canary,
        "max_abs_prob_drift_vs_locked_csv": max_dp,
        "max_abs_lean_drift_vs_locked_csv": max_dl,
        "max_abs_prob_drift_vs_frozen_canary": max_dc,
        "prob_abs_tol": PROB_ABS_TOL,
        "lean_abs_tol": LEAN_ABS_TOL,
        "canary_prob_abs_tol": CANARY_PROB_ABS_TOL,
    }


def run_model_audit(args: argparse.Namespace) -> int:
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

    literal_set = set(yes_ids) | set(no_ids)
    semantic_set = set(sem_yes_ids) | set(sem_no_ids)
    # Answer-like-but-non-literal == the frozen semantic shortlist minus the
    # literal YES/NO buckets. This is exactly the "answer-alternative
    # shortlist" the question asks about, taken verbatim from the frozen spec.
    answerlike_nonliteral_set = semantic_set - literal_set
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

    n = len(prompts)
    rows: List[Dict[str, Any]] = []
    full_topk: Dict[int, List[Dict[str, Any]]] = {}
    for i in range(n):
        wrapped = strategy(prompts[i], tokenizer)
        prefix = pipeline.prefix_readout(model, tokenizer, wrapped)
        probs = prefix["last_probs"]
        sc = _score_row_from_probs(
            probs, yes_ids, no_ids, sem_yes_ids, sem_no_ids, control_ids
        )
        topk = _topk_records(tokenizer, probs, k=args.topk)
        full_topk[i] = topk

        literal_dec = sc["decidedness"]
        # Off-literal answer-like mass = frozen semantic shortlist mass that is
        # NOT already counted in the literal YES/NO buckets. Recomputed
        # directly here (not from the >=0 clipped field) so the ratio is exact.
        answerlike_nonliteral_mass = float(
            np.sum(probs[sorted(answerlike_nonliteral_set)], dtype=np.float64)
        ) if answerlike_nonliteral_set else 0.0

        top1 = topk[0]
        top1_nonliteral = next(
            (r for r in topk if r["token_id"] not in literal_set), None
        )
        top1_answerlike_nl = _first_in(topk, answerlike_nonliteral_set)

        ratio = (
            answerlike_nonliteral_mass / literal_dec
            if literal_dec > 0 else math.inf
        )
        rows.append({
            "sample_idx": i,
            "label_B": int(labels[i]),
            "gold_yes_no": _gold_yes_no(int(labels[i])),
            "p_yes": sc["p_yes"],
            "p_no": sc["p_no"],
            "literal_decidedness": literal_dec,
            "semantic_yes_mass": sc["semantic_yes_mass"],
            "semantic_no_mass": sc["semantic_no_mass"],
            "semantic_decidedness": sc["semantic_decidedness"],
            "answerlike_nonliteral_mass": answerlike_nonliteral_mass,
            "answerlike_over_literal_ratio": ratio,
            "control_mass": sc["control_mass"],
            "decidedness_floor": sc["decidedness_floor"],
            "literal_above_floor": sc["above_floor"],
            "semantic_above_floor": bool(
                sc["semantic_decidedness"] > sc["decidedness_floor"]
            ),
            "lean": sc["lean"],
            "top1_token_id": top1["token_id"],
            "top1_decoded": top1["decoded"],
            "top1_prob": top1["prob"],
            "top1_class": _classify_token(
                top1["token_id"],
                literal=literal_set,
                answerlike=answerlike_nonliteral_set,
                control=control_set,
            ),
            "top1_nonliteral_token_id": (
                top1_nonliteral["token_id"] if top1_nonliteral else None
            ),
            "top1_nonliteral_decoded": (
                top1_nonliteral["decoded"] if top1_nonliteral else None
            ),
            "top1_nonliteral_prob": (
                top1_nonliteral["prob"] if top1_nonliteral else None
            ),
            "top1_nonliteral_class": (
                _classify_token(
                    top1_nonliteral["token_id"],
                    literal=literal_set,
                    answerlike=answerlike_nonliteral_set,
                    control=control_set,
                )
                if top1_nonliteral else None
            ),
            "top1_answerlike_nl_decoded": (
                top1_answerlike_nl["decoded"] if top1_answerlike_nl else None
            ),
            "top1_answerlike_nl_prob": (
                top1_answerlike_nl["prob"] if top1_answerlike_nl else None
            ),
            "answerlike_exceeds_literal": bool(
                answerlike_nonliteral_mass > literal_dec
            ),
            "answerlike_materially_exceeds_literal": bool(
                answerlike_nonliteral_mass > MATERIAL_RATIO * literal_dec
            ),
            "top1_is_answerlike_nonliteral": bool(
                top1["token_id"] in answerlike_nonliteral_set
            ),
        })

    if not rows:
        raise SystemExit(
            f"empty data slice -- no prompts to audit for {args.model}; "
            f"check --data ({data_path})"
        )

    gate = _gate_against_locked(
        model_slug=args.model,
        audit_rows=rows,
        locked_csv=readout_csv_path_for(frozen_run, args.model),
        canary_path=canary_path_for(frozen_run, args.model),
        full_topk=full_topk,
    )

    n_f = float(n)
    n_exceeds = sum(1 for r in rows if r["answerlike_exceeds_literal"])
    n_mat = sum(1 for r in rows if r["answerlike_materially_exceeds_literal"])
    n_top1_nl = sum(1 for r in rows if r["top1_class"] != "literal")
    n_top1_al = sum(1 for r in rows if r["top1_is_answerlike_nonliteral"])
    n_lit_floor = sum(1 for r in rows if r["literal_above_floor"])
    n_sem_floor = sum(1 for r in rows if r["semantic_above_floor"])
    n_recover_only_sem = sum(
        1 for r in rows
        if (not r["literal_above_floor"]) and r["semantic_above_floor"]
    )

    frac_mat = n_mat / n_f
    frac_top1_al = n_top1_al / n_f
    # Conservative descriptive bucket. Thresholds fixed here, before reading
    # any model's outputs; reported so a reader can re-bucket from the raw CSV.
    if frac_mat < 0.02 and frac_top1_al < 0.02:
        verdict = "literal-panel-basically-complete"
    elif frac_mat < 0.10 and frac_top1_al < 0.10:
        verdict = "small-answerlike-tail"
    else:
        verdict = "literal-panel-understates-recoverability"

    summary = {
        "schema_version": "t0_recoverability_audit_v1",
        "analysis_kind": "post_hoc_sensitivity_readonly",
        "model": args.model,
        "data_hash_sha256": data_hash,
        "frozen_run_dir": str(frozen_run),
        "frozen_spec_path": str(spec_path),
        "prompt_strategy_name": strategy.__name__,
        "n_total": n,
        "topk_captured": int(args.topk),
        "material_ratio": MATERIAL_RATIO,
        "integrity_gate": gate,
        "answerlike_shortlist_source": "frozen spec semantic_*_token_ids minus literal YES/NO",
        "n_answerlike_nonliteral_token_ids": len(answerlike_nonliteral_set),
        "literal_above_floor_coverage": n_lit_floor / n_f,
        "semantic_above_floor_coverage": n_sem_floor / n_f,
        "recovered_only_by_semantic_coverage": n_recover_only_sem / n_f,
        "frac_answerlike_exceeds_literal": n_exceeds / n_f,
        "frac_answerlike_materially_exceeds_literal": frac_mat,
        "frac_top1_nonliteral": n_top1_nl / n_f,
        "frac_top1_answerlike_nonliteral": frac_top1_al,
        "mean_literal_decidedness": float(
            np.mean([r["literal_decidedness"] for r in rows])
        ),
        "mean_semantic_decidedness": float(
            np.mean([r["semantic_decidedness"] for r in rows])
        ),
        "mean_answerlike_nonliteral_mass": float(
            np.mean([r["answerlike_nonliteral_mass"] for r in rows])
        ),
        "sensitivity_verdict": verdict,
        "note": (
            "Retrospective sensitivity analysis only. Does NOT amend, "
            "re-score, or reinterpret the preregistered locked panel. "
            "Locked artifacts read read-only; integrity gate confirms the "
            "forward pass reproduces the locked p_yes/p_no/lean and the "
            "frozen canary top-10 within tolerance."
        ),
    }

    csv_path = _audit_csv_path(out_dir, args.model)
    json_path = _audit_json_path(out_dir, args.model)
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _write_json(json_path, _sanitize_for_json(summary))

    print(f"[t0-audit] wrote {csv_path}")
    print(f"[t0-audit] wrote {json_path}")
    print(
        f"[t0-audit] gate OK "
        f"max|dp|={gate['max_abs_prob_drift_vs_locked_csv']:.2e} "
        f"max|dlean|={gate['max_abs_lean_drift_vs_locked_csv']:.2e} "
        f"max|dcanary|={gate['max_abs_prob_drift_vs_frozen_canary']:.2e}"
    )
    print(
        f"[t0-audit] verdict={verdict} "
        f"frac_answerlike>literal={n_exceeds / n_f:.3f} "
        f"frac_materially={frac_mat:.3f} "
        f"frac_top1_answerlike={frac_top1_al:.3f} "
        f"recovered_only_by_semantic={n_recover_only_sem / n_f:.3f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="post-hoc t=0 answer-recoverability sensitivity audit (read-only)"
    )
    p.add_argument("--model", required=True)
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
    p.add_argument("--topk", type=int, default=50)
    p.set_defaults(func=run_model_audit)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
