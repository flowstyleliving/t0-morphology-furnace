#!/usr/bin/env python3
"""Build the t0-prep panel-cell coverage matrix from calibrator profile JSONs.

Reads `experiments/t0-prep-calibrator-sweep/<DATE>/<RUN>/{v_norms,multistep}/*.profile.json`
and emits:
  1. A CSV at `<RUN>/coverage_matrix.csv` — one row per (model × panel × cell)
  2. A markdown summary to stdout — per-model winners + headline numbers,
     paste-friendly into `wiki/results/t0-prep-coverage-matrix-<DATE>.md`

OOB fields (`oob_median`, `oob_ci_lo`, `oob_ci_hi`, `winner_stability`) live at
profile level, not cell level — populated only on the winner row per panel.

Re-runnable: walking the directories every time means it tolerates partial
sweeps (some models' profiles missing) and rebuilds cleanly when more land.

Usage:
    .venv/bin/python scripts/build_t0_coverage_matrix.py \\
        [--sweep-dir experiments/t0-prep-calibrator-sweep/<DATE>/<RUN>]

Default sweep-dir: experiments/t0-prep-calibrator-sweep/2026-05-16/run-01.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

ATTENTION_LAYERS = ("final", "mid", "last_minus_1")  # ordered for prefix-match


def parse_attention_cell_label(cell_label: str) -> Optional[Tuple[int, str, str]]:
    """Parse e.g. 'attention[final_js_kv_groups] @ step 1' → (1, 'final', 'js_kv_groups').

    Anchors on known layer prefixes (last_minus_1 has underscores; metric names
    also do — naive rsplit on '_' breaks). Returns None on unparseable input.
    """
    parts = cell_label.split(" @ step ")
    if len(parts) != 2:
        return None
    try:
        step = int(parts[1])
    except ValueError:
        return None
    inner = parts[0]
    if not (inner.startswith("attention[") and inner.endswith("]")):
        return None
    layer_metric = inner[len("attention["):-1]
    for layer in ATTENTION_LAYERS:
        prefix = f"{layer}_"
        if layer_metric.startswith(prefix):
            metric = layer_metric[len(prefix):]
            return (step, layer, metric)
    return None


def rows_from_profile(profile_path: Path, panel_type: str) -> List[Dict[str, Any]]:
    """Read one profile JSON; return one row per candidate-panel cell."""
    with open(profile_path) as f:
        prof = json.load(f)

    model = prof["model"]["slug"]
    detector = prof["detector"]
    winner_family = detector["metric"]["family"]
    winner_label = detector["metric"]["label"]
    winner_step = int(detector["gen_step"])

    cs = prof["calibration_stats"]
    oob_median = cs.get("oob_auroc_median")
    oob_ci_lo = cs.get("oob_auroc_ci_lo")
    oob_ci_hi = cs.get("oob_auroc_ci_hi")
    winner_stability = cs.get("winner_stability")
    n_warnings = len(prof.get("warnings", []))

    try:
        rel_path = str(profile_path.relative_to(REPO_ROOT))
    except ValueError:
        # profile lives outside the repo (e.g. /tmp during testing) — record absolute
        rel_path = str(profile_path)

    rows: List[Dict[str, Any]] = []
    for cell in cs.get("candidate_panel", []):
        # Only handle Attention-family rows here — this script is t0-prep specific.
        if cell.get("family") != "Attention":
            continue
        parsed = parse_attention_cell_label(cell.get("cell", ""))
        if parsed is None:
            step_val, layer, metric = -1, "?", "?"
        else:
            step_val, layer, metric = parsed
        is_winner = (
            cell.get("family") == winner_family
            and cell.get("rank_label") == winner_label
            and int(cell.get("step", -1)) == winner_step
        )
        rows.append({
            "model": model,
            "panel": panel_type,
            "gen_step": step_val,
            "layer": layer,
            "metric": metric,
            "auroc": _fmt(cell.get("auroc")),
            "sign": cell.get("sign", ""),
            "n_eval": cell.get("n_evaluated", ""),
            "oob_median": _fmt(oob_median) if is_winner else "",
            "oob_ci_lo": _fmt(oob_ci_lo) if is_winner else "",
            "oob_ci_hi": _fmt(oob_ci_hi) if is_winner else "",
            "winner_stability": _fmt(winner_stability) if is_winner else "",
            "warnings_count": n_warnings,
            "is_winner": "TRUE" if is_winner else "FALSE",
            "profile_path": rel_path,
        })
    return rows


def _fmt(v: Any) -> str:
    """Format float to 4 decimals; passthrough non-floats; '' for None/NaN."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return ""
    return f"{f:.4f}"


def emit_markdown_summary(rows: List[Dict[str, Any]]) -> None:
    """Print a paste-into-wiki markdown summary to stdout."""
    if not rows:
        print("\n_No profiles found — sweep still pending._")
        return

    print("\n## Per-(model, panel) winners\n")
    print("| Model | Panel | Winner cell | AUROC | sign | OOB median | OOB CI | Stability | Warnings |")
    print("|---|---|---|---|:--:|---|---|---|---|")
    winners = [r for r in rows if r["is_winner"] == "TRUE"]
    winners.sort(key=lambda r: (r["model"], r["panel"]))
    for r in winners:
        short_model = r["model"].split("/")[-1]
        cell = f"{r['layer']}_{r['metric']} @ step {r['gen_step']}"
        ci = (
            f"[{r['oob_ci_lo']}, {r['oob_ci_hi']}]"
            if r["oob_ci_lo"] != "" and r["oob_ci_hi"] != ""
            else "—"
        )
        try:
            sign_str = f"{int(r['sign']):+d}"
        except (TypeError, ValueError):
            sign_str = "—"  # sign absent from profile JSON → defaulted to ""
        print(
            f"| {short_model} | {r['panel']} | {cell} | {r['auroc']} | {sign_str} | "
            f"{r['oob_median']} | {ci} | {r['winner_stability']} | {r['warnings_count']} |"
        )

    # Quick aggregate: how many models per panel have a winner with clean OOB CI?
    n_v_norms = sum(1 for r in winners if r["panel"] == "v_norms")
    n_multistep = sum(1 for r in winners if r["panel"] == "multistep")
    print(f"\n_Total winners surfaced: {len(winners)} ({n_v_norms} v_norms + {n_multistep} multistep)._")


# ─── Step 2.3: RAUQ + SinkProbe head-to-head join (NON-DESTRUCTIVE) ──────────
# Emits a sibling head_to_head.csv + markdown beside coverage_matrix.csv.
# coverage_matrix.csv itself is never read back or rewritten here — Step-1
# reproducibility is preserved. Re-runnable; tolerates missing baseline dirs.

RAUQ_DIR_DEFAULT = REPO_ROOT / "experiments" / "t0-baselines" / "2026-05-16" / "run-01" / "rauq"
SINK_DIR_DEFAULT = REPO_ROOT / "experiments" / "t0-baselines" / "2026-05-16" / "run-01" / "sinkprobe"
JS_METRICS = ("js", "js_kv_groups", "js_no_bos")
# Sentinel for "no Step-1 calibrator winner exists for this model" (e.g.
# Llama-3.1-8B, baseline-only). Distinct from "winner exists but has no OOB".
_OURS_BASELINE_ONLY = "N/A (baseline-only)"


def _f(v: Any) -> Optional[float]:
    """Parse a possibly-'' / None / str cell to float; None if not finite."""
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _ours_best(all_rows: List[Dict[str, Any]], model: str) -> Optional[Dict[str, Any]]:
    """Best Step-1 calibrator winner for `model`, ranked by OOB median
    (selection-bias-corrected) then in-sample AUROC. trust='clean' iff the
    OOB CI excludes 0.5 AND winner_stability >= 0.70 (flagged flukes like
    the Mistral-7B multistep last_minus_1_js@step3 thereby auto-demote)."""
    wins = [r for r in all_rows if r["model"] == model and r.get("is_winner") == "TRUE"]
    if not wins:
        return None

    def _key(r: Dict[str, Any]) -> Tuple[int, float]:
        # Never compare OOB-scale against in-sample-scale across rows: any
        # winner WITH an OOB median outranks every winner without one; the
        # in-sample auroc is only used to break ties among winners that all
        # lack OOB. Preserves the documented "OOB-median-first" intent.
        oob = _f(r.get("oob_median"))
        if oob is not None:
            return (1, oob)
        return (0, _f(r.get("auroc")) or 0.0)

    best = max(wins, key=_key)
    lo, hi, stab = _f(best.get("oob_ci_lo")), _f(best.get("oob_ci_hi")), _f(best.get("winner_stability"))
    clean = (
        lo is not None and hi is not None and stab is not None
        and (lo > 0.5 or hi < 0.5) and stab >= 0.70
    )
    return {
        "cell": f"{best['layer']}_{best['metric']} @ {best['panel']} step{best['gen_step']}",
        "auroc": _f(best.get("auroc")),
        "oob": _f(best.get("oob_median")),
        "trust": "clean" if clean else "flagged",
    }


def _ours_js_best(all_rows: List[Dict[str, Any]], model: str) -> Optional[float]:
    """Best in-sample AUROC among the model's js* cells (any panel) — the
    Step-2.3 'sink-driven framing' verification comparator."""
    vals = [
        _f(r.get("auroc")) for r in all_rows
        if r["model"] == model and r.get("metric") in JS_METRICS and _f(r.get("auroc")) is not None
    ]
    return max(vals) if vals else None


def _baseline_best(prof: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Best single-cell FIXED-direction AUROC (+ its sign-free) for a RAUQ or
    SinkProbe profile dict. Aggregate is NOT used for RAUQ (it sandbags when
    per-layer directions disagree) but its sign-free is carried for context."""
    best: Dict[str, Any] = {"auroc": None, "cell": "", "dir": "", "sf": None}
    agg_sf = None
    if kind == "rauq":
        for variant in ("1a_commit_only", "1b_prompt_recurrence"):
            res = prof["results"].get(variant, {})
            short = "1a" if variant.startswith("1a") else "1b"
            for layer, a in res.get("per_layer", {}).items():
                au = a.get("auroc")
                if au is not None and (best["auroc"] is None or au > best["auroc"]):
                    best = {"auroc": au, "cell": f"{short}/{layer}",
                            "dir": a.get("direction", ""), "sf": a.get("auroc_signfree")}
            asf = res.get("aggregate_max", {}).get("auroc_signfree")
            if asf is not None:
                agg_sf = asf if agg_sf is None else max(agg_sf, asf)
    else:  # sinkprobe
        for layer, mm in prof["results"].items():
            for metric, a in mm.items():
                au = a.get("auroc")
                if au is not None and (best["auroc"] is None or au > best["auroc"]):
                    best = {"auroc": au, "cell": f"{metric}/{layer}",
                            "dir": a.get("direction", ""), "sf": a.get("auroc_signfree")}
    best["agg_sf"] = agg_sf
    best["model"] = prof["model"]["slug"]
    best["data_hash"] = prof.get("data", {}).get("data_hash_sha256", "")
    return best


def _load_baseline_dir(d: Path, suffix: str, kind: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not d.exists():
        return out
    for p in sorted(d.glob(f"*{suffix}")):
        try:
            b = _baseline_best(json.loads(p.read_text()), kind)
            out[b["model"]] = b
        except Exception as e:  # tolerate a single corrupt file
            print(f"[h2h] error reading {p}: {e}", file=sys.stderr)
    return out


def _calib_data_hash(profile_path_str: str) -> str:
    try:
        return json.loads(
            (REPO_ROOT / profile_path_str).read_text()
        ).get("task", {}).get("data_hash_sha256", "")
    except Exception:
        return ""


def build_head_to_head(
    all_rows: List[Dict[str, Any]], rauq_dir: Path, sink_dir: Path
) -> List[Dict[str, Any]]:
    rauq = _load_baseline_dir(rauq_dir, ".rauq.json", "rauq")
    sink = _load_baseline_dir(sink_dir, ".sinkprobe.json", "sinkprobe")
    out: List[Dict[str, Any]] = []
    for m in sorted(set(rauq) | set(sink)):
        ob, rq, sk = _ours_best(all_rows, m), rauq.get(m), sink.get(m)
        prow = next((r for r in all_rows if r["model"] == m and r.get("profile_path")), None)
        ch = _calib_data_hash(prow["profile_path"]) if prow else ""
        hashes = [h for h in (ch, rq["data_hash"] if rq else "", sk["data_hash"] if sk else "") if h]
        data_hash_ok = len(hashes) >= 2 and len(set(hashes)) == 1
        # OOB-only: a winner without an OOB median must NOT compete on
        # in-sample AUROC (that re-introduces the scale-mix the _key fix
        # closed, and would let "ours" win a row the markdown labels N/A).
        ours_cmp = ob["oob"] if (ob and ob.get("oob") is not None) else None
        fixed = {k: v for k, v in (
            ("ours", ours_cmp), ("rauq", rq["auroc"] if rq else None),
            ("sinkprobe", sk["auroc"] if sk else None)) if v is not None}
        # ours has no sign-free concept (calibrator AUROC is direction-locked);
        # use the SAME OOB-gated comparator as winner_fixed so the secondary
        # column can't name an un-OOB'd "ours" the primary column excludes.
        sf = {k: v for k, v in (
            ("ours", ours_cmp), ("rauq", rq["sf"] if rq else None),
            ("sinkprobe", sk["sf"] if sk else None)) if v is not None}
        out.append({
            "model": m.split("/")[-1],
            "ours_cell": ob["cell"] if ob else _OURS_BASELINE_ONLY,
            "ours_auroc": ob["auroc"] if ob else None,
            "ours_oob": ob["oob"] if ob else None,
            "ours_trust": ob["trust"] if ob else "n/a",
            "rauq_cell": rq["cell"] if rq else "",
            "rauq_auroc_fixed": rq["auroc"] if rq else None,
            "rauq_dir": rq["dir"] if rq else "",
            "rauq_sf": rq["sf"] if rq else None,
            "rauq_agg_sf": rq.get("agg_sf") if rq else None,
            "sink_cell": sk["cell"] if sk else "",
            "sink_auroc_fixed": sk["auroc"] if sk else None,
            "sink_dir": sk["dir"] if sk else "",
            "sink_sf": sk["sf"] if sk else None,
            "winner_fixed": max(fixed, key=fixed.get) if fixed else "",
            "winner_signfree": max(sf, key=sf.get) if sf else "",
            "data_hash_ok": "TRUE" if data_hash_ok else "FALSE",
        })
    return out


_H2H_COLS = [
    "model", "ours_cell", "ours_auroc", "ours_oob", "ours_trust",
    "rauq_cell", "rauq_auroc_fixed", "rauq_dir", "rauq_sf", "rauq_agg_sf",
    "sink_cell", "sink_auroc_fixed", "sink_dir", "sink_sf",
    "winner_fixed", "winner_signfree", "data_hash_ok",
]


def emit_head_to_head_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_H2H_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in _H2H_COLS})


def emit_head_to_head_markdown(rows: List[Dict[str, Any]]) -> None:
    print("\n## RAUQ + SinkProbe vs ours — head-to-head (Step 2.3)\n")
    print("_Winner on FIXED-direction basis (honest; no post-hoc sign). `ours` "
          "comparator = Step-1 OOB median. sf = sign-free sensitivity._\n")
    print("| Model | ours (OOB) trust | RAUQ best (fixed/sf) | SinkProbe best (fixed/sf) | winner | winner(sf) | hash✓ |")
    print("|---|---|---|---|:--:|:--:|:--:|")
    for r in rows:
        def fmt(v):
            return "—" if v is None else f"{float(v):.3f}"
        if r["ours_cell"] == _OURS_BASELINE_ONLY:
            ours = _OURS_BASELINE_ONLY                       # no calibrator winner
        elif r["ours_oob"] is not None:
            ours = f"{fmt(r['ours_oob'])} ({r['ours_trust']})"
        else:                                                # winner exists, un-OOB'd
            ours = f"{fmt(r['ours_auroc'])} in-sample (no OOB)"
        print(
            f"| {r['model']} | {ours} | {r['rauq_cell']} {fmt(r['rauq_auroc_fixed'])}/{fmt(r['rauq_sf'])} "
            f"| {r['sink_cell']} {fmt(r['sink_auroc_fixed'])}/{fmt(r['sink_sf'])} "
            f"| {r['winner_fixed']} | {r['winner_signfree']} | {r['data_hash_ok']} |"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--sweep-dir",
        default=str(REPO_ROOT / "experiments" / "t0-prep-calibrator-sweep" / "2026-05-16" / "run-01"),
        help="Calibrator sweep root containing {v_norms,multistep}/ subdirs",
    )
    p.add_argument("--rauq-dir", default=str(RAUQ_DIR_DEFAULT),
                   help="dir of *.rauq.json (Step 2.1); skipped if absent")
    p.add_argument("--sinkprobe-dir", default=str(SINK_DIR_DEFAULT),
                   help="dir of *.sinkprobe.json (Step 2.2); skipped if absent")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    panel_dirs = {
        "v_norms": sweep_dir / "v_norms",
        "multistep": sweep_dir / "multistep",
    }

    all_rows: List[Dict[str, Any]] = []
    for panel_type, panel_dir in panel_dirs.items():
        if not panel_dir.exists():
            print(f"[matrix] {panel_dir} not found — skipping (sweep pending)", file=sys.stderr)
            continue
        for profile_path in sorted(panel_dir.glob("*.profile.json")):
            try:
                all_rows.extend(rows_from_profile(profile_path, panel_type))
            except Exception as e:
                print(f"[matrix] error reading {profile_path}: {e}", file=sys.stderr)

    cols = [
        "model", "panel", "gen_step", "layer", "metric",
        "auroc", "sign", "n_eval",
        "oob_median", "oob_ci_lo", "oob_ci_hi", "winner_stability",
        "warnings_count", "is_winner", "profile_path",
    ]
    out_csv = sweep_dir / "coverage_matrix.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[matrix] wrote {len(all_rows)} rows to {out_csv}", file=sys.stderr)

    emit_markdown_summary(all_rows)

    # ── Step 2.3 head-to-head (sibling artifact; coverage_matrix.csv untouched) ──
    rauq_dir, sink_dir = Path(args.rauq_dir).resolve(), Path(args.sinkprobe_dir).resolve()
    if rauq_dir.exists() or sink_dir.exists():
        h2h = build_head_to_head(all_rows, rauq_dir, sink_dir)
        if h2h:
            h2h_csv = sweep_dir / "head_to_head.csv"
            emit_head_to_head_csv(h2h, h2h_csv)
            print(f"[matrix] wrote {len(h2h)} head-to-head rows to {h2h_csv}", file=sys.stderr)
            emit_head_to_head_markdown(h2h)
            # Plan verification: SinkProbe should win/tie our js_* on the
            # two SinkProbe-aligned panel models.
            by_short = {r["model"]: r for r in h2h}
            full_by_short = {row["model"].split("/")[-1]: row["model"] for row in all_rows}
            for vm in ("Llama-3.2-3B-Instruct-4bit", "Mistral-Nemo-Instruct-2407-4bit"):
                hr = by_short.get(vm)
                if not hr or hr.get("sink_auroc_fixed") is None:
                    continue
                ojs = _ours_js_best(all_rows, full_by_short.get(vm, ""))
                sink_sf = hr.get("sink_sf")
                verdict = (
                    "n/a (no ours js*)" if ojs is None
                    else ("SinkProbe ≥ ours js*" if (sink_sf or 0) >= ojs - 1e-9
                          else "ours js* > SinkProbe")
                )
                print(f"[h2h-verify] {vm}: ours js*={ojs} sink(sf)={sink_sf} → {verdict}",
                      file=sys.stderr)
    else:
        print("[matrix] no baseline dirs found — head-to-head skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
