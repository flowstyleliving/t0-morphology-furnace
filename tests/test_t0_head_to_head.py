"""Fast unit tests for the Step 2.3 head-to-head join in
scripts/build_t0_coverage_matrix.py. No model load.

Covers _f parsing, _ours_best (OOB ranking + clean/flagged trust + no-winner),
_baseline_best (RAUQ + SinkProbe best-fixed-cell selection), and a
build_head_to_head integration over tmp JSON files incl. the data-hash
parity column.

Run:  .venv/bin/pytest tests/test_t0_head_to_head.py -q
"""
from __future__ import annotations

import json

from scripts.build_t0_coverage_matrix import (
    _OURS_BASELINE_ONLY,
    _baseline_best,
    _f,
    _ours_best,
    build_head_to_head,
    emit_head_to_head_markdown,
    emit_markdown_summary,
)


def test_f_parsing():
    assert _f("") is None and _f(None) is None and _f("abc") is None
    assert _f("0.73") == 0.73 and _f(0.5) == 0.5
    assert _f(float("nan")) is None


def _winrow(**kw):
    base = dict(model="mlx-community/M", panel="v_norms", gen_step="1",
                layer="final", metric="js", auroc="", oob_median="",
                oob_ci_lo="", oob_ci_hi="", winner_stability="",
                is_winner="TRUE", profile_path="")
    base.update(kw)
    return base


def test_ours_best_ranks_by_oob_and_demotes_fluke():
    rows = [
        _winrow(panel="v_norms", metric="js", auroc="0.66", oob_median="0.60",
                oob_ci_lo="0.55", oob_ci_hi="0.72", winner_stability="0.80"),
        # Mistral-7B-like fluke: huge in-sample, OOB 0.50, CI spans 0.5, low stab.
        _winrow(panel="multistep", layer="last_minus_1", gen_step="3",
                metric="js", auroc="0.90", oob_median="0.50",
                oob_ci_lo="0.00", oob_ci_hi="1.00", winner_stability="0.22"),
    ]
    best = _ours_best(rows, "mlx-community/M")
    assert best["oob"] == 0.60                       # picked by OOB, not in-sample 0.90
    assert best["trust"] == "clean"                   # CI excludes 0.5, stab >= 0.70
    assert "v_norms" in best["cell"]


def test_ours_best_flags_untrustworthy_and_handles_none():
    only_fluke = [_winrow(panel="multistep", auroc="0.90", oob_median="0.50",
                          oob_ci_lo="0.00", oob_ci_hi="1.00", winner_stability="0.22")]
    assert _ours_best(only_fluke, "mlx-community/M")["trust"] == "flagged"
    assert _ours_best([], "mlx-community/M") is None
    # winner row exists but for a different model
    assert _ours_best([_winrow(model="other/X")], "mlx-community/M") is None


def test_baseline_best_rauq_picks_max_fixed_not_aggregate():
    prof = {
        "model": {"slug": "x/m"}, "data": {"data_hash_sha256": "abc"},
        "results": {
            "1a_commit_only": {
                "per_layer": {
                    "final": {"auroc": 0.40, "direction": "lo", "auroc_signfree": 0.60},
                    "mid": {"auroc": 0.55, "direction": "hi", "auroc_signfree": 0.55},
                },
                "aggregate_max": {"auroc": 0.42, "auroc_signfree": 0.58},
            },
            "1b_prompt_recurrence": {
                "per_layer": {"final": {"auroc": 0.70, "direction": "hi",
                                        "auroc_signfree": 0.70}},
                "aggregate_max": {"auroc": 0.50, "auroc_signfree": 0.66},
            },
        },
    }
    b = _baseline_best(prof, "rauq")
    assert b["auroc"] == 0.70 and b["cell"] == "1b/final" and b["sf"] == 0.70
    assert b["agg_sf"] == 0.66            # max sign-free aggregate carried for context
    assert b["data_hash"] == "abc"


def test_baseline_best_sinkprobe():
    prof = {
        "model": {"slug": "x/s"}, "data": {"data_hash_sha256": "d"},
        "results": {
            "final": {"sink_bos": {"auroc": 0.52, "direction": "hi", "auroc_signfree": 0.52},
                      "sink_top1_vw": {"auroc": 0.81, "direction": "hi", "auroc_signfree": 0.81}},
            "mid": {"sink_bos": {"auroc": 0.40, "direction": "lo", "auroc_signfree": 0.60}},
        },
    }
    b = _baseline_best(prof, "sinkprobe")
    assert b["auroc"] == 0.81 and b["cell"] == "sink_top1_vw/final"


def test_build_head_to_head_integration(tmp_path):
    slug = "mlx-community/M"
    calib = tmp_path / "M.profile.json"
    calib.write_text(json.dumps({"task": {"data_hash_sha256": "HASH"}}))
    rdir, sdir = tmp_path / "rauq", tmp_path / "sinkprobe"
    rdir.mkdir(); sdir.mkdir()
    (rdir / "M.rauq.json").write_text(json.dumps({
        "model": {"slug": slug}, "data": {"data_hash_sha256": "HASH"},
        "results": {
            "1a_commit_only": {"per_layer": {"final": {"auroc": 0.58, "direction": "hi",
                                                       "auroc_signfree": 0.58}},
                               "aggregate_max": {"auroc": 0.50, "auroc_signfree": 0.55}},
            "1b_prompt_recurrence": {"per_layer": {}, "aggregate_max": {}},
        }}))
    (sdir / "M.sinkprobe.json").write_text(json.dumps({
        "model": {"slug": slug}, "data": {"data_hash_sha256": "HASH"},
        "results": {"final": {"sink_top1_vw": {"auroc": 0.81, "direction": "hi",
                                               "auroc_signfree": 0.81}}}}))
    all_rows = [{
        "model": slug, "panel": "v_norms", "gen_step": "1", "layer": "final",
        "metric": "js", "auroc": "0.66", "oob_median": "0.60", "oob_ci_lo": "0.55",
        "oob_ci_hi": "0.72", "winner_stability": "0.80", "is_winner": "TRUE",
        "profile_path": str(calib),
    }]
    h2h = build_head_to_head(all_rows, rdir, sdir)
    assert len(h2h) == 1
    row = h2h[0]
    assert row["model"] == "M"
    assert row["data_hash_ok"] == "TRUE"           # calib == rauq == sink
    assert row["sink_auroc_fixed"] == 0.81
    assert row["winner_fixed"] == "sinkprobe"       # 0.81 > ours OOB 0.60 > rauq 0.58
    assert row["winner_signfree"] == "sinkprobe"    # sink sf 0.81 > ours OOB 0.60 > rauq sf 0.58
    assert row["ours_trust"] == "clean"


def _summary_row(sign, **kw):
    base = dict(model="mlx-community/M", panel="v_norms", layer="final",
                metric="js", gen_step="1", auroc="0.660", sign=sign,
                oob_median="0.600", oob_ci_lo="0.55", oob_ci_hi="0.72",
                winner_stability="0.80", warnings_count="0", is_winner="TRUE")
    base.update(kw)
    return base


def test_emit_markdown_summary_empty_sign_does_not_crash(capsys):
    """Greptile PR#16 finding: r['sign'] defaults to '' when absent from the
    profile; the old f'{...:+d}' raised ValueError. Must render '—' instead."""
    rows = [
        _summary_row(""),                              # absent → "" (the regression)
        _summary_row("-1", model="mlx-community/N"),   # numeric-string sign
        _summary_row(1, model="mlx-community/P"),       # int sign
        _summary_row("", model="mlx-community/Q", is_winner="FALSE"),  # filtered out
    ]
    emit_markdown_summary(rows)                          # must NOT raise
    out = capsys.readouterr().out
    assert "—" in out and "-1" in out and "+1" in out
    assert "mlx" not in out or "M" in out  # short model name rendered


def test_emit_markdown_summary_no_winners_no_crash():
    emit_markdown_summary([_summary_row("", is_winner="FALSE")])


def test_no_oob_winner_not_contradictory(tmp_path, capsys):
    """Greptile PR#16 round-2: a calibrator winner with NO OOB must not (a)
    win on in-sample AUROC nor (b) be mislabeled 'N/A (baseline-only)'."""
    slug = "mlx-community/W"
    calib = tmp_path / "W.profile.json"
    calib.write_text(json.dumps({"task": {"data_hash_sha256": "H"}}))
    rdir, sdir = tmp_path / "r", tmp_path / "s"
    rdir.mkdir(); sdir.mkdir()
    (rdir / "W.rauq.json").write_text(json.dumps({
        "model": {"slug": slug}, "data": {"data_hash_sha256": "H"},
        "results": {"1a_commit_only": {"per_layer": {"final": {
            "auroc": 0.62, "direction": "hi", "auroc_signfree": 0.62}},
            "aggregate_max": {}}, "1b_prompt_recurrence": {"per_layer": {},
            "aggregate_max": {}}}}))
    (sdir / "W.sinkprobe.json").write_text(json.dumps({
        "model": {"slug": slug}, "data": {"data_hash_sha256": "H"},
        "results": {"final": {"sink_bos": {"auroc": 0.55, "direction": "hi",
                                           "auroc_signfree": 0.55}}}}))
    # Calibrator winner present but oob_median + CI + stability all absent.
    all_rows = [{
        "model": slug, "panel": "v_norms", "gen_step": "1", "layer": "final",
        "metric": "js", "auroc": "0.99", "oob_median": "", "oob_ci_lo": "",
        "oob_ci_hi": "", "winner_stability": "", "is_winner": "TRUE",
        "profile_path": str(calib),
    }]
    row = build_head_to_head(all_rows, rdir, sdir)[0]
    assert row["ours_oob"] is None
    assert row["ours_cell"] != _OURS_BASELINE_ONLY        # winner DOES exist
    assert row["ours_auroc"] == 0.99
    # in-sample 0.99 must NOT win EITHER column — ours excluded from both
    # races when un-OOB'd (winner_fixed AND winner_signfree symmetric).
    assert row["winner_fixed"] == "rauq"                   # 0.62 > sink 0.55
    assert row["winner_signfree"] == "rauq"                # ours excluded from sf too
    emit_head_to_head_markdown([row])
    out = capsys.readouterr().out
    assert "in-sample (no OOB)" in out
    assert _OURS_BASELINE_ONLY not in out                  # not mislabeled
