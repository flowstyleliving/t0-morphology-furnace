"""Shared loader for ACE (Attention Commitment Estimator) t=0 sealed profiles.

Reads the 18 sealed profile JSONs (9 models x 2 datasets) into a flat list of
records consumed by the t0 paper figure + table scripts.

Sealed run: experiments/t0-sealed/2026-05-26/profiles/{anli,triviaqa}/
Pre-reg: T0_ACE_PRE_REGISTRATION_PLAN.md (frozen 2026-05-26)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SEALED_DIR = REPO / "experiments" / "t0-sealed" / "2026-05-26" / "profiles"

DATASETS = ("anli", "triviaqa")

# 9-model canonical order + display names + colors. Order = pre-reg panel order
# (Llama, Mistral family, Phi family, Qwen family, Gemma). Colors are
# tab10-compatible and consistent across figures.
MODEL_ORDER: tuple[tuple[str, str, str], ...] = (
    ("Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B", "#1f77b4"),
    ("Mistral-7B-Instruct-v0.3-4bit", "Mistral-7B", "#ff7f0e"),
    ("Mistral-Nemo-Instruct-2407-4bit", "Mistral-Nemo", "#d62728"),
    ("Phi-3.5-mini-instruct-4bit", "Phi-3.5-mini", "#9467bd"),
    ("Phi-4-mini-instruct-4bit", "Phi-4-mini", "#8c564b"),
    ("Qwen2.5-7B-Instruct-4bit", "Qwen2.5-7B", "#2ca02c"),
    ("Qwen3-1.7B-4bit", "Qwen3-1.7B", "#17becf"),
    ("Qwen3-8B-4bit", "Qwen3-8B", "#bcbd22"),
    ("gemma-3-4b-it-4bit", "Gemma-3-4B", "#e377c2"),
)

BLOCK_PREFIXES = ("final_", "mid_", "last_minus_1_")


@dataclass(frozen=True)
class CellRecord:
    """One cell within a model's 21-cell candidate panel."""
    label: str
    block_prefix: str
    metric_name: str
    auroc: float
    sign: int
    step: int


@dataclass(frozen=True)
class ProfileRecord:
    """One (model, dataset) sealed ACE profile."""
    model_filename: str
    model_short: str
    color: str
    dataset: str
    winner_label: str
    block_prefix: str
    metric_name: str
    sign: int
    layer: str
    gen_step: int
    auroc_in_sample: float
    oob_median: float
    oob_ci_lo: float
    oob_ci_hi: float
    stability: float
    warnings: tuple[str, ...]
    candidate_panel: tuple[CellRecord, ...]

    @property
    def passes_e_a1(self) -> bool:
        return self.oob_ci_lo > 0.50

    @property
    def winner_triple(self) -> tuple[str, str, int]:
        """The (block_prefix, metric_name, sign) tuple used for E_A2 transfer matching."""
        return (self.block_prefix, self.metric_name, self.sign)


def _split_metric_label(label: str) -> tuple[str, str]:
    """Split 'last_minus_1_js_no_bos' into ('last_minus_1', 'js_no_bos')."""
    for prefix in BLOCK_PREFIXES:
        if label.startswith(prefix):
            return prefix.rstrip("_"), label[len(prefix):]
    raise ValueError(f"unrecognized metric label prefix: {label!r}")


def _load_one(path: Path, model_short: str, color: str, dataset: str) -> ProfileRecord:
    p = json.loads(path.read_text())
    det = p["detector"]
    cs = p["calibration_stats"]
    winner_label = det["metric"]["label"]
    block_prefix, metric_name = _split_metric_label(winner_label)
    panel = tuple(
        CellRecord(
            label=c["rank_label"],
            block_prefix=_split_metric_label(c["rank_label"])[0],
            metric_name=_split_metric_label(c["rank_label"])[1],
            auroc=float(c["auroc"]),
            sign=int(c["sign"]),
            step=int(c["step"]),
        )
        for c in cs["candidate_panel"]
    )
    return ProfileRecord(
        model_filename=path.stem.replace(".profile", ""),
        model_short=model_short,
        color=color,
        dataset=dataset,
        winner_label=winner_label,
        block_prefix=block_prefix,
        metric_name=metric_name,
        sign=int(det["sign"]),
        layer=str(det["layer"]),
        gen_step=int(det["gen_step"]),
        auroc_in_sample=float(cs["auroc"]),
        oob_median=float(cs["oob_auroc_median"]),
        oob_ci_lo=float(cs["oob_auroc_ci_lo"]),
        oob_ci_hi=float(cs["oob_auroc_ci_hi"]),
        stability=float(cs["winner_stability"]),
        warnings=tuple(p.get("warnings", [])),
        candidate_panel=panel,
    )


def load_all() -> list[ProfileRecord]:
    """Load all 18 sealed profiles in canonical (model, dataset) order."""
    records = []
    for filename, model_short, color in MODEL_ORDER:
        for dataset in DATASETS:
            path = SEALED_DIR / dataset / f"{filename}.profile.json"
            records.append(_load_one(path, model_short, color, dataset))
    return records


def by_dataset(dataset: str) -> list[ProfileRecord]:
    return [r for r in load_all() if r.dataset == dataset]


def paired_by_model() -> list[tuple[ProfileRecord, ProfileRecord]]:
    """Return [(anli_record, triviaqa_record), ...] in canonical model order."""
    by_model: dict[str, dict[str, ProfileRecord]] = {}
    for r in load_all():
        by_model.setdefault(r.model_short, {})[r.dataset] = r
    return [(by_model[m]["anli"], by_model[m]["triviaqa"]) for _, m, _ in MODEL_ORDER]


if __name__ == "__main__":
    records = load_all()
    print(f"loaded {len(records)} sealed profiles")
    print(f"{'model':<14} {'dataset':<10} {'winner':<32} {'OOB':>6} {'CI_lo':>6} {'CI_hi':>6} {'stab':>5} {'pass':>4}")
    for r in records:
        print(
            f"{r.model_short:<14} {r.dataset:<10} {r.winner_label:<32} "
            f"{r.oob_median:>6.3f} {r.oob_ci_lo:>6.3f} {r.oob_ci_hi:>6.3f} "
            f"{r.stability:>5.2f} {str(r.passes_e_a1):>4}"
        )
