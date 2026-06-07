"""
Synthetic logic puzzle dataset generation for contradiction-injection experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, List


CELL_SHORT_NO = "short_no_contradiction"
CELL_SHORT_YES = "short_with_contradiction"
CELL_LONG_NO = "long_no_contradiction"
CELL_LONG_YES = "long_with_contradiction"

ALL_CELLS = (
    CELL_SHORT_NO,
    CELL_SHORT_YES,
    CELL_LONG_NO,
    CELL_LONG_YES,
)

PROMPT_TEMPLATE_BASELINE = "baseline"
PROMPT_TEMPLATE_WORKED_EXAMPLE_V2 = "worked_example_v2"
PROMPT_TEMPLATE_VARIANTS = {
    PROMPT_TEMPLATE_BASELINE,
    PROMPT_TEMPLATE_WORKED_EXAMPLE_V2,
}

WORKED_EXAMPLE_V2 = (
    "Worked Example:\n"
    "Answer with ONLY the single property word. No explanation.\n\n"
    "Premises:\n"
    "1. All round things are smooth.\n"
    "2. Blex is a round thing.\n"
    "Question: What texture is Blex?\n"
    "Answer: smooth\n"
    "---"
)


@dataclass
class SyntheticLogicConfig:
    n_per_cell: int = 200
    seed: int = 42
    short_chain_steps: int = 2
    long_chain_steps: int = 5
    prompt_template_variant: str = PROMPT_TEMPLATE_BASELINE
    include_worked_example: bool | None = None

    def validate(self) -> "SyntheticLogicConfig":
        if self.n_per_cell < 1:
            raise ValueError("n_per_cell must be >= 1")
        if self.short_chain_steps < 1:
            raise ValueError("short_chain_steps must be >= 1")
        if self.long_chain_steps <= self.short_chain_steps:
            raise ValueError("long_chain_steps must be > short_chain_steps")
        if self.prompt_template_variant not in PROMPT_TEMPLATE_VARIANTS:
            raise ValueError(
                f"prompt_template_variant must be one of {sorted(PROMPT_TEMPLATE_VARIANTS)}"
            )
        return self

    def resolved_include_worked_example(self) -> bool:
        if self.include_worked_example is not None:
            return bool(self.include_worked_example)
        return self.prompt_template_variant == PROMPT_TEMPLATE_WORKED_EXAMPLE_V2


@dataclass
class SyntheticLogicSample:
    sample_id: str
    prompt: str
    cell: str
    chain_steps: int
    has_contradiction: bool
    anchor_char_index: int
    target_property: str
    subject: str
    injected_statement: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SUBJECT_POOL = [
    "Flib",
    "Nara",
    "Tovin",
    "Rell",
    "Sema",
    "Varn",
    "Kiro",
    "Mela",
    "Drax",
    "Luni",
    "Pavo",
    "Rima",
]

_TERM_POOL = [
    "glorp",
    "blen",
    "trune",
    "vask",
    "mordin",
    "krel",
    "zenith",
    "prax",
    "nuvin",
    "seral",
    "thalen",
    "quorin",
    "dravan",
    "melta",
    "sorin",
    "valen",
    "torin",
    "dorin",
    "virel",
    "jorin",
]

_PROPERTY_RELATIONS = [
    ("color", ["blue", "red", "green", "yellow", "black", "white"]),
    ("texture", ["smooth", "rough", "soft", "hard", "sticky", "dry"]),
    ("shape", ["round", "square", "triangular", "flat", "curved", "straight"]),
    ("material", ["metal", "wooden", "plastic", "stone", "glass", "paper"]),
]


def _pick_unique_terms(rng: random.Random, n: int) -> List[str]:
    if n > len(_TERM_POOL):
        raise ValueError("TERM_POOL too small for requested chain length")
    return rng.sample(_TERM_POOL, n)


def _build_prompt(
    rng: random.Random,
    chain_steps: int,
    has_contradiction: bool,
    sample_id: str,
    prompt_template_variant: str = PROMPT_TEMPLATE_BASELINE,
    include_worked_example: bool = False,
) -> SyntheticLogicSample:
    answer_mode = "yes_no"
    relation_label = None
    if prompt_template_variant == PROMPT_TEMPLATE_WORKED_EXAMPLE_V2:
        # Use natural-language properties for the final target to support
        # single-word extraction (e.g., blue, smooth) in the pilot.
        relation_label, property_pool = rng.choice(_PROPERTY_RELATIONS)
        target = rng.choice(property_pool)
        terms = _pick_unique_terms(rng, chain_steps)
        answer_mode = "property_word"
    else:
        terms = _pick_unique_terms(rng, chain_steps + 1)
        target = terms[-1]
    subject_base = rng.choice(_SUBJECT_POOL)
    subject = f"{subject_base}{sample_id.split('_')[-1]}"

    premise_lines: List[str] = []
    if answer_mode == "property_word":
        if chain_steps > 1:
            for i in range(chain_steps - 1):
                premise_lines.append(f"{i + 1}. All {terms[i]}s are {terms[i + 1]}s.")
            premise_lines.append(
                f"{chain_steps}. All {terms[-1]}s are {target}."
            )
        else:
            premise_lines.append(f"1. All {terms[0]}s are {target}.")
    else:
        for i in range(chain_steps):
            premise_lines.append(f"{i + 1}. All {terms[i]}s are {terms[i + 1]}s.")

    subject_line = f"{chain_steps + 1}. {subject} is a {terms[0]}."
    if has_contradiction:
        injected_statement = f"{chain_steps + 2}. {subject} is not a {target}."
    else:
        injected_statement = f"{chain_steps + 2}. {subject} is a {target}."

    if answer_mode == "property_word":
        question = f"Question: What {relation_label} is {subject}?\nAnswer:"
    else:
        question = (
            f"Question: Is {subject} a {target}? "
            "Answer with only YES or NO."
        )

    intro = (
        "Instruction: Read the premises and answer the final question from those premises."
    )
    pre_anchor = "\n".join([intro, "", "Premises:", *premise_lines, subject_line]) + "\n"
    worked_example_prefix = ""
    if include_worked_example:
        worked_example_prefix = WORKED_EXAMPLE_V2 + "\nNow solve:\n"

    anchor_char_index = len(worked_example_prefix) + len(pre_anchor)
    prompt = worked_example_prefix + pre_anchor + injected_statement + "\n" + question

    # Derive the cell from the sample id prefix so custom pilot settings (e.g., short=1, long=2)
    # do not get misclassified by a hardcoded chain-length threshold.
    cell = sample_id.rsplit("_", 1)[0]
    if cell not in ALL_CELLS:
        raise ValueError(f"Sample id must start with a valid cell prefix: {sample_id}")

    return SyntheticLogicSample(
        sample_id=sample_id,
        prompt=prompt,
        cell=cell,
        chain_steps=chain_steps,
        has_contradiction=has_contradiction,
        anchor_char_index=anchor_char_index,
        target_property=target,
        subject=subject,
        injected_statement=injected_statement,
        metadata={
            "terms": terms,
            "subject_line": subject_line,
            "question": question,
            "prompt_template_variant": prompt_template_variant,
            "include_worked_example": bool(include_worked_example),
            "answer_mode": answer_mode,
            "relation_label": relation_label,
        },
    )


def generate_synthetic_logic_dataset(
    cfg: SyntheticLogicConfig | None = None,
) -> List[Dict[str, Any]]:
    config = (cfg or SyntheticLogicConfig()).validate()
    rng = random.Random(config.seed)
    samples: List[SyntheticLogicSample] = []
    include_worked_example = config.resolved_include_worked_example()

    for i in range(config.n_per_cell):
        samples.append(
            _build_prompt(
                rng=rng,
                chain_steps=config.short_chain_steps,
                has_contradiction=False,
                sample_id=f"{CELL_SHORT_NO}_{i:04d}",
                prompt_template_variant=config.prompt_template_variant,
                include_worked_example=include_worked_example,
            )
        )
        samples.append(
            _build_prompt(
                rng=rng,
                chain_steps=config.short_chain_steps,
                has_contradiction=True,
                sample_id=f"{CELL_SHORT_YES}_{i:04d}",
                prompt_template_variant=config.prompt_template_variant,
                include_worked_example=include_worked_example,
            )
        )
        samples.append(
            _build_prompt(
                rng=rng,
                chain_steps=config.long_chain_steps,
                has_contradiction=False,
                sample_id=f"{CELL_LONG_NO}_{i:04d}",
                prompt_template_variant=config.prompt_template_variant,
                include_worked_example=include_worked_example,
            )
        )
        samples.append(
            _build_prompt(
                rng=rng,
                chain_steps=config.long_chain_steps,
                has_contradiction=True,
                sample_id=f"{CELL_LONG_YES}_{i:04d}",
                prompt_template_variant=config.prompt_template_variant,
                include_worked_example=include_worked_example,
            )
        )

    rng.shuffle(samples)
    return [s.to_dict() for s in samples]


def save_split(data: List[Dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_split(input_path: str | Path) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
