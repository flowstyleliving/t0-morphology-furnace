"""I/O plugins for the PRI v3 pipeline: parser tiers + prompt strategies.

Two orthogonal concerns, one file:

  1. PARSER TIERS — model OUTPUT → YES/NO label.
     Ordered list of `(text) -> Optional["YES"|"NO"]` callables. `parse_yes_no`
     walks the tiers in order; first non-None result wins. Add a new tier by
     appending to `DEFAULT_TIERS` (or pass your own list).

  2. PROMPT STRATEGIES — puzzle text → wrapped input prompt for the model.
     Per-model dict of `(puzzle, tokenizer) -> str` callables. Default is
     `raw_passthrough` (preserves the v3.2 sealed protocol for the original
     six models). Newer chat-tuned models (Mistral-Nemo, Gemma-3-1B, Dolphin)
     need `apply_chat_template` to "check in at the receptionist desk" — see
     wiki/learn/chat-template-gap-eli12.md for the metaphor + evidence.

Both registries are pure-Python with no MLX dependency, so they can be
tested in isolation. See `scripts/test_io_plugins.py`.

History
-------
2026-05-11 — Created after the n=20 smoke-output collection revealed three
distinct failure modes for Mistral-Nemo / Gemma-3-1B / Dolphin, all rooted
in the pipeline passing raw prompts to `mlx_generate` instead of wrapping
with the tokenizer's chat template. The Tier-0 first-word parser handles
the bare YES/NO outputs that newer chat-tuned models produce after the
chat-template fix (e.g., Mistral-Nemo emits just `'YES'` or `'NO.'`).
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# PARSER TIERS — model output → "YES" / "NO" / None
# ─────────────────────────────────────────────────────────────────────────────


ParserTier = Callable[[str], Optional[str]]

_ROLE_HEADER_TOKENS = {"ASSISTANT", "USER", "SYSTEM"}
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")


def _strip_specials(text: str) -> str:
    """Strip `<|special|>` tokens so they don't break tier-0 first-word match."""
    return _SPECIAL_TOKEN_RE.sub(" ", str(text).strip())


#: Prefixes that signal an emphatic closing commitment. Take precedence over
#: plain `Answer:` (Tier 1) because models use these to override earlier
#: hedged `Answer:` statements inside chain-of-thought. Extend by appending
#: a new prefix (lowercase) — no parser code changes needed. Matched
#: case-insensitively via re.IGNORECASE.
EMPHATIC_CLOSING_PREFIXES: List[str] = [
    "final answer",
    "conclusion",
    # Future candidates if observed on real model outputs:
    #   "verdict", "decision", "final verdict", "my answer",
    #   "the answer is", "in conclusion",
]


def tier_emphatic_closing(text: str) -> Optional[str]:
    """Tier 0.5 — explicit closing-commitment prefixes (`Final Answer:`,
    `Conclusion:`, etc., as enumerated in `EMPHATIC_CLOSING_PREFIXES`).

    These are the model's emphatic final commitment, often overriding
    earlier hedged `Answer:` statements inside chain-of-thought. Takes
    precedence over Tier 1 so that `"Answer: NO. ... Final Answer: YES."`
    returns YES (the final commitment) rather than NO (the earlier hedge).
    Take the LAST occurrence if multiple matches appear.

    Tolerates `: ` / `:` / `=` / ` is ` between prefix and answer; tolerates
    optional surrounding quotes. Case-insensitive on both prefix and answer
    (returns uppercase YES/NO).

    Observed on: Gemma-3-1B (`'Final Answer: YES Explanation: ...'`) and
    other reasoning-tuned models that wrap their conclusion in "Final ..." /
    "Conclusion: ...".
    """
    s = _strip_specials(text)
    prefix_alt = "|".join(re.escape(p) for p in EMPHATIC_CLOSING_PREFIXES)
    pattern = rf"(?:{prefix_alt})\s*(?:is\s+)?[:=]?\s*[\"']?(YES|NO)\b"
    last = None
    for m in re.finditer(pattern, s, re.IGNORECASE):
        last = m.group(1).upper()
    return last


def tier_answer_prefix(text: str) -> Optional[str]:
    """Tier 1 — explicit `Answer: YES|NO` after a line boundary, period,
    or colon. Strongest signal AFTER `Final Answer:` (Tier 0.5). Note this
    regex's anchor `(?:(?:^|\n)\s*|[\.\:]\s+)` deliberately does NOT match
    `'Final Answer:'` (the "answer" there is preceded by a space, not a
    period/colon/newline) — that case is owned by `tier_final_answer`.
    Take the LAST occurrence if multiple `Answer: X` statements appear."""
    s = _strip_specials(text)
    last = None
    for m in re.finditer(
        r"(?:(?:^|\n)\s*|[\.\:]\s+)answer\s*[:=]?\s*[\"']?(YES|NO)\b",
        s,
        re.IGNORECASE,
    ):
        last = m.group(1).upper()
    return last


def tier_bare_first_word(text: str) -> Optional[str]:
    """Tier 0 — first significant alphabetic token is YES or NO.

    Catches the format newer chat-tuned models produce after the chat-template
    fix: Mistral-Nemo emits `'YES'` or `'NO.'`, Gemma-3-1B emits
    `'YES<end_of_turn>...'`. Skips role-header tokens (ASSISTANT/USER/SYSTEM).
    Returns None if the first non-header token is anything else (e.g.
    `'Answer:'`, `'Analysis:'`, `'Let'`), letting later tiers handle.

    Placed AFTER `tier_answer_prefix` in `DEFAULT_TIERS` because `'Answer: YES'`
    is a stronger signal than bare `YES` when both are present.
    """
    s = _strip_specials(text)
    for m in re.finditer(r"[A-Za-z]+", s):
        tok = m.group(0).upper()
        if tok in _ROLE_HEADER_TOKENS:
            continue
        if tok in {"YES", "NO"}:
            return tok
        # First non-header token is something else — don't fire.
        return None
    return None


def tier_trailing_line(text: str) -> Optional[str]:
    """Tier 2 — the last non-empty line is just YES/NO (with trivial
    punctuation/markdown padding). Catches direct-answer models whose
    final line is `'YES.'` even if mid-CoT mentioned `'NO'`."""
    s = _strip_specials(text)
    for ln in reversed([l.strip() for l in s.splitlines() if l.strip()]):
        m = re.fullmatch(
            r"[\s\*\"'\.\!\?\-\:\(\)]*(YES|NO)[\s\*\"'\.\!\?\-\:\(\)]*",
            ln,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()
    return None


def tier_last_match_anywhere(text: str) -> Optional[str]:
    """Tier 3 — last YES/NO word anywhere in the output. Safety net for
    outputs that don't match earlier tiers (e.g., reasoning chains that
    end with `'...so the answer must be NO'`)."""
    s = _strip_specials(text)
    last = None
    for m in re.finditer(r"[A-Za-z]+", s):
        tok = m.group(0).upper()
        if tok in {"YES", "NO"}:
            last = tok
    return last


# Order matters: stronger signals first.
#   Tier 0.5 (final_answer) — most emphatic; closing commitment.
#   Tier 1   (answer_prefix) — anchored "Answer: X" elsewhere in output.
#   Tier 0   (bare_first_word) — Mistral-Nemo-style bare YES/NO.
#   Tier 2   (trailing_line) — last non-empty line is just YES/NO.
#   Tier 3   (last_match) — last YES/NO anywhere (safety net).
DEFAULT_TIERS: List[ParserTier] = [
    tier_emphatic_closing,
    tier_answer_prefix,
    tier_bare_first_word,
    tier_trailing_line,
    tier_last_match_anywhere,
]


def parse_yes_no(
    text: Optional[str],
    tiers: Optional[List[ParserTier]] = None,
) -> Optional[str]:
    """Walk parser tiers in order; return first non-None result.

    Returns None only if every tier abstains — caller decides what to do
    with that (typically a final substring fallback or UNPARSEABLE label).
    Pass `tiers=` to use a custom tier list (e.g. exclude Tier 3 for strict
    sealed-protocol parity).
    """
    if text is None:
        return None
    use_tiers = tiers if tiers is not None else DEFAULT_TIERS
    for tier in use_tiers:
        result = tier(text)
        if result is not None:
            return result
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT STRATEGIES — puzzle text + tokenizer → wrapped input prompt
# ─────────────────────────────────────────────────────────────────────────────


PromptStrategy = Callable[[str, object], str]  # (puzzle, tokenizer) -> wrapped


def raw_passthrough(puzzle: str, tokenizer: object) -> str:
    """Default strategy — pass the puzzle through unchanged.

    Used by older chat-tuned models (Llama 3.2-3B, Mistral 7B v0.3, Qwen 2.5,
    Qwen 3, Phi-3.5, Gemma 3-4B, Llama 3.1-8B, Phi-4-mini) which all
    tolerate raw instruction prompts because they were trained on enough
    raw text alongside chat data. PRESERVES the v3.2 sealed protocol —
    AUROC numbers for these models are stable across the raw-prompt regime.
    """
    return puzzle


def apply_chat_template(puzzle: str, tokenizer: object) -> str:
    """Strategy for newer chat-tuned models that require the tokenizer's
    chat-template wrap (`[INST]...[/INST]` for Mistral-family, ChatML
    `<|im_start|>user\\n...\\n<|im_end|>\\n<|im_start|>assistant\\n` for
    Dolphin-family, etc.). Without this wrap they produce empty output
    or chain-of-thought that never reaches a YES/NO.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": puzzle}],
        tokenize=False,
        add_generation_prompt=True,
    )


# Per-model strategy dispatch. Default = raw_passthrough; opt-in per slug.
# Add a new model by appending one line here after smoke-testing.
#
# Maintenance note: changing this dict can affect AUROC numbers for the
# listed models. Don't move a model OUT of this dict without re-running
# its v3.2 baseline.
PROMPT_STRATEGY_BY_MODEL: Dict[str, PromptStrategy] = {
    # 2026-05-11 smoke confirmed: these emit empty / CoT-overflow / garbled
    # output on raw prompts and clean YES/NO on chat-template-wrapped input.
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit": apply_chat_template,
    "mlx-community/gemma-3-1b-it-4bit":              apply_chat_template,
    "mlx-community/dolphin-2.9.3-mistral-nemo-12b-4bit": apply_chat_template,
    # NOTE: Phi-4-mini-instruct-4bit is NOT in this dict — it works fine on
    # raw prompts (smoke 4/4 after the Phi3Adapter tied-embedding patch).
}


def get_prompt_strategy(model_slug: str) -> PromptStrategy:
    """Look up the prompt strategy for a model slug. Default = raw_passthrough.

    Use this at every call site that passes a prompt to `mlx_generate`:

        strategy = get_prompt_strategy(model_name)
        input_prompt = strategy(puzzle, tokenizer)
        out = mlx_generate(model, tokenizer, prompt=input_prompt, ...)
    """
    return PROMPT_STRATEGY_BY_MODEL.get(model_slug, raw_passthrough)
