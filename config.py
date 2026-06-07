"""
Configuration for the PRI-at-commitment synthetic contradiction experiment.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable


@dataclass
class UncertaintyConfig:
    # PRI parameters
    pri_alpha: float = 0.1
    selected_prob_clamp: Tuple[float, float] = (1e-10, 1.0)
    cosine_epsilon: float = 1e-8

    # Delta-sigma / SVD numerical stability
    delta_sigma_jsd_epsilon: float = 1e-8
    svd_epsilon: float = 1e-8

    # v3 null-space direction signal — ranks at which to compute null_ratio
    # and Fisher energy ε(r) = Σ_{i≤r} σ_i² / Σ_i σ_i².
    v3_rank_values: Iterable[int] = field(default_factory=lambda: (1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64))


DEFAULT_UNCERTAINTY_CONFIG = UncertaintyConfig()

# MLX model configurations used in this experiment.
MODEL_CONFIGS = [
    {
        "name": "llama_3.2_3b",
        "path": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "display_name": "Llama 3.2 3B Instruct",
        "model_type": "llama",
    },
    {
        "name": "mistral_7b",
        "path": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "display_name": "Mistral 7B Instruct",
        "model_type": "mistral",
    },
    {
        "name": "qwen_2.5_7b",
        "path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "display_name": "Qwen 2.5 7B Instruct",
        "model_type": "qwen",
    },
    {
        "name": "gemma_3_1b",
        "path": "mlx-community/gemma-3-1b-it-4bit",
        "display_name": "Gemma 3 1B IT",
        "model_type": "gemma3",
    },
    {
        "name": "gemma_3_4b",
        "path": "mlx-community/gemma-3-4b-it-4bit",
        "display_name": "Gemma 3 4B IT",
        "model_type": "gemma3",
    },
    {
        "name": "qwen3_8b",
        "path": "mlx-community/Qwen3-8B-4bit",
        "display_name": "Qwen3 8B",
        "model_type": "qwen3",
    },
    {
        "name": "phi_3.5_mini",
        "path": "mlx-community/Phi-3.5-mini-instruct-4bit",
        "display_name": "Phi-3.5 Mini Instruct",
        "model_type": "phi3",
    },
]


# Prereq-8 primary-gate thresholds: the minimum deviation from the random
# baseline √((d−r)/d) (per-layer max |dev|) for a model to pass the null_ratio
# direction-depth gate. The random baseline itself varies per model with its
# hidden dim, but a deviation of 0.020 sits in a regime that is comparable
# across architectures in the validated set.
#
# 0.020 is the Qwen-calibrated value from the 2026-04-18 Prereq 8 ladder run
# (normed Option A rerun, rank 32, n=4/cell; max |dev from baseline 0.9955|
# = 0.0302 at Qwen's final layer). Other models inherit the default until
# their own calibration lands. Override per-model by adding an entry keyed by
# MODEL_CONFIGS[i]["model_type"].
GATE_THRESHOLD_DEFAULT: float = 0.020
GATE_THRESHOLDS: Dict[str, float] = {
    "qwen": 0.020,
}


def gate_threshold_for(model_type: str) -> float:
    """Return the Prereq-8 primary-gate threshold for a given model_type."""
    return GATE_THRESHOLDS.get(model_type, GATE_THRESHOLD_DEFAULT)
