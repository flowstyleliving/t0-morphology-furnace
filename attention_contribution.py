"""
Attention Contribution Ratio (ACR) collection utilities.

ACR captures the per-layer attention contribution before residual addition,
normalized by the residual stream norm, and aggregates across depth.
"""

from typing import Dict, Optional, Tuple


def get_acr_layer_range(
    num_layers: int,
    start_pct: float = 0.25,
    end_pct: float = 0.75,
) -> Tuple[int, int]:
    """
    Compute a proportional layer range for mid-layer ACR.

    Args:
        num_layers: Total number of transformer layers
        start_pct: Start fraction of depth (default 0.25)
        end_pct: End fraction of depth (default 0.75)

    Returns:
        Tuple (start, end) with end exclusive.
    """
    start = int(num_layers * start_pct)
    end = int(num_layers * end_pct)
    if end <= start:
        start = 0
        end = num_layers
    return start, end


class ACRCollector:
    """
    Collector for per-layer attention/residual norms at each generation step.

    Call start() once per token step (before the forward pass), then record()
    per layer during the forward traversal. Use compute_acr_mean() after the
    forward pass to aggregate across layers.
    """

    def __init__(
        self,
        min_attn_norm: float = 1e-4,
        layer_range: Optional[Tuple[int, int]] = None,
        residual_mode: str = "raw",
        start_pct: float = 0.25,
        end_pct: float = 0.75,
    ) -> None:
        """
        Args:
            min_attn_norm: Minimum attention norm to include a layer in ACR.
            layer_range: Optional (start, end) slice over layers (end exclusive).
            residual_mode: "raw" for ||h_{l-1}||, "delta" for ||h_l-1 - h_l-2||.
        """
        self.min_attn_norm = float(min_attn_norm)
        self.layer_range = layer_range
        self.residual_mode = residual_mode
        self.start_pct = float(start_pct)
        self.end_pct = float(end_pct)
        self.attn_norms: Dict[int, float] = {}
        self.residual_norms: Dict[int, float] = {}
        self.missing_layers: Dict[int, str] = {}

    def configure_for_layers(self, num_layers: int) -> None:
        """
        Configure proportional mid-layer range if not explicitly set.
        """
        if self.layer_range is None:
            self.layer_range = get_acr_layer_range(
                num_layers=num_layers,
                start_pct=self.start_pct,
                end_pct=self.end_pct,
            )

    def start(self) -> None:
        """Reset per-step buffers."""
        self.attn_norms.clear()
        self.residual_norms.clear()

    def record(self, layer_idx: int, attn_norm: float, residual_norm: float) -> None:
        """Record per-layer norms for the current step."""
        self.attn_norms[int(layer_idx)] = float(attn_norm)
        self.residual_norms[int(layer_idx)] = float(residual_norm)

    def note_missing_layer(self, layer_idx: int, reason: str) -> None:
        """Record a missing-layer diagnostic (non-fatal)."""
        self.missing_layers[int(layer_idx)] = reason

    def num_layers_recorded(self) -> int:
        return len(self.attn_norms)

    def compute_acr_mean(self, layer_range: Optional[Tuple[int, int]] = None) -> float:
        """
        Compute mean ACR across recorded layers for the current step.

        Returns:
            Mean ACR in [0, 1], or 0.0 if no valid layers.
        """
        acr_by_layer = self.compute_acr_by_layer(layer_range=layer_range)
        if not acr_by_layer:
            return 0.0
        return float(sum(acr_by_layer.values()) / len(acr_by_layer))

    def compute_acr_by_layer(
        self, layer_range: Optional[Tuple[int, int]] = None
    ) -> Dict[int, float]:
        """
        Compute per-layer ACR for the current step.

        Returns:
            Dict[layer_idx] = ACR value
        """
        if not self.attn_norms:
            return {}

        if layer_range is None:
            layer_range = self.layer_range

        acr_values: Dict[int, float] = {}
        for layer_idx in sorted(self.attn_norms.keys()):
            if layer_range is not None:
                start, end = layer_range
                if layer_idx < start or layer_idx >= end:
                    continue

            attn_norm = self.attn_norms.get(layer_idx, 0.0)
            residual_norm = self.residual_norms.get(layer_idx, 0.0)

            if attn_norm < self.min_attn_norm:
                continue

            denom = attn_norm + residual_norm
            if denom <= 1e-8:
                continue

            acr_values[layer_idx] = attn_norm / denom

        return acr_values

    def __repr__(self) -> str:
        return (
            f"ACRCollector(layers={len(self.attn_norms)}, min_attn_norm={self.min_attn_norm}, "
            f"layer_range={self.layer_range}, residual_mode={self.residual_mode})"
        )
