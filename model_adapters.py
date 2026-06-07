"""
Model-specific adapters for manual block-by-block forward passes.

Each adapter handles model structure introspection and hidden state extraction
without KV cache for Phase 1 correctness validation.

IMPORTANT: This implementation is designed for MLX-LM models which often require
mask, cache, and position inputs to transformer blocks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models.llama import create_attention_mask
except Exception:
    create_attention_mask = None

import hidden_state_collector
import attention_contribution


# ---------------------------------------------------------------------------
# Shared attention-mask + layer-forward helpers
#
# Consolidates the SWA mask-building pattern that was previously duplicated
# across pri_v2_mlx_pipeline.py, scripts/e22_direction_depth.py, and
# scripts/sup_spectral_band.py. Any helper that runs a manual block-by-block
# forward pass should use these three functions so the SWA handling (and
# MLX-LM layer signature variants) stay in a single place.
# ---------------------------------------------------------------------------


def build_attention_masks(core: Any, h: mx.array):
    """Build (full_causal, sliding_window_or_None) attention masks for a core model.

    `swa_mask` is None for models without Sliding-Window Attention. Per-layer
    routing is handled by `pick_layer_mask`.

    Supported SWA shapes:
      * Mistral — `core.swa_idx` set; per-layer flag `layer.use_sliding`.
      * Gemma 3 — `core.sliding_window_pattern > 1`; per-layer flag
        `layer.is_sliding` (interleaved: every pattern-th layer is global,
        rest sliding). Window size on `core.window_size` or `sliding_window`.
      * No SWA — neither attr present → `swa_mask` is None.
    """
    if create_attention_mask is None:
        raise RuntimeError(
            "mlx_lm.models.llama.create_attention_mask is unavailable; "
            "cannot build attention masks."
        )
    fa_mask = create_attention_mask(h, None)
    swa_mask = None
    if hasattr(core, "swa_idx") and getattr(core, "swa_idx") is not None:
        swa_mask = create_attention_mask(
            h, None, window_size=getattr(core, "sliding_window", None)
        )
    elif (
        hasattr(core, "sliding_window_pattern")
        and int(getattr(core, "sliding_window_pattern", 1) or 1) > 1
    ):
        window = (
            getattr(core, "window_size", None)
            or getattr(core, "sliding_window", 512)
        )
        swa_mask = create_attention_mask(h, None, window_size=int(window))
    return fa_mask, swa_mask


def pick_layer_mask(layer: Any, fa_mask: Any, swa_mask: Optional[Any]) -> Any:
    """Pick the correct attention mask for a transformer layer.

    Uses `swa_mask` when a sliding mask was built and the layer opts into it:
      * Mistral layers flag via `use_sliding`.
      * Gemma 3 layers flag via `is_sliding`.
    Falls back to `fa_mask` (full causal) otherwise.
    """
    if swa_mask is not None:
        if hasattr(layer, "use_sliding") and layer.use_sliding:
            return swa_mask
        if hasattr(layer, "is_sliding") and layer.is_sliding:
            return swa_mask
    return fa_mask


def post_embed_scale(core: Any, h: mx.array) -> mx.array:
    """Apply any architecture-specific post-embedding, pre-layer scaling.

    Gemma 3 scales the embedding output by sqrt(hidden_size) before the first
    transformer block — see `gemma3_text.Gemma3Model.__call__`. Without this
    scale, every downstream hidden state diverges from the native forward by
    a constant factor, which breaks any absolute-magnitude metric (and makes
    manual-forward logits mismatch native-forward logits).

    Detection uses `core.sliding_window_pattern` as the Gemma 3 signature —
    Mistral uses `swa_idx` (not `sliding_window_pattern`) so this won't
    false-positive on the other SWA family. Llama / Qwen 2.5 / Qwen3 / Phi-3.5
    have no `sliding_window_pattern` and return `h` unchanged.
    """
    if not hasattr(core, "sliding_window_pattern"):
        return h
    args = getattr(core, "args", None)
    hidden = int(getattr(args, "hidden_size", 0) or 0) if args is not None else 0
    if hidden <= 0:
        return h
    scale = float(hidden) ** 0.5
    return h * mx.array(scale, h.dtype)


def forward_layer(layer: Any, h: mx.array, mask: Any) -> mx.array:
    """Apply a transformer block with tolerant argument handling.

    MLX-LM layer signatures vary across model families:
      - layer(h, mask, cache=None)   (most common)
      - layer(h, mask, None)         (positional cache)
      - layer(h, mask)               (no cache arg)
    Try the most expressive form first; fall back as needed.
    """
    try:
        return layer(h, mask, cache=None)
    except TypeError:
        try:
            return layer(h, mask, None)
        except TypeError:
            return layer(h, mask)


class ModelAdapter(ABC):
    """
    Abstract base class for model-specific adapters.
    
    Adapters implement manual block-by-block forward passes to capture
    hidden states at each transformer layer during generation.
    """
    
    def __init__(
        self,
        model: Any,
        collector: hidden_state_collector.HiddenStateCollector,
        acr_collector: Optional[attention_contribution.ACRCollector] = None,
    ):
        """
        Initialize adapter with model and collector.
        
        Args:
            model: MLX model instance
            collector: Shared HiddenStateCollector instance
        """
        self.model = model
        self.collector = collector
        self.layers: List = []
        self.embed_tokens: Any = None
        self.norm: Any = None
        self.lm_head: Any = None
        self.acr_collector: Optional[attention_contribution.ACRCollector] = acr_collector
        self._acr_layer_cache: Dict[int, Optional[dict]] = {}
        self._prev_acr_input: Optional[mx.array] = None
        
        # Locate model components during initialization
        self._locate_components()
        self._validate_components()
        if self.acr_collector is not None:
            self.acr_collector.configure_for_layers(len(self.layers))
    
    @abstractmethod
    def _locate_components(self) -> None:
        """
        Introspect model structure to find components.
        
        Must set: self.layers, self.embed_tokens, self.norm, self.lm_head
        
        Raises:
            ValueError: If model structure is unknown or components not found
        """
        pass
    
    def _validate_components(self) -> None:
        """
        Validate that all required components were located.
        
        Raises:
            ValueError: If any component is missing
        """
        if self.embed_tokens is None:
            raise ValueError("embed_tokens not found in model")
        if self.norm is None:
            raise ValueError("norm layer not found in model")
        # Note: lm_head may not exist for models using weight tying
        if self.layers is None or len(self.layers) == 0:
            raise ValueError("No transformer layers found in model")
    
    def _extract_last_token_hidden(self, x: mx.array) -> mx.array:
        """
        Normalize shape to extract last-token vector.
        
        Handles various hidden state shapes:
        - [dim] -> return as-is
        - [seq, dim] -> return x[-1, :]
        - [batch, seq, dim] -> return x[0, -1, :]
        
        Args:
            x: Hidden state MLX array
            
        Returns:
            MLX array of shape [dim]
        """
        if x.ndim == 1:
            # Already [dim]
            return x
        elif x.ndim == 2:
            # [seq, dim] -> extract last token
            return x[-1, :]
        elif x.ndim == 3:
            # [batch, seq, dim] -> extract first batch, last token
            return x[0, -1, :]
        else:
            raise ValueError(f"Unexpected hidden state shape: {x.shape}")

    def _make_causal_mask(self, seq_len: int, dtype: mx.Dtype = mx.float16) -> Optional[mx.array]:
        """
        Create causal attention mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length
            dtype: Data type for mask (default float16 for Qwen/Phi-3 compatibility)
            
        Returns:
            Causal mask of shape [seq_len, seq_len] or None if not needed
        """
        # Create causal mask: upper triangular matrix of -inf
        # mask[i, j] = 0 if i >= j else -inf (can only attend to past)
        # Use float16 by default for Qwen/Phi-3 compatibility
        mask = mx.full((seq_len, seq_len), float('-inf'), dtype=dtype)
        mask = mx.triu(mask, k=1)  # Upper triangle above diagonal
        return mask

    def _make_attention_mask(self, x: mx.array, cache: Optional[Any]) -> Optional[mx.array]:
        """
        Create attention mask using MLX-LM helper when available.

        Falls back to a simple causal mask if the helper is unavailable.
        """
        if create_attention_mask is not None:
            try:
                return create_attention_mask(x, cache)
            except Exception:
                pass
        seq_len = x.shape[1] if x.ndim >= 2 else int(x.shape[0])
        return self._make_causal_mask(seq_len, dtype=x.dtype if hasattr(x, "dtype") else mx.float16)

    def _l2_norm(self, v: mx.array, epsilon: float = 1e-8) -> float:
        """
        Compute L2 norm of a vector and return as float.
        """
        return float(mx.sqrt(mx.sum(v * v) + epsilon).item())

    def _layer_supports_acr(self, layer: Any) -> bool:
        """
        Check whether a layer exposes the components needed for ACR.
        """
        return all(
            hasattr(layer, name)
            for name in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
        )

    def _forward_layer_with_optional_acr(
        self,
        layer_idx: int,
        layer: Any,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Forward a transformer layer and optionally record ACR.

        Falls back to robust layer call when ACR is disabled or unsupported.
        """
        if self.acr_collector is None:
            return self._call_layer_robust(layer, x, mask=mask, cache=cache)

        if not self._layer_supports_acr(layer):
            self.acr_collector.note_missing_layer(layer_idx, "missing_attn_components")
            return self._call_layer_robust(layer, x, mask=mask, cache=cache)

        # Residual stream entering the layer
        input_last = self._extract_last_token_hidden(x)
        if self.acr_collector.residual_mode == "delta" and self._prev_acr_input is not None:
            residual_vec = input_last - self._prev_acr_input
        else:
            residual_vec = input_last

        # Attention output before residual addition
        attn_out = layer.self_attn(layer.input_layernorm(x), mask, cache)
        attn_last = self._extract_last_token_hidden(attn_out)

        attn_norm = self._l2_norm(attn_last)
        residual_norm = self._l2_norm(residual_vec)
        self.acr_collector.record(layer_idx, attn_norm, residual_norm)
        self._prev_acr_input = input_last

        # MLP + residual (mirror layer forward)
        h = x + attn_out
        r = layer.mlp(layer.post_attention_layernorm(h))
        out = h + r
        return out
    
    def _call_layer_robust(
        self,
        layer: Any,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Call transformer layer with robust signature handling.
        
        MLX-LM layers often have different signatures:
        - layer(x, mask, cache)
        - layer(x, mask)
        - layer(x)
        
        And may return:
        - x (hidden state only)
        - (x, cache) (hidden state + new cache)
        
        Args:
            layer: Transformer layer module
            x: Input hidden states
            mask: Optional attention mask
            
        Returns:
            Output hidden states (unwrapped from tuple if necessary)
        """
        # Try different signatures in order of likelihood
        try:
            # Most common: layer(x, mask=mask, cache=cache)
            output = layer(x, mask=mask, cache=cache)
        except TypeError:
            try:
                # Fallback: layer(x, mask=mask)
                output = layer(x, mask=mask)
            except TypeError:
                try:
                    # Fallback: layer(x)
                    output = layer(x)
                except Exception as e:
                    raise RuntimeError(f"Failed to call layer with any signature: {e}")
        
        # Unwrap tuple if layer returns (hidden_states, cache)
        if isinstance(output, tuple):
            return output[0]
        
        return output
    
    @abstractmethod
    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        """
        Full prefix forward pass with hidden state collection.
        
        Performs block-by-block traversal, calling collector.record() after
        each transformer block. Without KV cache, recomputes full prefix.
        
        IMPORTANT: This method calls collector.start() to reset state for
        the current token generation step.
        
        Args:
            input_ids: MLX array of token IDs, shape [seq_len] or [1, seq_len]
            
        Returns:
            Logits for next token, MLX array of shape [vocab_size]
        """
        pass
    
    def next_token_logits(self, input_ids: mx.array) -> mx.array:
        """
        Wrapper to get next token logits.
        
        Args:
            input_ids: MLX array of token IDs
            
        Returns:
            Logits for next token, MLX array of shape [vocab_size]
        """
        return self.forward_prefix_with_collection(input_ids)


class LlamaAdapter(ModelAdapter):
    """
    Adapter for Llama 3.2 3B Instruct model.
    
    Model structure:
    - model.model.embed_tokens: Embedding layer
    - model.model.layers: List of transformer blocks
    - model.model.norm: Final RMS normalization
    - model.lm_head: Output projection to vocabulary
    """
    
    def _locate_components(self) -> None:
        """
        Locate Llama model components.
        
        Tries multiple attribute patterns to be robust.
        """
        # Try standard Llama structure first
        if hasattr(self.model, 'model'):
            self.embed_tokens = getattr(self.model.model, 'embed_tokens', None)
            self.layers = getattr(self.model.model, 'layers', None)
            self.norm = getattr(self.model.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        else:
            # Try flat structure
            self.embed_tokens = getattr(self.model, 'embed_tokens', None)
            self.layers = getattr(self.model, 'layers', None)
            self.norm = getattr(self.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
    
    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        """
        Llama-specific forward pass with hidden state collection.
        
        Matches MLX-LM's LlamaModel.__call__ exactly for logit parity.
        
        Args:
            input_ids: Token IDs, shape [seq_len] or [1, seq_len]
            
        Returns:
            Logits for next token, shape [vocab_size]
        """
        # Reset collector for this token step
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None
        
        # Normalize input shape to [batch=1, seq_len]
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]  # [seq_len] -> [1, seq_len]
        
        # Embed tokens: [1, seq_len, hidden_dim]
        x = self.embed_tokens(input_ids)
        
        # Create cache and mask (prefer MLX-LM helper; fallback to causal mask)
        cache = [None] * len(self.layers)
        mask = self._make_attention_mask(x, cache[0])
        
        # Pass through each transformer block
        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask, cache=c)
            
            # Extract and record last-token hidden state
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract last token: [1, seq_len, hidden_dim] -> [hidden_dim]
        last_token = self._extract_last_token_hidden(x)
        
        # Project to vocabulary using weight tying (as_linear)
        # MLX-LM Llama models use weight tying: embed_tokens.as_linear()
        logits = self.embed_tokens.as_linear(last_token)
        
        # Normalize logits shape to [vocab_size]
        if logits.ndim == 2:
            logits = logits.squeeze(0)
        
        return logits


class QwenAdapter(ModelAdapter):
    """
    Adapter for Qwen 2.5 7B Instruct model.
    
    Qwen models may use different attribute names than Llama.
    """
    
    def _locate_components(self) -> None:
        """
        Locate Qwen model components with fallbacks.
        """
        # Try multiple attribute name patterns
        if hasattr(self.model, 'model'):
            # Try Qwen-specific names
            self.embed_tokens = (
                getattr(self.model.model, 'embed_tokens', None) or
                getattr(self.model.model, 'tok_embeddings', None) or
                getattr(self.model.model, 'wte', None)
            )
            self.layers = getattr(self.model.model, 'layers', None)
            self.norm = getattr(self.model.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        elif hasattr(self.model, 'transformer'):
            # Alternative: transformer attribute
            self.embed_tokens = (
                getattr(self.model.transformer, 'embed_tokens', None) or
                getattr(self.model.transformer, 'wte', None)
            )
            self.layers = getattr(self.model.transformer, 'layers', None) or getattr(self.model.transformer, 'h', None)
            self.norm = getattr(self.model.transformer, 'norm', None) or getattr(self.model.transformer, 'ln_f', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        else:
            # Flat structure
            self.embed_tokens = getattr(self.model, 'embed_tokens', None) or getattr(self.model, 'tok_embeddings', None)
            self.layers = getattr(self.model, 'layers', None)
            self.norm = getattr(self.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
    
    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        """
        Qwen-specific forward pass with hidden state collection.
        Covers Qwen 2.5 (float16) and Qwen3 (bfloat16) — mask is built after
        embedding so it matches the activation dtype.
        """
        # Reset collector for this token step
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None

        # Normalize input shape
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        # Embed tokens first, THEN build mask so its dtype matches x. A
        # hardcoded float16 mask breaks scaled_dot_product_attention under
        # bfloat16 activations (Qwen3-8B-4bit); _make_attention_mask reads
        # the dtype off x.
        x = self.embed_tokens(input_ids)
        cache = [None] * len(self.layers)
        mask = self._make_attention_mask(x, cache[0])

        # Pass through transformer blocks
        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask, cache=c)

            # Record hidden state
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract last token and project
        last_token = self._extract_last_token_hidden(x)
        # 2026-05-12: Qwen3-1.7B uses TIED EMBEDDINGS (`lm_head=None` on the
        # outer model — the lm_head shares weights with `embed_tokens`).
        # Mirrors the 2026-05-11 Phi3Adapter patch. Fall back to projecting
        # through the tied input embedding when lm_head is None.
        if self.lm_head is not None:
            logits = self.lm_head(last_token)
        else:
            logits = self.embed_tokens.as_linear(last_token)

        # Normalize shape
        if logits.ndim == 2:
            logits = logits.squeeze(0)

        return logits


class Phi3Adapter(ModelAdapter):
    """
    Adapter for Phi-3 Mini Instruct model.
    
    Phi-3 models may use different attribute names.
    """
    
    def _locate_components(self) -> None:
        """
        Locate Phi-3 model components with fallbacks.
        """
        # Try multiple attribute name patterns
        if hasattr(self.model, 'model'):
            self.embed_tokens = (
                getattr(self.model.model, 'embed_tokens', None) or
                getattr(self.model.model, 'wte', None)
            )
            self.layers = getattr(self.model.model, 'layers', None) or getattr(self.model.model, 'h', None)
            self.norm = getattr(self.model.model, 'norm', None) or getattr(self.model.model, 'ln_f', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        elif hasattr(self.model, 'transformer'):
            self.embed_tokens = getattr(self.model.transformer, 'embed_tokens', None) or getattr(self.model.transformer, 'wte', None)
            self.layers = getattr(self.model.transformer, 'layers', None) or getattr(self.model.transformer, 'h', None)
            self.norm = getattr(self.model.transformer, 'norm', None) or getattr(self.model.transformer, 'ln_f', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        else:
            self.embed_tokens = getattr(self.model, 'embed_tokens', None)
            self.layers = getattr(self.model, 'layers', None)
            self.norm = getattr(self.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
    
    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        """
        Phi-3-specific forward pass with hidden state collection.
        """
        # Reset collector for this token step
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None
        
        # Normalize input shape
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        
        seq_len = input_ids.shape[1]
        mask = self._make_causal_mask(seq_len)
        
        # Embed tokens
        x = self.embed_tokens(input_ids)
        
        # Pass through transformer blocks
        for layer_idx, layer in enumerate(self.layers):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask)
            
            # Record hidden state
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract last token and project
        last_token = self._extract_last_token_hidden(x)
        # 2026-05-11: Phi-4-mini uses TIED EMBEDDINGS (`lm_head=None` on the
        # outer model — the lm_head shares weights with `embed_tokens`).
        # Fall back to `embed_tokens.as_linear()` in that case, matching
        # the LlamaAdapter convention (line 465). Without this guard the
        # smoke fails with `TypeError: 'NoneType' object is not callable`.
        if self.lm_head is not None:
            logits = self.lm_head(last_token)
        else:
            logits = self.embed_tokens.as_linear(last_token)

        # Normalize shape
        if logits.ndim == 2:
            logits = logits.squeeze(0)

        return logits


class MistralAdapter(ModelAdapter):
    """
    Adapter for Mistral-style models.

    Mistral MLX-LM models generally follow Llama-like structure.
    """

    def _locate_components(self) -> None:
        if hasattr(self.model, "model"):
            self.embed_tokens = (
                getattr(self.model.model, "embed_tokens", None)
                or getattr(self.model.model, "tok_embeddings", None)
                or getattr(self.model.model, "wte", None)
            )
            self.layers = getattr(self.model.model, "layers", None) or getattr(self.model.model, "h", None)
            self.norm = getattr(self.model.model, "norm", None) or getattr(self.model.model, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)
        else:
            self.embed_tokens = (
                getattr(self.model, "embed_tokens", None)
                or getattr(self.model, "tok_embeddings", None)
                or getattr(self.model, "wte", None)
            )
            self.layers = getattr(self.model, "layers", None) or getattr(self.model, "h", None)
            self.norm = getattr(self.model, "norm", None) or getattr(self.model, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)

    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        x = self.embed_tokens(input_ids)
        cache = [None] * len(self.layers)
        mask = self._make_attention_mask(x, cache[0])

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask, cache=c)
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)

        x = self.norm(x)
        last_token = self._extract_last_token_hidden(x)
        logits = self.lm_head(last_token)
        if logits.ndim == 2:
            logits = logits.squeeze(0)
        return logits


class GemmaAdapter(ModelAdapter):
    """
    Adapter for Gemma 3 family (1B text-only, 4B/12B/27B multimodal).

    MLX-LM dispatches Gemma 3 to one of two modules:
      - `gemma3_text` — text-only (1B only). Top-level wrapper is
        `gemma3_text.Model`, with `model.model` = `Gemma3Model` and
        `model.lm_head` directly.
      - `gemma3` — multimodal (4B/12B/27B, text + vision). Top-level is
        `gemma3.Model` which has `self.language_model = gemma3_text.Model(...)`
        and NO `self.model`. The `Gemma3Model` is at
        `model.language_model.model`, and `lm_head` at `model.language_model.lm_head`.

    This adapter handles both shapes; `_locate_components` detects which
    wrapper is in front and navigates accordingly. `_gemma3_core` is cached
    for the forward pass so mask building reads pattern/window from the
    Gemma3Model instance directly.

    Attention pattern:
      Gemma 3 interleaves sliding-window and global attention. Every
      `sliding_window_pattern`-th layer (default 6) is global; all other
      layers use a sliding-window mask with `window_size == sliding_window`
      (default 512). Mask routing here mirrors `Gemma3Model.__call__` in
      MLX-LM so hidden-state capture matches the native forward exactly.

    Separate adapters would be needed for legacy `gemma` / `gemma2` / the
    multimodal-audio variant `gemma3n` — this class is `model_type == "gemma3"`
    across both text-only (1B) and multimodal (4B+) builds.
    """

    def _locate_components(self) -> None:
        # Detect which Gemma 3 wrapper is in front and navigate to the inner
        # Gemma3Model (where embed_tokens / layers / norm / sliding attrs live).
        if hasattr(self.model, "language_model"):
            # Multimodal wrapper (gemma3.Model): model.language_model is a
            # gemma3_text.Model; its .model is the Gemma3Model we need.
            lm_outer = self.model.language_model
            gemma3_core = getattr(lm_outer, "model", None)
            lm_head_holder = lm_outer
        elif hasattr(self.model, "model"):
            # Text-only wrapper (gemma3_text.Model): model.model is Gemma3Model.
            gemma3_core = self.model.model
            lm_head_holder = self.model
        else:
            gemma3_core = self.model
            lm_head_holder = self.model

        self._gemma3_core = gemma3_core
        self.embed_tokens = getattr(gemma3_core, "embed_tokens", None) if gemma3_core is not None else None
        self.layers = getattr(gemma3_core, "layers", None) if gemma3_core is not None else None
        self.norm = getattr(gemma3_core, "norm", None) if gemma3_core is not None else None
        # lm_head is absent on weight-tied checkpoints — base _validate_components
        # tolerates None; forward_prefix_with_collection falls back to as_linear.
        self.lm_head = getattr(lm_head_holder, "lm_head", None)

    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        x = self.embed_tokens(input_ids)

        # Gemma 3 scales embeddings by sqrt(hidden_size) after the embed
        # lookup — see gemma3_text.Gemma3Model.__call__. Without this, every
        # downstream hidden state captured through the adapter is off by
        # that factor. Shared helper handles detection.
        core = self._gemma3_core if self._gemma3_core is not None else self.model
        x = post_embed_scale(core, x)

        # Gemma 3 interleaves sliding + global attention. Build both masks
        # and route per-layer exactly as Gemma3Model.__call__ does. Read
        # pattern/window from the cached Gemma3Model (handles multimodal
        # wrapper where these attrs are nested under .language_model.model).
        pattern = int(getattr(core, "sliding_window_pattern", 6))
        window = int(
            getattr(core, "window_size", getattr(core, "sliding_window", 512))
        )

        if create_attention_mask is not None:
            global_mask = create_attention_mask(x, None)
            sliding_mask = (
                create_attention_mask(x, None, window_size=window)
                if pattern > 1
                else None
            )
        else:
            seq_len = x.shape[1] if x.ndim >= 2 else int(x.shape[0])
            # Match the activation dtype — _make_causal_mask defaults to float16,
            # which breaks scaled_dot_product_attention on Gemma 3-4B under
            # bfloat16 activations. Mirrors the QwenAdapter fallback pattern.
            global_mask = self._make_causal_mask(
                seq_len, dtype=x.dtype if hasattr(x, "dtype") else mx.float16
            )
            sliding_mask = None  # no helper to build windowed mask in fallback

        for layer_idx, layer in enumerate(self.layers):
            is_global = (layer_idx % pattern) == (pattern - 1)
            mask = global_mask if (is_global or sliding_mask is None) else sliding_mask
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask)
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)

        x = self.norm(x)
        last_token = self._extract_last_token_hidden(x)

        if self.lm_head is not None:
            logits = self.lm_head(last_token)
        else:
            # Weight-tied checkpoint: project via embedding table.
            logits = self.embed_tokens.as_linear(last_token)

        if logits.ndim == 2:
            logits = logits.squeeze(0)
        return logits


class SmolLMAdapter(ModelAdapter):
    """
    Adapter for SmolLM models (Llama-like in MLX).
    """

    def _locate_components(self) -> None:
        if hasattr(self.model, "model"):
            self.embed_tokens = (
                getattr(self.model.model, "embed_tokens", None)
                or getattr(self.model.model, "tok_embeddings", None)
                or getattr(self.model.model, "wte", None)
            )
            self.layers = getattr(self.model.model, "layers", None) or getattr(self.model.model, "h", None)
            self.norm = getattr(self.model.model, "norm", None) or getattr(self.model.model, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)
        elif hasattr(self.model, "transformer"):
            self.embed_tokens = (
                getattr(self.model.transformer, "embed_tokens", None)
                or getattr(self.model.transformer, "tok_embeddings", None)
                or getattr(self.model.transformer, "wte", None)
            )
            self.layers = getattr(self.model.transformer, "layers", None) or getattr(self.model.transformer, "h", None)
            self.norm = getattr(self.model.transformer, "norm", None) or getattr(self.model.transformer, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)
        else:
            self.embed_tokens = (
                getattr(self.model, "embed_tokens", None)
                or getattr(self.model, "tok_embeddings", None)
                or getattr(self.model, "wte", None)
            )
            self.layers = getattr(self.model, "layers", None) or getattr(self.model, "h", None)
            self.norm = getattr(self.model, "norm", None) or getattr(self.model, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)

    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        x = self.embed_tokens(input_ids)
        cache = [None] * len(self.layers)
        mask = self._make_attention_mask(x, cache[0])

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask, cache=c)
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)

        x = self.norm(x)
        last_token = self._extract_last_token_hidden(x)
        logits = self.lm_head(last_token)
        if logits.ndim == 2:
            logits = logits.squeeze(0)
        return logits


class LLaVAMiniAdapter(ModelAdapter):
    """
    Adapter stub for LLaVA-mini style multimodal models.

    This codebase is text-only (mlx-lm). LLaVA models are multimodal and
    typically require mlx-vlm. We expose this adapter to make the intended
    model type explicit, but it will raise with guidance at runtime.
    """

    def _locate_components(self) -> None:
        raise RuntimeError(
            "LLaVA-mini is a multimodal model. Use mlx-vlm for loading and "
            "inference; this text-only adapter is not implemented."
        )

    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        raise RuntimeError(
            "LLaVA-mini is a multimodal model. Use mlx-vlm for loading and "
            "inference; this text-only adapter is not implemented."
        )


def create_adapter(
    model: Any,
    collector: hidden_state_collector.HiddenStateCollector,
    model_type: str = "llama",
    acr_collector: Optional[attention_contribution.ACRCollector] = None,
) -> ModelAdapter:
    """
    Factory function to create appropriate adapter for model.
    
    Args:
        model: MLX model instance
        collector: HiddenStateCollector instance
        model_type: One of "llama", "qwen", "phi3"
        
    Returns:
        Appropriate ModelAdapter subclass instance
        
    Raises:
        ValueError: If model_type is unknown
    """
    adapters = {
        "llama": LlamaAdapter,
        "qwen": QwenAdapter,
        "qwen3": QwenAdapter,  # Qwen3 shares Qwen2's component layout; same adapter.
        "phi3": Phi3Adapter,
        "mistral": MistralAdapter,
        "gemma3": GemmaAdapter,
        "smollm": SmolLMAdapter,
        "llava": LLaVAMiniAdapter
    }
    
    if model_type.lower() not in adapters:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of: {list(adapters.keys())}"
        )
    
    adapter_class = adapters[model_type.lower()]
    return adapter_class(model, collector, acr_collector=acr_collector)
