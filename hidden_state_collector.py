"""
Hidden state collector for capturing transformer block outputs during generation.

The collector follows a simple lifecycle:
1. start() - Reset state for new token generation
2. record() - Called after each transformer block to capture hidden state
3. get_all_blocks() - Retrieve all captured states for dispersion computation
"""

from typing import Dict, List
import mlx.core as mx


class HiddenStateCollector:
    """
    Stateful collector for hidden states during token generation.
    
    Must be reset via start() at each token boundary to ensure clean capture
    of layer-wise hidden states for the current token step.
    """
    
    def __init__(self):
        """Initialize empty block storage."""
        self._blocks: Dict[int, mx.array] = {}
    
    def start(self) -> None:
        """
        Reset internal buffer for new token generation.
        
        Call this at the beginning of each token generation step before
        running the forward pass through transformer blocks.
        """
        self._blocks.clear()
    
    def record(self, layer_idx: int, hidden_vector: mx.array) -> None:
        """
        Store last-token hidden vector from transformer block.
        
        Args:
            layer_idx: 0-indexed transformer block number
            hidden_vector: MLX array of shape [dim] representing the last-token
                          hidden state from this block
                          
        Raises:
            ValueError: If hidden_vector is not 1-dimensional
        """
        # Validate shape
        if hidden_vector.ndim != 1:
            raise ValueError(
                f"Expected 1D hidden vector [dim], got shape {hidden_vector.shape}. "
                f"Use _extract_last_token_hidden() in adapter to normalize shape."
            )
        
        # Store the hidden vector
        self._blocks[layer_idx] = hidden_vector
    
    def get_all_blocks(self) -> List[mx.array]:
        """
        Return all captured hidden vectors in layer order.
        
        Returns:
            List of MLX arrays, each of shape [dim], ordered by layer index.
            Used for computing Δσ dispersion metric.
            
        Raises:
            ValueError: If no blocks have been recorded
        """
        if not self._blocks:
            raise ValueError(
                "No blocks recorded. Ensure record() is called during forward pass."
            )
        
        # Sort by layer index and return vectors
        sorted_layers = sorted(self._blocks.keys())
        return [self._blocks[idx] for idx in sorted_layers]
    
    def num_blocks_recorded(self) -> int:
        """
        Return the number of blocks currently recorded.
        
        Useful for debugging and validation.
        """
        return len(self._blocks)
    
    def __repr__(self) -> str:
        return f"HiddenStateCollector(num_blocks={len(self._blocks)})"