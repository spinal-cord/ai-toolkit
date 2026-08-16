"""
Tanh Softcapping for Attention - score modification inspired by Gemma2 and Grok-1.

Applies: soft_cap * tanh(score / soft_cap) to attention scores before softmax.
This prevents attention scores from becoming too extreme, improving training stability
by avoiding overly sharp attention distributions.

Usage:
    from toolkit.util.attention_softcapping import create_tanh_softcap_score_mod, apply_attention_with_softcap

    # Create score modification function
    score_mod = create_tanh_softcap_score_mod(soft_cap=30.0)

    # Use with flex_attention (inside attention processor)
    output = apply_attention_with_softcap(query, key, value, score_mod=score_mod, attn_mask=None)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable

# Check if flex_attention is available (PyTorch 2.5+)
# Note: In some builds it's not exported from torch.nn.attention but available via direct import
_flex_attention_func = None
try:
    from torch.nn.attention.flex_attention import flex_attention as _fa
    _flex_attention_func = _fa
except ImportError:
    pass

HAS_FLEX_ATTENTION = _flex_attention_func is not None

def _get_flex_attention():
    """Get flex_attention function, raising ImportError if not available."""
    if not HAS_FLEX_ATTENTION:
        raise ImportError(
            "Tanh softcapping requires PyTorch 2.5+ with flex_attention support. "
            "Please upgrade PyTorch or disable softcapping."
        )
    return _flex_attention_func


def create_tanh_softcap_score_mod(soft_cap: float = 30.0) -> Callable:
    """
    Create a tanh softcapping score modification function for flex_attention.

    Args:
        soft_cap: The soft cap value. Larger = gentler capping.
                  Recommended: 20-40 for most models.

    Returns:
        A score_mod function compatible with torch.nn.attention.flex_attention.

    Example:
        score_mod = create_tanh_softcap_score_mod(soft_cap=30.0)
        output = flex_attention(query, key, value, score_mod=score_mod)
    """
    def tanh_softcap(score, b, h, q_idx, kv_idx):
        # Apply: soft_cap * tanh(score / soft_cap)
        return soft_cap * torch.tanh(score / soft_cap)

    tanh_softcap.__name__ = f"tanh_softcap_{soft_cap}"
    return tanh_softcap


def _sdpa_mask_to_mask_mod(attn_mask: torch.Tensor, target_device: torch.device, is_causal: bool = False) -> Callable:
    """
    Convert an SDPA-style attention mask to a flex_attention mask_mod function.

    This is used to create a BlockMask that allows flex_attention to skip
    computation for fully-masked blocks.

    IMPORTANT: mask_mod must use pure tensor operations (no Python control flow)
    because flex_attention uses vmap internally to vectorize the mask computation.

    SDPA mask value conventions:
    - Boolean: True = UNMASKED (keep/can attend), False = MASKED (drop/cannot attend)
    - Float: 0 = UNMASKED (keep/can attend), -inf = MASKED (drop/cannot attend)

    Args:
        attn_mask: Attention mask tensor
        target_device: Device to ensure mask is on (prevents device mismatch)
        is_causal: If True, also apply causal masking

    Returns:
        mask_mod function compatible with create_block_mask
    """
    # Ensure mask is on the correct device (fixes device mismatch issue)
    attn_mask = attn_mask.to(device=target_device, non_blocking=True)

    # Normalize mask: convert to boolean where True = masked (CANNOT attend)
    # CRITICAL: Handle boolean and float masks differently!
    if attn_mask.dtype == torch.bool:
        # Boolean mask: True = can attend, False = cannot attend
        # So is_masked = ~attn_mask (False becomes True = masked)
        is_masked = ~attn_mask
    else:
        # Float mask: -inf = cannot attend, 0 = can attend
        is_masked = torch.isneginf(attn_mask.to(torch.float32))

    # Optimize: use Python if/else to avoid creating tensor inside vmapped function
    if is_causal:
        def mask_mod_causal(b, h, q_idx, kv_idx):
            # Index into the mask tensor using tensor ops
            if is_masked.ndim == 4:
                is_pos_masked = is_masked[b, h, q_idx, kv_idx]
            elif is_masked.ndim == 3:
                is_pos_masked = is_masked[b, q_idx, kv_idx]
            else:
                is_pos_masked = is_masked[q_idx, kv_idx]

            # Position is valid if: causal check passes AND not masked
            return (q_idx >= kv_idx) & ~is_pos_masked
        return mask_mod_causal
    else:
        def mask_mod_no_causal(b, h, q_idx, kv_idx):
            # Index into the mask tensor using tensor ops
            if is_masked.ndim == 4:
                is_pos_masked = is_masked[b, h, q_idx, kv_idx]
            elif is_masked.ndim == 3:
                is_pos_masked = is_masked[b, q_idx, kv_idx]
            else:
                is_pos_masked = is_masked[q_idx, kv_idx]

            # Position is valid if not masked
            return ~is_pos_masked
        return mask_mod_no_causal


def apply_attention_with_softcap(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Optional[Callable] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply attention with optional tanh softcapping.

    Uses flex_attention's proper architecture:
    - BlockMask for structural masking (padding, causal) - allows skipping masked blocks
    - score_mod for numerical transformation (softcapping) only

    Falls back to standard scaled_dot_product_attention if:
    - flex_attention is not available
    - score_mod is None

    Args:
        query: (B, H, S, D) query tensor
        key: (B, H, S, D) key tensor
        value: (B, H, S, D) value tensor
        score_mod: Optional score modification function (from create_tanh_softcap_score_mod)
        attn_mask: Optional attention mask - now properly supported via BlockMask!
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        softmax_scale: Optional softmax scale (None uses default 1/sqrt(D))

    Returns:
        Attention output tensor
    """
    # Fall back to standard attention if softcapping not requested or not available
    if score_mod is None or not HAS_FLEX_ATTENTION:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=softmax_scale,
        )

    # Use flex_attention for softcapping
    flex_attn = _get_flex_attention()
    from torch.nn.attention.flex_attention import create_block_mask

    # Extract dimensions
    B, H, Q_LEN, _ = query.shape
    KV_LEN = key.shape[2]
    device = query.device

    # Create BlockMask if we have a mask or causal attention
    block_mask = None
    if attn_mask is not None or is_causal:
        if attn_mask is not None:
            mask_mod = _sdpa_mask_to_mask_mod(attn_mask, device, is_causal)
        else:
            # Pure causal mask
            def causal_mask_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            mask_mod = causal_mask_mod

        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
        )

    return flex_attn(
        query, key, value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=softmax_scale,
    )


class TanhSoftcapAttentionWrapper:
    """
    Wrapper that adds tanh softcapping to an attention processor.

    Usage:
        processor = WanAttnProcessor2_0()
        wrapped = TanhSoftcapAttentionWrapper(processor, soft_cap=30.0)
        output = wrapped(attn, hidden_states, ...)
    """

    def __init__(
        self,
        base_processor,
        soft_cap: float = 30.0,
        enabled: bool = True,
    ):
        self.base_processor = base_processor
        self.soft_cap = soft_cap
        self.enabled = enabled and HAS_FLEX_ATTENTION
        self.score_mod = create_tanh_softcap_score_mod(soft_cap) if self.enabled else None

    def __call__(self, attn, *args, **kwargs):
        # If disabled or not available, use base processor
        if not self.enabled:
            return self.base_processor(attn, *args, **kwargs)

        # For now, delegate to base processor
        # Full integration would require modifying the attention processors
        # to use apply_attention_with_softcap instead of F.scaled_dot_product_attention
        return self.base_processor(attn, *args, **kwargs)


def check_flex_attention_support() -> dict:
    """
    Check if flex_attention is available and supported.

    Returns:
        Dict with availability info.
    """
    result = {
        'available': HAS_FLEX_ATTENTION,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if HAS_FLEX_ATTENTION:
        # Try a simple test
        try:
            B, H, S, D = 1, 2, 4, 8
            q = torch.randn(B, H, S, D)
            k = torch.randn(B, H, S, D)
            v = torch.randn(B, H, S, D)

            score_mod = create_tanh_softcap_score_mod(30.0)
            flex_attn = _get_flex_attention()
            out = flex_attn(q, k, v, score_mod=score_mod)

            result['test_passed'] = True
            result['output_shape'] = list(out.shape)
        except Exception as e:
            result['test_passed'] = False
            result['error'] = str(e)

    return result
