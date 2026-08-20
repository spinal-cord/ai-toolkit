import math
import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
from functools import partial, lru_cache


# Global attention config - set by trainer when enabled
# Hierarchy: per-type-per-expert → per-type → global
# The WanAttnProcessor2_0 resolves which value to use and passes it directly to _apply_attention_with_softcap
_attention_config = {
    'softcap_enabled': True,
    # Whether tanh softcapping is also applied during SAMPLING (independent of
    # the training toggle). Off by default to match standard inference; users
    # can opt in via sample.attention_tanh_softcap_enabled in the job config.
    'softcap_sample_enabled': False,
    'softcap_value': 30.0,  # Global default
    'f32_rope_enabled': True,
    # Attention backend selection (set by the trainer from the job config)
    #   'auto'  = flex_attention when training with softcap, otherwise SDPA (legacy default)
    #   'flex'  = torch flex_attention (the only kernel that can apply tanh softcapping)
    #   'sdpa'  = F.scaled_dot_product_attention (PyTorch picks flash/mem-efficient kernel)
    #   'flash' = flash-attn v2 package (fp16/bf16, no arbitrary attention masks;
    #             native tanh softcapping via flash_attn 2.8.3+)
    # 'train_backend' applies during training steps, 'sample_backend' during sampling.
    'train_backend': 'auto',
    'sample_backend': 'auto',
    # Toggled by the trainer around sampling so training and sampling can use
    # different backends. NOTE: reading this inside compiled code makes Dynamo
    # specialize on it - the resulting graph variants are cached, so switching
    # only costs a recompile the first time each mode is entered.
    'in_sampling': False,
}


_VALID_BACKENDS = ('auto', 'flex', 'sdpa', 'flash')


def _normalize_backend(backend) -> str:
    backend = str(backend or 'auto').lower()
    if backend in ('native', 'sdpa_default'):
        return 'auto'
    if backend in _VALID_BACKENDS:
        return backend
    raise ValueError(
        f"Unknown attention backend '{backend}'. "
        f"Valid values: {', '.join(_VALID_BACKENDS)} (or 'native' as alias for 'auto')."
    )


def set_attention_backend_choice(train_backend: str = 'auto', sample_backend: str = 'auto'):
    """
    Set the attention kernel used during training and during sampling.

    Args:
        train_backend: 'auto' | 'flex' | 'sdpa' | 'flash' (also accepts 'native' as 'auto')
        sample_backend: same options, applied while generating samples

    NOTE: when tanh softcapping is enabled for a mode (training and/or
    sampling, see set_attention_softcapping) the cap is applied by whichever
    kernel can do it:
      - 'flash' applies it natively in-kernel (flash_attn 2.8.3+) and keeps its
        backend; it only defers to 'flex' if that flash_attn build predates
        native softcapping.
      - 'auto'/'sdpa' have no score hook, so an enabled softcap resolves them
        to 'flex' (the cap is implemented as a flex_attention score_mod).
    Softcapping is skipped (with a one-time warning) for fp32 layers under
    'flash', since flash_attn only accepts fp16/bf16.
    """
    _attention_config['train_backend'] = _normalize_backend(train_backend)
    _attention_config['sample_backend'] = _normalize_backend(sample_backend)


def set_sampling_mode(enabled: bool):
    """Toggle whether attention calls are currently part of sampling."""
    _attention_config['in_sampling'] = bool(enabled)


def get_effective_backend(in_sampling: Optional[bool] = None) -> str:
    """
    Resolve the kernel that will actually be used (after softcap resolution).
    Used for logging the effective configuration at setup time.
    """
    if in_sampling is None:
        in_sampling = _attention_config['in_sampling']
    backend = _attention_config['sample_backend'] if in_sampling else _attention_config['train_backend']
    softcap_on = (_attention_config['softcap_sample_enabled'] if in_sampling
                  else _attention_config['softcap_enabled'])
    if softcap_on:
        if backend == 'flash':
            # flash applies the cap natively (2.8.3+); it only defers to flex
            # when this flash_attn build predates native softcapping.
            if _flash_softcap_supported():
                return 'flash (softcap active)'
            return 'flex (softcap active)' if _flex_attention_available() \
                else 'flash (softcap skipped)'
        # auto/sdpa/flex all resolve to flex when softcap is on
        return 'flex (softcap active)'
    if backend == 'auto':
        return 'sdpa'
    return backend


_warned_once = set()


def _warn_once(key: str, message: str):
    """Warn at most once per key (avoids log spam from per-attention-call fallbacks)."""
    if key in _warned_once:
        return
    _warned_once.add(key)
    import warnings
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def check_flash_softcap_support() -> bool:
    """
    True if the installed flash_attn package supports native tanh softcapping
    (flash_attn_func gained the `softcap` argument in 2.8.3).
    """
    try:
        import inspect
        from flash_attn import flash_attn_func
        return 'softcap' in inspect.signature(flash_attn_func).parameters
    except Exception:
        return False


def _apply_flash_attention(query, key, value, dropout_p: float = 0.0,
                           is_causal: bool = False, attn_mask: Optional[torch.Tensor] = None,
                           softcap: float = 0.0):
    """
    flash-attn v2 kernel (flash_attn.flash_attn_func).

    Expects (B, H, L, D) layout; flash_attn uses (B, L, H, D) internally.
    Falls back to SDPA (with a one-time warning) when the request is not
    expressible with flash_attn (arbitrary masks, unsupported dtype).

    softcap > 0 enables flash_attn's native tanh softcapping:
        softcap * tanh(QK^T * softmax_scale / softcap)
    with softmax_scale defaulting to 1/sqrt(head_dim) - the identical
    Gemma2/Grok-1 semantics used by the flex_attention score_mod path. The
    cap is applied to the fp32 score accumulator inside the kernel (fwd+bwd).
    It is only available for fp16/bf16; for other dtypes (e.g. fp32 layers) or
    when an arbitrary mask is present we fall back to SDPA and softcapping is
    skipped (with a one-time warning).
    """
    if attn_mask is not None:
        if softcap > 0:
            _warn_once('flash_mask_softcap',
                       "flash attention softcapping cannot be combined with an "
                       "attention mask; skipping softcapping and using SDPA.")
        else:
            _warn_once('flash_mask',
                       "flash attention backend does not support attention masks; "
                       "using SDPA for masked attention.")
        return F.scaled_dot_product_attention(query, key, value,
                                              attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    if query.dtype not in (torch.float16, torch.bfloat16):
        if softcap > 0:
            _warn_once('flash_softcap_dtype',
                       f"flash attention softcapping requires fp16/bf16 (got {query.dtype}); "
                       "skipping softcapping for this layer and using SDPA.")
        else:
            _warn_once('flash_dtype',
                       f"flash attention backend requires fp16/bf16 (got {query.dtype}); using SDPA.")
        return F.scaled_dot_product_attention(query, key, value,
                                              attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    from flash_attn import flash_attn_func
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    # Native softcap (2.8.3+). 0.0 = deactivated -> identical to pre-softcap behavior.
    sc = float(softcap) if softcap and softcap > 0 else 0.0
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal, softcap=sc)
    return out.transpose(1, 2)


# Lazily-cached capability flags (None = unknown). Importing/inspecting the
# kernels on every attention call would be wasteful, so we probe once and cache.
_flash_softcap_support = None
_flex_available = None


def _flash_softcap_supported() -> bool:
    """Cached: can the installed flash_attn apply native softcapping?"""
    global _flash_softcap_support
    if _flash_softcap_support is None:
        _flash_softcap_support = check_flash_softcap_support()
    return _flash_softcap_support


def _flex_attention_available() -> bool:
    """Cached: is torch.nn.attention.flex_attention importable (PyTorch 2.5+)?"""
    global _flex_available
    if _flex_available is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
            _flex_available = True
        except Exception:
            _flex_available = False
    return _flex_available


def _compiler_disable(fn):
    """torch.compiler.disable with a safe fallback for older PyTorch."""
    try:
        return torch.compiler.disable(fn)
    except Exception:
        return fn

# Per-type softcap values (None = use global)
# These are resolved by WanAttnProcessor2_0 based on attn_type and current expert
_softcap_overrides = {
    # Per-attention-type overrides
    'self_attn': None,          # All self-attention
    'cross_attn': None,         # All cross-attention
    # Per-expert overrides
    'high_noise': None,         # All attention in high-noise expert
    'low_noise': None,          # All attention in low-noise expert
    # Per-type-per-expert overrides (most specific)
    'self_attn_high_noise': None,   # Self-attention in high-noise expert
    'self_attn_low_noise': None,    # Self-attention in low-noise expert
    'cross_attn_high_noise': None,  # Cross-attention in high-noise expert
    'cross_attn_low_noise': None,   # Cross-attention in low-noise expert
}

# BlockMask cache uses LRU to prevent memory leaks with dynamic padding masks
# Without LRU, every new mask allocation (different data_ptr) would create a new entry
# causing CUDA OOM over long training runs with variable-length sequences.


# GELU acceleration for Wan 2.2 FeedForward layers
_gelu_acceleration_enabled = False


def enable_gelu_acceleration():
    """
    Enable hardware-accelerated GELU for Wan 2.2 FeedForward layers.
    
    Wan 2.2 uses gelu-approximate in all FF layers. This patches diffusers' GELU class
    to use tanh.approx.f32 PTX instruction instead of standard tanh, providing a small
    speedup (~2-5% on FF layers) with identical numerical output.
    
    NOTE: This patches the GELU class globally for the process. This is safe because:
    - The accelerated GELU produces identical numerical output to standard GELU
    - It only affects models using gelu-approximate (like Wan 2.2)
    - Other models using different GELU variants are unaffected
    
    Returns:
        True if acceleration is enabled (either newly enabled or already enabled)
        False if acceleration failed to enable
    """
    global _gelu_acceleration_enabled
    if _gelu_acceleration_enabled:
        return True  # Already enabled
    
    try:
        from toolkit.util.gelu_acceleration import gelu_accelerated
        import diffusers.models.activations as activations_module
        
        # Save original GELU.gelu method
        original_gelu_method = activations_module.GELU.gelu
        
        # Patch to use accelerated version
        def accelerated_gelu_method(self, gate):
            if gate.device.type == "mps":
                # MPS doesn't support our custom ops, use original
                return original_gelu_method(self, gate)
            if getattr(self, 'approximate', 'none') != 'tanh':
                # Only accelerate tanh approximation; fall back for exact GELU (erf-based)
                return original_gelu_method(self, gate)
            
            # Use accelerated GELU (tanh approximation)
            return gelu_accelerated(gate)
        
        # Defensive check: only patch if GELU has a .gelu method
        if not hasattr(activations_module.GELU, 'gelu'):
            raise AttributeError("diffusers.GELU has no 'gelu' method - API may have changed")
        
        activations_module.GELU.gelu = accelerated_gelu_method
        _gelu_acceleration_enabled = True
        return True
    
    except Exception as e:
        print(f"Warning: Failed to enable GELU acceleration: {e}")
        return False


def is_gelu_acceleration_enabled():
    """Check if GELU acceleration is enabled."""
    return _gelu_acceleration_enabled


# Hardware-accelerated approximate tanh using PTX instruction
# Based on attention-gym reference implementation
# Uses tanh.approx.f32 which is significantly faster than standard torch.tanh
_tanh_approx = None


def _get_tanh_approx():
    """
    Get hardware-accelerated approximate tanh operator with custom autograd.
    Falls back to torch.tanh if custom op registration fails.
    
    Always uses our optimized autograd wrapper (_TanhApprox.apply) to ensure
    the backward pass reuses the saved forward output, avoiding recomputation.
    """
    global _tanh_approx
    if _tanh_approx is not None:
        return _tanh_approx

    try:
        # Import torch.compile internals for custom op registration
        from torch._inductor.lowering import make_pointwise, register_lowering
        from torch._inductor.virtualized import ops

        # Register custom op if not already present
        # Use nested hasattr to handle older PyTorch versions gracefully
        if not hasattr(torch.ops, 'approx') or not hasattr(torch.ops.approx, 'tanh'):
            @torch.library.custom_op("approx::tanh", mutates_args=())
            def _tanh_approx_impl(inp: torch.Tensor) -> torch.Tensor:
                return torch.tanh(inp)

            @_tanh_approx_impl.register_fake
            def _(inp: torch.Tensor) -> torch.Tensor:
                return torch.tanh(inp)

        # ALWAYS register the PTX lowering, even if the op already exists.
        # register_lowering safely overwrites existing lowerings. This ensures
        # the tanh.approx.f32 optimization is used even if another library
        # registered approx::tanh without a lowering.
        def _tanh_approx_lowering(inp):
            fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
            return make_pointwise(fn)(inp)

        register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)
        
        # Always use our optimized autograd wrapper, even if op already exists
        # This ensures the backward pass reuses the saved forward output
        class _TanhApprox(torch.autograd.Function):
            @staticmethod
            def forward(x):
                return torch.ops.approx.tanh(x)

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                result = output
                ctx.save_for_backward(result)

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * (1 - result * result)

            @staticmethod
            def vmap(info, in_dims, x):
                # Use torch.tanh for vmap compatibility (matches attention-gym reference)
                # flex_attention uses torch.compile, not vmap, so this path isn't perf-critical
                return torch.tanh(x), 0

        _tanh_approx = _TanhApprox.apply
        return _tanh_approx

    except (ImportError, RuntimeError, AttributeError):
        # Fall back to standard torch.tanh
        _tanh_approx = torch.tanh
        return _tanh_approx

# Softcapping stats collector
# NOTE: These are only mutated inside torch.compiler.disable'd regions
# (_maybe_collect_softcap_stats, _resolve_softcap_for_current_expert), so
# stats collection works in eager mode AND under torch.compile (as a cheap
# graph break) without creating Dynamo guards / recompilation storms.
#
# IMPORTANT: current_step is set by the trainer (not auto-incremented per attention)
# to reflect actual training steps, not attention operation count.
#
# Stats are now tracked per attention type (self/cross) and per expert (high/low noise)
# to enable better tuning of softcap values for different attention patterns.
_softcap_stats = {
    'enabled': False,
    'sample_every_n_steps': 100,  # Sample/collect stats every N training steps
    'current_step': 0,  # Set by trainer, NOT incremented per attention call
    'current_expert': 'single',  # Current active expert: 'single', 'high', or 'low'
    'sample_stats': [],  # Recent sampling results (now includes type/expert labels)
    'fallback_count': 0,  # Count of flex_attention fallbacks to SDPA
}


def configure_softcap_logging(enabled: bool = True, sample_every_n_steps: int = 100):
    """
    Configure softcapping statistics sampling.

    Args:
        enabled: Whether to collect and log stats
        sample_every_n_steps: Sample attention scores every N training steps.
            Note: This controls sampling frequency, not logging frequency.
            Logging is triggered by the trainer separately.
    """
    _softcap_stats['enabled'] = enabled
    _softcap_stats['sample_every_n_steps'] = sample_every_n_steps


def get_softcap_stats():
    """
    Get current softcapping statistics with per-type breakdown.
    
    Returns stats organized by attention type (self/cross) and expert (high/low noise).
    """
    stats = {
        'enabled': _softcap_stats['enabled'],
        'current_step': _softcap_stats['current_step'],
        'current_expert': _softcap_stats['current_expert'],
        'fallback_count': _softcap_stats['fallback_count'],
        'sample_stats': _softcap_stats['sample_stats'][-50:],  # Keep more samples for breakdown
    }
    
    # Organize stats by attention type and expert for easier analysis
    # Group last 50 samples into buckets: {type: {expert: [stats]}}
    grouped = {'self_attn': {'single': [], 'high': [], 'low': []},
               'cross_attn': {'single': [], 'high': [], 'low': []}}
    
    for s in stats['sample_stats']:
        attn_type = s.get('attn_type', 'self_attn')
        expert = s.get('expert', 'single')
        if attn_type in grouped and expert in grouped[attn_type]:
            grouped[attn_type][expert].append(s)
    
    stats['grouped'] = grouped
    return stats


def reset_softcap_stats():
    """Reset softcapping statistics."""
    _softcap_stats['current_step'] = 0
    _softcap_stats['sample_stats'] = []


def update_softcap_step(step: int, expert: str = 'single'):
    """
    Update the current training step and active expert (called by trainer).
    
    IMPORTANT: This is called once per training step, NOT per attention operation.
    This ensures logging frequency is based on training steps, not attention count.
    
    Args:
        step: Current training step number
        expert: Active expert label ('single', 'high', or 'low')
    """
    _softcap_stats['current_step'] = step
    _softcap_stats['current_expert'] = expert


# Global cache for score_mod functions
# Prevents unnecessary recompilation under torch.compile by reusing the same function object
_score_mod_cache = {}


def _get_score_mod(soft_cap: float, head_dim: int):
    """
    Get or create a cached score_mod function for the given soft_cap and head_dim.
    
    This prevents torch.compile from seeing a new closure on every call,
    which would cause unnecessary graph breaks or cache misses.
    
    IMPORTANT: score_mod receives RAW QK^T scores (NOT pre-scaled by 1/sqrt(d)).
    We apply scaling AND softcapping here to match Gemma2/Grok behavior:
    soft_cap * tanh( (QK^T / sqrt(d)) / soft_cap )
    
    PRESCALE_QK=True was removed because it neuters softcapping:
    - With PRESCALE_QK, score_mod sees scores already divided by sqrt(d)
    - For head_dim=128, sqrt(d)≈11.3, so scores are ~[-6,6] instead of [-60,60]
    - With soft_cap=30, tanh(score/30) ≈ identity for [-6,6] → NO softcapping effect
    """
    # Normalize to float to avoid cache duplicates (int 30 vs float 30.0)
    soft_cap = float(soft_cap)
    cache_key = (soft_cap, head_dim)
    
    if cache_key not in _score_mod_cache:
        tanh_fn = _get_tanh_approx()
        # Combine scaling (1/sqrt(d)) and softcap (1/soft_cap) into single factor
        inv_scale = 1.0 / (soft_cap * math.sqrt(head_dim))

        def tanh_softcap(score, b, h, q_idx, kv_idx):
            # score is raw QK^T; scale then apply tanh softcap
            return soft_cap * tanh_fn(score * inv_scale)

        tanh_softcap.__name__ = f"tanh_softcap_{soft_cap}_hd{head_dim}"
        _score_mod_cache[cache_key] = tanh_softcap
    
    return _score_mod_cache[cache_key]


def set_attention_softcapping(
    enabled: bool = True,
    soft_cap: float = 30.0,
    sample_enabled: Optional[bool] = None,  # None = leave sampling setting unchanged
    # Per-attention-type overrides
    soft_cap_self_attn: float = None,
    soft_cap_cross_attn: float = None,
    # Per-expert overrides
    soft_cap_high_noise: float = None,
    soft_cap_low_noise: float = None,
    # Per-type-per-expert overrides (most specific)
    soft_cap_self_attn_high_noise: float = None,
    soft_cap_self_attn_low_noise: float = None,
    soft_cap_cross_attn_high_noise: float = None,
    soft_cap_cross_attn_low_noise: float = None,
):
    """
    Set attention softcapping configuration with per-type and per-expert overrides.
    
    Hierarchy (most specific to least):
        per-type-per-expert → per-type → per-expert → global
    
    The WanAttnProcessor2_0 resolves which value to use and passes it directly
    to _apply_attention_with_softcap as a float parameter.
    
    Args:
        enabled: Whether softcapping is enabled during training
        sample_enabled: Whether softcapping is also applied during sampling
            (None = do not change the current sampling setting). When enabled,
            sampling uses the same soft_cap/overrides as training.
        soft_cap: Global default softcap value
        soft_cap_self_attn: Override for all self-attention
        soft_cap_cross_attn: Override for all cross-attention
        soft_cap_high_noise: Override for all attention in high-noise expert
        soft_cap_low_noise: Override for all attention in low-noise expert
        soft_cap_self_attn_high_noise: Override for self-attention in high-noise expert
        soft_cap_self_attn_low_noise: Override for self-attention in low-noise expert
        soft_cap_cross_attn_high_noise: Override for cross-attention in high-noise expert
        soft_cap_cross_attn_low_noise: Override for cross-attention in low-noise expert
    """
    _attention_config['softcap_enabled'] = enabled
    if sample_enabled is not None:
        _attention_config['softcap_sample_enabled'] = bool(sample_enabled)
    _attention_config['softcap_value'] = soft_cap
    
    # Store overrides in the dedicated dict
    _softcap_overrides['self_attn'] = soft_cap_self_attn
    _softcap_overrides['cross_attn'] = soft_cap_cross_attn
    _softcap_overrides['high_noise'] = soft_cap_high_noise
    _softcap_overrides['low_noise'] = soft_cap_low_noise
    _softcap_overrides['self_attn_high_noise'] = soft_cap_self_attn_high_noise
    _softcap_overrides['self_attn_low_noise'] = soft_cap_self_attn_low_noise
    _softcap_overrides['cross_attn_high_noise'] = soft_cap_cross_attn_high_noise
    _softcap_overrides['cross_attn_low_noise'] = soft_cap_cross_attn_low_noise


def set_attention_f32_rope(enabled: bool = True):
    """Set global attention F32 RoPE acceleration configuration."""
    _attention_config['f32_rope_enabled'] = enabled


@_compiler_disable
def _resolve_softcap_for_current_expert(attn_type: str) -> float:
    """
    Resolve the effective softcap value for the expert currently set by the
    trainer (via update_softcap_step).

    This runs in a compiler-disabled region so that reading the mutable global
    _softcap_stats['current_expert'] does NOT create Dynamo guards - otherwise
    switching between high/low-noise experts (Wan 2.2 dual-expert) would force
    a full recompile of the model on every switch.
    """
    current_expert = _softcap_stats.get('current_expert', 'single')
    if current_expert == 'high':
        expert = 'high_noise'
    elif current_expert == 'low':
        expert = 'low_noise'
    else:
        expert = 'single'
    return resolve_softcap_value(attn_type, expert)


@_compiler_disable
def _maybe_collect_softcap_stats(query, key, lse, soft_cap, attn_type, q_len):
    """
    Sample a small subset of attention scores for softcapping statistics.

    Runs in a compiler-disabled region (a graph break under torch.compile),
    which keeps the mutable _softcap_stats global out of the compiled graph
    while still letting stats collection work with compile enabled. The real
    work (randperm + matmul + one .cpu() transfer) only happens every
    `sample_every_n_steps` training steps; on all other calls this is a cheap
    no-op check.
    """
    if not _softcap_stats['enabled']:
        return
    step = _softcap_stats['current_step']
    if step <= 0 or step % _softcap_stats['sample_every_n_steps'] != 0:
        return
    try:
        sample_stats = _sample_attention_scores(query, key, soft_cap)
        if lse is not None:
            # Lower LSE = softer/more diffuse attention (softcapping working)
            sample_stats['avg_lse'] = float(lse.detach().mean().cpu())
        sample_stats['step'] = step
        sample_stats['seq_len'] = q_len
        sample_stats['attn_type'] = attn_type
        sample_stats['expert'] = _softcap_stats['current_expert']
        sample_stats['soft_cap_used'] = soft_cap
        _softcap_stats['sample_stats'].append(sample_stats)
        # Trim to prevent unbounded growth (memory leak over long runs)
        if len(_softcap_stats['sample_stats']) > 200:
            del _softcap_stats['sample_stats'][:-200]
    except Exception:
        pass  # stats must never break training


def resolve_softcap_value(attn_type: str, expert: str) -> float:
    """
    Resolve the effective softcap value for given attention type and expert.
    
    Hierarchy (most specific to least):
        per-type-per-expert → per-type → per-expert → global
    
    This is called by WanAttnProcessor2_0 BEFORE passing to _apply_attention_with_softcap.
    
    Args:
        attn_type: 'self_attn' or 'cross_attn'
        expert: 'single', 'high_noise', or 'low_noise'
    
    Returns:
        The float softcap value to use
    """
    # Most specific: per-type-per-expert
    combined_key = f"{attn_type}_{expert}" if expert != 'single' else None
    if combined_key and _softcap_overrides.get(combined_key) is not None:
        return float(_softcap_overrides[combined_key])
    
    # Per-type override
    if _softcap_overrides.get(attn_type) is not None:
        return float(_softcap_overrides[attn_type])
    
    # Per-expert override (only if not single)
    if expert != 'single' and _softcap_overrides.get(expert) is not None:
        return float(_softcap_overrides[expert])
    
    # Global default
    return float(_attention_config['softcap_value'])


def _sdpa_mask_to_mask_mod(attn_mask: torch.Tensor, target_device: torch.device, is_causal: bool = False):
    """
    Convert an SDPA-style attention mask to a flex_attention mask_mod function.

    This is used to create a BlockMask that allows flex_attention to skip
    computation for fully-masked blocks, instead of computing all scores and
    masking them afterward.

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
        # Float mask: -inf or very negative values = cannot attend, 0 = can attend
        # Some implementations use -1e9 instead of -inf (especially in fp16)
        is_masked = attn_mask.to(torch.float32) < -1e4

    # Create padding mask_mod
    def padding_mask_mod(b, h, q_idx, kv_idx):
        # Index into the mask tensor using tensor ops
        if is_masked.ndim == 4:
            is_pos_masked = is_masked[b, h, q_idx, kv_idx]
        elif is_masked.ndim == 3:
            is_pos_masked = is_masked[b, q_idx, kv_idx]
        else:
            is_pos_masked = is_masked[q_idx, kv_idx]

        # Position is valid if not masked
        return ~is_pos_masked

    # Use flex_attention's and_masks for cleaner composition
    from torch.nn.attention.flex_attention import and_masks
    
    if is_causal:
        def causal_mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        return and_masks(causal_mask_mod, padding_mask_mod)
    else:
        return padding_mask_mod


def _sample_attention_scores(query, key, soft_cap):
    """
    Sample a small subset of attention scores to analyze softcapping effect.
    Uses random sampling to avoid bias from always sampling first positions.
    
    OPTIMIZED: Single .cpu() transfer instead of 10+ separate calls to avoid
    multiple CUDA synchronizations.
    
    Uses FP32 for score computation to avoid BF16 precision loss/overflow
    in statistical reductions (min, max, std, etc.).
    """
    seq_len = query.shape[2]
    sample_size = min(4, seq_len)
    # Random sampling - early positions have different dynamics than later ones
    indices = torch.randperm(seq_len, device=query.device)[:sample_size]
    # Cast to FP32 for precise stat gathering and to avoid BF16 overflow
    q_sample = query[:, :, indices].to(torch.float32)
    k_sample = key[:, :, indices].to(torch.float32)

    # Compute raw scores (Q @ K^T / sqrt(d))
    # CORRECT: q_sample @ k_sample^T gives shape (B, H, sample_size, sample_size)
    head_dim = query.shape[-1]
    raw_scores = torch.matmul(
        q_sample,
        k_sample.transpose(-2, -1)
    ) / (head_dim ** 0.5)

    # Apply softcapping
    tanh_fn = _get_tanh_approx()
    capped_scores = soft_cap * tanh_fn(raw_scores / soft_cap)

    # OPTIMIZED: Gather all stats in a single tensor, one .cpu() transfer
    abs_raw = torch.abs(raw_scores)
    abs_capped = torch.abs(capped_scores)
    
    # Single reduction pass
    stats_tensor = torch.stack([
        raw_scores.min(),
        raw_scores.max(),
        raw_scores.mean(),
        raw_scores.std(),
        capped_scores.min(),
        capped_scores.max(),
        capped_scores.mean(),
        capped_scores.std(),
        ((abs_raw > soft_cap).float().mean() * 100),  # pct_capped
        (1 - abs_capped.max() / torch.clamp(abs_raw.max(), min=1e-8)) * 100,  # max_reduction_pct
    ], dim=0)
    
    # Single CPU transfer (avoids 10+ CUDA syncs)
    stats_list = stats_tensor.cpu().tolist()
    
    return {
        'raw_min': stats_list[0],
        'raw_max': stats_list[1],
        'raw_mean': stats_list[2],
        'raw_std': stats_list[3],
        'capped_min': stats_list[4],
        'capped_max': stats_list[5],
        'capped_mean': stats_list[6],
        'capped_std': stats_list[7],
        'pct_capped': stats_list[8],
        'max_reduction_pct': stats_list[9],
    }


# BlockMask cache with LRU-like eviction to prevent memory leaks
# Uses OrderedDict to track access order and evict oldest entries when full
from collections import OrderedDict

_block_mask_cache = OrderedDict()
# 256 entries ≈ 256MB worst case (all 8Kx8K masks at ~2.5MB each)
# In practice: most training uses fixed dimensions → ~5-20 entries → <10MB
_BLOCK_MASK_CACHE_MAX_SIZE = 256


def _clear_block_mask_cache():
    """Clear the BlockMask cache (call when changing device or model)."""
    _block_mask_cache.clear()


def _compute_mask_hash(attn_mask: torch.Tensor) -> str:
    """
    Compute a collision-resistant hash for an attention mask.
    
    CRITICAL: Must use the SAME masking logic as _sdpa_mask_to_mask_mod
    to avoid cache collisions where hash says "same mask" but mask_mod
    treats positions differently.
    
    Called ONCE per unique mask (then cached), so we can afford a proper checksum.
    Uses multiple independent checksums to make collisions practically impossible.
    
    Uses nonzero() instead of torch.arange(n)[flat] to avoid allocating a massive
    temporary array for large masks (e.g., 8192x8192 = 536 MB for int64 arange).
    
    Args:
        attn_mask: Attention mask tensor
            - Boolean: True = unmasked, False = masked
            - Float: 0 = unmasked, < -1e4 = masked (includes -inf and -1e9)
        
    Returns:
        String hash uniquely identifying the mask content
    """
    # Use SAME masking logic as _sdpa_mask_to_mask_mod to avoid collisions
    if attn_mask.dtype == torch.bool:
        # Boolean mask: True = unmasked, False = masked
        is_masked = ~attn_mask
    else:
        # Float mask: < -1e4 = masked (handles both -inf and -1e9)
        is_masked = attn_mask.to(torch.float32) < -1e4
    
    flat = is_masked.flatten()
    
    # Get masked positions directly with nonzero() - avoids allocating torch.arange(n)
    # For 8192x8192 mask, torch.arange would use 536 MB temporarily
    masked_indices = flat.nonzero(as_tuple=True)[0]  # Returns 1D tensor of indices directly
    
    count = masked_indices.numel()
    if count == 0:
        return f"{tuple(attn_mask.shape)}:{attn_mask.dtype}:none"
    
    # Three independent checksums - collisions would require:
    # - Same count AND same sum of indices AND same sum of squared indices
    # - Statistically impossible for different masks
    # Stack all stats into one tensor, single .cpu() call to avoid multiple CUDA syncs
    stats = torch.stack([
        torch.tensor(count, dtype=torch.int64, device=flat.device),
        masked_indices.sum(),
        (masked_indices * masked_indices).sum(),
    ])
    count, sum_idx, sum_sq_idx = stats.cpu().tolist()
    
    return f"{tuple(attn_mask.shape)}:{attn_mask.dtype}:c{count}:s{sum_idx}:sq{sum_sq_idx}"


def _get_cached_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device, mask_type: str, mask_hash: str = None):
    """
    Get or create a cached BlockMask to avoid expensive recomputation.
    
    create_block_mask uses torch.vmap internally and is SLOW:
    - 1024x1024: 128ms
    - 4096x4096: 9ms  
    - 8192x8192: 35ms
    
    For a 40-layer model: 5+ seconds per forward pass without caching.
    With caching: ~0.03ms per lookup.
    
    NOTE on hash collisions: If two different masks hash to the same value,
    the cached BlockMask would be incorrectly reused. However, the 3-checksum
    hash (count + sum + sum_sq of masked indices) makes collisions statistically
    impossible for any realistic mask pattern.
    
    The cached BlockMask is a traced/compiled version of the mask_mod - it does
    NOT retain a reference to the original mask tensor. Once created, the
    BlockMask is independent of the mask_mod closure.
    
    Args:
        mask_mod: The mask modification function
        B, H, Q_LEN, KV_LEN: Attention dimensions
        device: Device for the mask
        mask_type: Type of mask ('causal', 'padding', 'combined')
        mask_hash: Collision-resistant hash for dynamic masks

    Returns:
        BlockMask object
    """
    from torch.nn.attention.flex_attention import create_block_mask

    # Cache key: dimensions + device + hash
    device_str = str(device)
    if mask_hash:
        cache_key = (mask_type, B, H, Q_LEN, KV_LEN, device_str, mask_hash)
    else:
        cache_key = (mask_type, B, H, Q_LEN, KV_LEN, device_str)

    # Check cache (moves to end if found - LRU behavior)
    if cache_key in _block_mask_cache:
        _block_mask_cache.move_to_end(cache_key)
        return _block_mask_cache[cache_key]

    # Evict oldest entries if cache is full (LRU)
    while len(_block_mask_cache) >= _BLOCK_MASK_CACHE_MAX_SIZE:
        _block_mask_cache.popitem(last=False)

    # Create new BlockMask (expensive - only called once per unique mask)
    block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )

    # Cache it (at end - most recently used)
    _block_mask_cache[cache_key] = block_mask
    return block_mask


def _apply_attention_with_softcap(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    soft_cap: float = 30.0,  # Pre-resolved softcap value (caller must resolve from config)
    # Metadata for stats tracking only - NOT used for config resolution
    attn_type: str = 'self_attn',  # For stats logging: 'self_attn' or 'cross_attn'
    expert: str = 'single',        # For stats logging: 'single', 'high_noise', or 'low_noise'
    backend: str = 'auto',         # 'auto' | 'flex' | 'sdpa' | 'flash' (pre-resolved by caller)
) -> torch.Tensor:
    """
    Apply attention with the configured backend and optional tanh softcapping.

    Backend resolution (softcap-aware):
      - 'auto'  → flex_attention when softcap is enabled for the current mode
                  (training or sampling), else SDPA (legacy default)
      - 'flex'  → flex_attention (score_mod softcapping when enabled)
      - 'sdpa'  → F.scaled_dot_product_attention; if softcap is enabled it is
                  forced to 'flex' (SDPA has no score hook)
      - 'flash' → flash-attn v2 package. When softcap is enabled it applies the
                  cap NATIVELY in-kernel (flash_attn 2.8.3+) and is NOT
                  overridden to 'flex'. Falls back to SDPA (softcap skipped)
                  for masks / non-fp16-bf16 dtypes such as fp32 layers.
      Training and sampling have independent softcap toggles.

    Uses flex_attention's proper architecture:
    - BlockMask for structural masking (padding, causal) - allows skipping masked blocks
    - score_mod for numerical transformation (softcapping) only

    Optimizations:
    - Caches BlockMask to avoid expensive recomputation (with LRU eviction)
    - Caches score_mod functions (prevents torch.compile recompilation storms)
    - Uses hardware-accelerated tanh.approx.f32 PTX instruction
    - Division converted to multiplication in score_mod
    - ROWS_GUARANTEED_SAFE + BLOCKS_ARE_CONTIGUOUS for causal masks
    - No device mismatch errors
    - torch.compile safe (mutable global state only touched in compiler-disabled regions)

    IMPORTANT: soft_cap and backend are passed as parameters, NOT looked up from
    config inside. The caller (WanAttnProcessor2_0) is responsible for resolving
    the correct values.

    Args:
        soft_cap: The softcap value to use (already resolved by caller)
        attn_type: For stats logging only - 'self_attn' or 'cross_attn'
        expert: For stats logging only - 'single', 'high_noise', or 'low_noise'
        backend: The attention backend to use (already resolved by caller)
    """
    in_sampling = _attention_config['in_sampling']

    if backend in ('native', 'sdpa_default'):
        backend = 'auto'

    # Tanh softcapping is configured independently per mode (training and
    # sampling). It is applied by whichever kernel can do it:
    #   - flex_attention via a score_mod (auto/sdpa/flex)
    #   - flash_attn natively in-kernel (2.8.3+, flash backend)
    # 'auto'/'sdpa' have no score hook of their own, so when softcap is on they
    # resolve to 'flex'. 'flash' keeps its backend and applies the cap natively.
    apply_softcap = (_attention_config['softcap_sample_enabled'] if in_sampling
                     else _attention_config['softcap_enabled'])
    if backend == 'auto':
        backend = 'flex' if apply_softcap else 'sdpa'
    elif backend == 'sdpa' and apply_softcap:
        backend = 'flex'
    # NOTE: 'flash' is intentionally NOT overridden to 'flex' here - it applies
    # softcapping natively (see the backend == 'flash' branch below).
    # The one exception: if this flash_attn build predates native softcapping
    # (pre-2.8.3) and flex_attention is available, defer to flex so the cap is
    # preserved instead of being silently dropped.
    if backend == 'flash' and apply_softcap and not _flash_softcap_supported():
        if _flex_attention_available():
            _warn_once('flash_softcap_unsupported',
                       "flash_attn does not support native softcapping (needs >= 2.8.3); "
                       "using flex_attention to apply the cap.")
            backend = 'flex'
        else:
            _warn_once('flash_softcap_unsupported_noflex',
                       "flash_attn lacks native softcapping and flex_attention is "
                       "unavailable; softcapping will be skipped (plain flash).")

    if backend == 'flex':
        try:
            # soft_cap is already resolved by caller - use it directly
            # Try to import flex_attention
            from torch.nn.attention.flex_attention import flex_attention, AuxRequest

            # Extract dimensions from query tensor
            B, H, Q_LEN, head_dim = query.shape
            KV_LEN = key.shape[2]
            device = query.device

            # Get cached score_mod function (prevents compilation storms)
            # Pass head_dim so score_mod can apply correct scaling
            score_mod = _get_score_mod(soft_cap, head_dim) if apply_softcap else None
            # CRITICAL when softcapping: scale=1.0 disables PyTorch's default
            # 1/sqrt(d) scaling because score_mod handles scaling internally.
            # Without this, we get double-scaling: softmax((1/sqrt(d)) *
            # score_mod(QK^T)), which makes soft_cap=30 effectively become
            # soft_cap=30/sqrt(d)≈2.65, way too aggressive.
            flex_scale = 1.0 if apply_softcap else None

            # Create/cached BlockMask if we have a mask or causal attention
            block_mask = None
            if attn_mask is not None or is_causal:
                if attn_mask is not None:
                    # For padding masks, compute content-based hash (no stale cache risk)
                    mask_hash = _compute_mask_hash(attn_mask)
                    mask_mod = _sdpa_mask_to_mask_mod(attn_mask, device, is_causal)
                    mask_type = "combined" if is_causal else "padding"
                    block_mask = _get_cached_block_mask(
                        mask_mod, B, H, Q_LEN, KV_LEN, device, mask_type, mask_hash
                    )
                else:
                    # Pure causal mask - always cacheable (same for all calls with same dims)
                    def causal_mask_mod(b, h, q_idx, kv_idx):
                        return q_idx >= kv_idx
                    block_mask = _get_cached_block_mask(
                        causal_mask_mod, B, H, Q_LEN, KV_LEN, device, "causal"
                    )

            # Always request LSE - during training (grad enabled) it's free
            # (flex_attention computes it for backward regardless of return_aux)
            # and it is used for the attention-sharpness stat.
            return_aux = AuxRequest(lse=True)

            # Kernel options
            # ROWS_GUARANTEED_SAFE: every query has ≥1 valid key (only safe for pure causal)
            # BLOCKS_ARE_CONTIGUOUS: optimizes block traversal for causal patterns
            # NOTE: PRESCALE_QK removed - it neuters softcapping by dividing scores
            # by sqrt(d) BEFORE score_mod sees them, making soft_cap=30 a no-op.
            # Scaling is now handled inside score_mod for correct Gemma2/Grok behavior.
            kernel_options = {}
            if is_causal and attn_mask is None:  # Only pure causal - combined masks may have empty rows
                kernel_options.update({
                    "ROWS_GUARANTEED_SAFE": True,
                    "BLOCKS_ARE_CONTIGUOUS": True,
                })

            # Run flex_attention
            output, aux = flex_attention(
                query, key, value,
                score_mod=score_mod,
                block_mask=block_mask,
                scale=flex_scale,
                return_aux=return_aux,
                kernel_options=kernel_options,
            )

            # Periodic stats sampling. Runs in a compiler-disabled region
            # (graph break under torch.compile), so stats work with compile
            # enabled; on non-sample steps it is a cheap no-op check.
            _maybe_collect_softcap_stats(
                query, key, aux.lse if apply_softcap else None, soft_cap, attn_type, Q_LEN)

            return output

        except (ImportError, AttributeError) as e:
            # ImportError/AttributeError: flex_attention not available or misconfigured
            if apply_softcap:
                # Permanently disable softcapping to avoid re-importing and re-raising every call
                set_attention_softcapping(enabled=False)
                import warnings
                warnings.warn(
                    f"flex_attention not available ({type(e).__name__}: {e}). "
                    f"Softcapping disabled permanently. Ensure PyTorch >= 2.5.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                _warn_once('flex_unavailable',
                           f"flex_attention not available ({type(e).__name__}: {e}); "
                           f"using SDPA instead.")
        except RuntimeError as e:
            # RuntimeError: transient issue (OOM, CUDA error, shape mismatch)
            # Log and fall back for this call, but disable permanently after repeated failures
            import warnings
            # Track fallback count for monitoring
            _softcap_stats['fallback_count'] += 1
            # Only warn if this is the first time we see this error type
            if not hasattr(_apply_attention_with_softcap, '_last_runtime_error') or \
               _apply_attention_with_softcap._last_runtime_error != type(e).__name__:
                _apply_attention_with_softcap._last_runtime_error = type(e).__name__
                warnings.warn(
                    f"flex_attention runtime error ({type(e).__name__}: {str(e)[:200]}). "
                    f"Falling back to SDPA for this operation. "
                    f"(Total fallbacks: {_softcap_stats['fallback_count']})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # Permanently disable softcap after 50 fallbacks to avoid retry overhead
            if apply_softcap and _softcap_stats['fallback_count'] > 50:
                set_attention_softcapping(enabled=False)
                warnings.warn(
                    f"flex_attention disabled permanently after {_softcap_stats['fallback_count']} fallbacks. "
                    f"Training without softcapping.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if backend == 'flash':
        try:
            out = _apply_flash_attention(
                query, key, value,
                dropout_p=dropout_p, is_causal=is_causal, attn_mask=attn_mask,
                softcap=soft_cap if apply_softcap else 0.0,
            )
            # Periodic softcap stats (compiler-disabled; no-op on most steps).
            # flash_attn does not return LSE here, so lse=None (the sharpness
            # stat is skipped, but score-range/pct_capped sampling still works).
            if apply_softcap:
                _maybe_collect_softcap_stats(query, key, None, soft_cap, attn_type,
                                             query.shape[2])
            return out
        except Exception as e:
            _warn_once('flash_failed',
                       f"flash attention backend failed ({type(e).__name__}: {e}); "
                       f"falling back to SDPA.")

    # Standard scaled_dot_product_attention
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


# modified to set the image embedder size
class WanAttnProcessor2_0:
    def __init__(self, num_img_tokens: int = 257):
        self.num_img_tokens = num_img_tokens
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Determine attention type BEFORE encoder_hidden_states gets modified
        # Self-attention: encoder_hidden_states is None (will be set to hidden_states)
        # Cross-attention: encoder_hidden_states is provided separately
        is_self_attn = encoder_hidden_states is None
        attn_type = 'self_attn' if is_self_attn else 'cross_attn'

        # Determine current expert (for per-expert softcap resolution).
        # Done in a compiler-disabled region so the mutable global does not
        # create Dynamo guards (no recompile on high/low expert switches).
        # Hierarchy: per-type-per-expert → per-type → per-expert → global
        soft_cap = _resolve_softcap_for_current_expert(attn_type)

        # Resolve the attention backend. Training and sampling can use
        # different backends (set via set_attention_backend_choice; the
        # trainer toggles sampling mode around generate_images). Reading
        # these globals makes Dynamo specialize on them - the resulting
        # graph variants are cached, so each mode only compiles once.
        in_sampling = _attention_config['in_sampling']
        backend = _attention_config['sample_backend'] if in_sampling else _attention_config['train_backend']
        expert = 'single'  # stats metadata only; actual expert is read in the stats probe
        
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:,
                                                              :self.num_img_tokens]
            encoder_hidden_states = encoder_hidden_states[:,
                                                          self.num_img_tokens:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            # Use F32 acceleration if enabled (faster than F64, more stable than BF16/FP16)
            rope_dtype = torch.float32 if _attention_config['f32_rope_enabled'] else torch.float64

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                # Save original dtype BEFORE any casting
                orig_dtype = hidden_states.dtype
                # Use rope_dtype for numerically stable RoPE computation
                rope_states = hidden_states.to(rope_dtype)
                # Only make contiguous if needed (avoids unnecessary copy)
                if not rope_states.is_contiguous():
                    rope_states = rope_states.contiguous()
                x_rotated = torch.view_as_complex(
                    rope_states.unflatten(3, (-1, 2)))
                # freqs is complex (from Rope.forward()). Cast to matching complex dtype
                # to avoid implicit upcast or discarding imaginary part.
                rope_complex_dtype = torch.complex64 if rope_dtype == torch.float32 else torch.complex128
                freqs_casted = freqs.to(rope_complex_dtype) if freqs.dtype != rope_complex_dtype else freqs
                x_out = torch.view_as_real(x_rotated * freqs_casted).flatten(3, 4)
                # CRITICAL: cast back to original model dtype (e.g., bf16), not rope_dtype
                return x_out.to(orig_dtype)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(
                2, (attn.heads, -1)).transpose(1, 2)

            # I2V cross-attention from image encoder - resolve its own softcap value
            soft_cap_img = _resolve_softcap_for_current_expert('cross_attn')
            hidden_states_img = _apply_attention_with_softcap(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False,
                soft_cap=soft_cap_img,
                attn_type='cross_attn',
                expert=expert,
                backend=backend,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = _apply_attention_with_softcap(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            soft_cap=soft_cap,
            attn_type=attn_type,
            expert=expert,
            backend=backend,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
