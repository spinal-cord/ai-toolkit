"""
Integration tests: the toolkit's WanAttnProcessor2_0 installed on the STANDARD
diffusers Wan path (the gap that made sampling/per-sample softcapping a no-op).

Covers:
1. Dual-format RoPE: the processor must handle the real (freqs_cos, freqs_sin)
   tuple that the installed diffusers passes (WanRotaryPosEmbed.forward), and
   the legacy single complex tensor. Verified by full-model forward equivalence
   with the stock diffusers WanAttnProcessor.
2. install_softcap_processors: covers both experts of a dual-expert model,
   is idempotent, and preserves existing WanAttnProcessor2_0 instances.
3. Softcapping end-to-end through an installed processor:
     - training mode + flex backend: output matches a manual
       soft_cap * tanh(score/soft_cap) capped-softmax computation
     - sampling mode + per-sample override: per-sample value / enable-disable
       takes effect end-to-end
     - cross-attention (attn2) with an encoder hidden state
4. CUDA bf16 + flash backend (skipped when unavailable): native in-kernel
   softcap path runs and matches the capped-softmax reference (loose tol).

Run: python -m pytest tests/test_softcap_processor_integration.py -v
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from diffusers import WanTransformer3DModel

from toolkit.models.wan21 import wan_attn
from toolkit.models.wan21.wan_attn import (
    WanAttnProcessor2_0,
    _apply_attention_with_softcap,  # noqa: F401  (imported for side-effect parity)
    install_softcap_processors,
    resolve_softcap_value,
    set_attention_backend_choice,
    set_attention_softcapping,
    set_sample_softcap_override,
    set_sampling_mode,
)

try:
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor as StockProcessor
except ImportError:  # pragma: no cover
    StockProcessor = None

if StockProcessor is None:  # pragma: no cover
    pytest.skip("stock diffusers WanAttnProcessor not available", allow_module_level=True)


def _make_tiny_model(**kwargs):
    """A tiny WanTransformer3DModel that runs in milliseconds on CPU."""
    defaults = dict(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=32,
        ffn_dim=64,
        num_layers=1,
        rope_max_seq_len=64,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return WanTransformer3DModel(**defaults).eval()


def _tiny_inputs(model):
    torch.manual_seed(1)
    hidden = torch.randn(1, model.in_channels, 2, 8, 8)   # B, C, F, H, W
    timestep = torch.tensor([500.0])
    encoder = torch.randn(1, 5, model.text_dim)
    return hidden, timestep, encoder


@pytest.fixture(autouse=True)
def reset_wan_attn_state():
    set_attention_softcapping(
        enabled=False,
        soft_cap=30.0,
        sample_enabled=False,
        sample_soft_cap=None,
        soft_cap_self_attn=None,
        soft_cap_cross_attn=None,
        soft_cap_high_noise=None,
        soft_cap_low_noise=None,
        soft_cap_self_attn_high_noise=None,
        soft_cap_self_attn_low_noise=None,
        soft_cap_cross_attn_high_noise=None,
        soft_cap_cross_attn_low_noise=None,
    )
    set_attention_backend_choice(train_backend='auto', sample_backend='auto')
    set_sample_softcap_override(None, None)
    set_sampling_mode(False)
    yield
    set_sample_softcap_override(None, None)
    set_sampling_mode(False)


def _stock_forward(model, hidden, timestep, encoder):
    """Full forward with the stock diffusers processor (reference)."""
    for block in model.blocks:
        block.attn1.set_processor(StockProcessor())
        block.attn2.set_processor(StockProcessor())
    with torch.no_grad():
        out = model(hidden, timestep, encoder)
    return out.sample if hasattr(out, "sample") else out


def _toolkit_forward(model, hidden, timestep, encoder):
    for block in model.blocks:
        block.attn1.set_processor(WanAttnProcessor2_0())
        block.attn2.set_processor(WanAttnProcessor2_0())
    with torch.no_grad():
        out = model(hidden, timestep, encoder)
    return out.sample if hasattr(out, "sample") else out


# ---------------------------------------------------------------------------
# 1. Dual-format RoPE
# ---------------------------------------------------------------------------

def test_full_forward_equivalence_real_rope_tuple():
    """Installed processor + real (cos, sin) tuple RoPE == stock processor."""
    model = _make_tiny_model()
    hidden, timestep, encoder = _tiny_inputs(model)
    ref = _stock_forward(model, hidden, timestep, encoder)
    got = _toolkit_forward(model, hidden, timestep, encoder)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_complex_rope_format_equivalent_to_real_tuple():
    """Legacy complex RoPE (e^{i theta}) == real (cos, sin) tuple format.

    The stock diffusers processor only accepts the real tuple, so the legacy
    path is validated against the tuple path of the same toolkit processor -
    both are the same rotation and must agree exactly.
    """
    model = _make_tiny_model()
    hidden, _, _ = _tiny_inputs(model)
    install_softcap_processors(model)

    attn = model.blocks[0].attn1
    x = model.patch_embedding(hidden).flatten(2).transpose(1, 2)
    x = model.blocks[0].norm1(x)
    cos, sin = model.rope(hidden)
    # Legacy format: complex (1, S, 1, D/2) = cos(theta) + i sin(theta), where the
    # real tuple is repeat-interleaved (even/odd positions carry the same angle).
    complex_freqs = torch.complex(cos[..., 0::2], sin[..., 1::2])
    assert complex_freqs.shape == (1, cos.shape[1], 1, cos.shape[3] // 2)

    with torch.no_grad():
        out_tuple = attn(x, rotary_emb=(cos, sin))
        out_complex = attn(x, rotary_emb=complex_freqs)
    assert out_complex.shape == out_tuple.shape
    torch.testing.assert_close(out_complex, out_tuple, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. install_softcap_processors
# ---------------------------------------------------------------------------

def test_install_covers_single_and_dual_expert_and_is_idempotent():
    m1 = _make_tiny_model()
    m2 = _make_tiny_model(num_layers=2)

    # Single-expert model: attn1 + attn2 of each block.
    n = install_softcap_processors(m1)
    assert n == 2
    for block in m1.blocks:
        assert isinstance(block.attn1.processor, WanAttnProcessor2_0)
        assert isinstance(block.attn2.processor, WanAttnProcessor2_0)
    # Idempotent.
    assert install_softcap_processors(m1) == 0

    # Dual-expert wrapper: both experts get covered.
    dual = types.SimpleNamespace(transformer_1=m1, transformer_2=m2)
    n = install_softcap_processors(dual)
    assert n == 4  # m2 has 2 blocks; m1 already installed
    for block in m2.blocks:
        assert isinstance(block.attn1.processor, WanAttnProcessor2_0)
        assert isinstance(block.attn2.processor, WanAttnProcessor2_0)
    assert install_softcap_processors(dual) == 0

    # None / non-wan models are safe no-ops.
    assert install_softcap_processors(None) == 0
    assert install_softcap_processors(torch.nn.Linear(4, 4)) == 0


def test_install_preserves_custom_num_img_tokens():
    """Existing WanAttnProcessor2_0 (e.g. from the I2V adapter) is not replaced."""
    m = _make_tiny_model()
    for block in m.blocks:
        block.attn2.set_processor(WanAttnProcessor2_0(num_img_tokens=42))
    install_softcap_processors(m)
    for block in m.blocks:
        assert block.attn2.processor.num_img_tokens == 42
        assert isinstance(block.attn1.processor, WanAttnProcessor2_0)


# ---------------------------------------------------------------------------
# 3. Softcapping end-to-end through the installed processor
# ---------------------------------------------------------------------------

def _apply_rope_real(x, freqs_cos, freqs_sin):
    """Stock-processor RoPE math (real tuple format) on (B, H, S, D) tensors."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    if cos.dim() == 4 and cos.shape[2] == 1:
        cos = cos.permute(0, 2, 1, 3)
        sin = sin.permute(0, 2, 1, 3)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _manual_self_attn(attn, hidden, rope, soft_cap=None):
    """Manual self-attention with the toolkit's exact softcap semantics:
    score = QK^T / sqrt(d); capped = soft_cap * tanh(score / soft_cap)."""
    q = attn.to_q(hidden)
    k = attn.to_k(hidden)
    v = attn.to_v(hidden)
    q = attn.norm_q(q)
    k = attn.norm_k(k)
    q = q.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    k = k.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    v = v.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    if rope is not None:
        q = _apply_rope_real(q, rope[0], rope[1])
        k = _apply_rope_real(k, rope[0], rope[1])
    score = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if soft_cap is not None:
        score = soft_cap * torch.tanh(score / soft_cap)
    w = torch.softmax(score, dim=-1)
    out = (w @ v).transpose(1, 2).flatten(2, 3)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    return out


def test_training_softcap_flex_matches_manual_capped_softmax():
    model = _make_tiny_model()
    hidden, timestep, encoder = _tiny_inputs(model)
    install_softcap_processors(model)

    attn = model.blocks[0].attn1
    # Self-attention input: patch-embedded tokens (same input is fed to both the
    # processor and the manual reference, so block-level conditioning is irrelevant).
    rope = model.rope(hidden)
    x = model.patch_embedding(hidden).flatten(2).transpose(1, 2)
    x = model.blocks[0].norm1(x)

    set_attention_softcapping(enabled=True, soft_cap=8.0)
    set_attention_backend_choice(train_backend='flex', sample_backend='flex')

    with torch.no_grad():
        got = attn(x, rotary_emb=rope)
        ref = _manual_self_attn(attn, x, rope, soft_cap=8.0)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    # Sanity: the cap actually changed the result vs uncapped attention.
    with torch.no_grad():
        uncapped = _manual_self_attn(attn, x, rope, soft_cap=None)
    assert not torch.allclose(got, uncapped, atol=1e-3)


def test_sampling_per_sample_override_end_to_end():
    """Per-sample enabled/value overrides resolve end-to-end in sampling mode."""
    model = _make_tiny_model()
    hidden, timestep, encoder = _tiny_inputs(model)
    install_softcap_processors(model)

    attn = model.blocks[0].attn1
    rope = model.rope(hidden)
    x = model.patch_embedding(hidden).flatten(2).transpose(1, 2)
    x = model.blocks[0].norm1(x)

    # Global sampling softcap OFF, training softcap OFF.
    set_attention_softcapping(enabled=False, soft_cap=30.0, sample_enabled=False)
    set_attention_backend_choice(train_backend='flex', sample_backend='flex')
    set_sampling_mode(True)

    with torch.no_grad():
        base = attn(x, rotary_emb=rope)  # no cap anywhere
        ref_uncapped = _manual_self_attn(attn, x, rope, soft_cap=None)
    torch.testing.assert_close(base, ref_uncapped, atol=1e-5, rtol=1e-5)

    # Per-sample override: ENABLED with its own value (beats global-off).
    set_sample_softcap_override(True, 5.0)
    assert resolve_softcap_value('self_attn', 'single') == 5.0
    with torch.no_grad():
        got = attn(x, rotary_emb=rope)
        ref = _manual_self_attn(attn, x, rope, soft_cap=5.0)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    # Per-sample override: explicitly DISABLED (beats global-on).
    set_attention_softcapping(enabled=False, soft_cap=30.0, sample_enabled=True,
                              sample_soft_cap=20.0)
    set_sample_softcap_override(False, None)
    with torch.no_grad():
        got = attn(x, rotary_emb=rope)
    torch.testing.assert_close(got, ref_uncapped, atol=1e-5, rtol=1e-5)

    # Per-sample override: value only (enabled follows global sampling toggle).
    set_sample_softcap_override(None, 12.0)
    with torch.no_grad():
        got = attn(x, rotary_emb=rope)
        ref = _manual_self_attn(attn, x, rope, soft_cap=12.0)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
    set_sample_softcap_override(None, None)

    # Inherited sampling global value (no per-sample override).
    with torch.no_grad():
        got = attn(x, rotary_emb=rope)
        ref = _manual_self_attn(attn, x, rope, soft_cap=20.0)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
    set_sampling_mode(False)


def test_cross_attention_with_softcap():
    """attn2 (cross-attention, no RoPE) matches the manual capped computation."""
    model = _make_tiny_model()
    hidden, timestep, encoder = _tiny_inputs(model)
    install_softcap_processors(model)

    attn = model.blocks[0].attn2
    x = model.patch_embedding(hidden).flatten(2).transpose(1, 2)
    x = model.blocks[0].norm1(x)
    enc_hidden = torch.randn(1, 5, model.text_dim)
    enc_hidden = torch.nn.functional.layer_norm(
        enc_hidden, (model.text_dim,))  # just needs to be a valid K/V source

    set_attention_softcapping(enabled=True, soft_cap=6.0)
    set_attention_backend_choice(train_backend='flex', sample_backend='flex')

    with torch.no_grad():
        got = attn(x, encoder_hidden_states=enc_hidden)

        # Manual reference (same q/k/v projections as the processor).
        q = attn.norm_q(attn.to_q(x))
        k = attn.norm_k(attn.to_k(enc_hidden))
        v = attn.to_v(enc_hidden)
        q = q.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        k = k.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        v = v.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        score = 6.0 * torch.tanh(score / 6.0)
        w = torch.softmax(score, dim=-1)
        ref = (w @ v).transpose(1, 2).flatten(2, 3)
        ref = attn.to_out[0](ref)
        ref = attn.to_out[1](ref)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. CUDA bf16 + flash native softcap
# ---------------------------------------------------------------------------

_flash_attn = pytest.importorskip("flash_attn", reason="flash_attn not installed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_bf16_flash_native_softcap():
    model = _make_tiny_model().to("cuda", torch.bfloat16)
    hidden, timestep, encoder = _tiny_inputs(model)
    hidden = hidden.to("cuda", torch.bfloat16)
    encoder = encoder.to("cuda", torch.bfloat16)
    timestep = timestep.to("cuda")
    install_softcap_processors(model)

    attn = model.blocks[0].attn1
    rope = model.rope(hidden)
    x = model.patch_embedding(hidden).flatten(2).transpose(1, 2)
    x = model.blocks[0].norm1(x)

    set_attention_softcapping(enabled=True, soft_cap=8.0)
    set_attention_backend_choice(train_backend='flash', sample_backend='flash')

    with torch.no_grad():
        got = attn(x, rotary_emb=rope)
        assert got.dtype == torch.bfloat16
        assert torch.isfinite(got.float()).all()

        # Manual reference in fp32 (flash applies the cap to the fp32 scores).
        q = attn.norm_q(attn.to_q(x)).float()
        k = attn.norm_k(attn.to_k(x)).float()
        v = attn.to_v(x).float()
        q = q.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        k = k.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        v = v.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        q = _apply_rope_real(q, rope[0].float(), rope[1].float())
        k = _apply_rope_real(k, rope[0].float(), rope[1].float())
        score = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        score = 8.0 * torch.tanh(score / 8.0)
        w = torch.softmax(score, dim=-1)
        ref = (w @ v).transpose(1, 2).flatten(2, 3)
        ref = attn.to_out[0](ref.to(torch.bfloat16))
        ref = attn.to_out[1](ref)
    # bf16 end-to-end: loose tolerance.
    assert (got.float() - ref.float()).abs().max().item() < 0.2, \
        f"flash native softcap output diverges from reference: " \
        f"max diff {(got.float() - ref.float()).abs().max().item():.4f}"
