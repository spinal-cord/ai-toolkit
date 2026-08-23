"""
Tests for sampling-level and per-sample attention tanh softcap support.

Covers:
1. Config parsing: SampleItem / SampleConfig / GenerateImageConfig new fields.
2. resolve_softcap_value hierarchy:
     per-sample (sampling) -> per-type-per-expert -> per-type -> per-expert ->
     sampling global (sampling) -> training global
3. set_sample_softcap_override / set_attention_softcapping state handling.
4. _apply_attention_with_softcap: per-sample enabled override takes precedence
   over the global sampling toggle (verified via the applied score_mod).

Run: python -m pytest tests/test_sample_softcap.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from toolkit.config_modules import GenerateImageConfig, SampleConfig, SampleItem
from toolkit.models.wan21 import wan_attn
from toolkit.models.wan21.wan_attn import (
    _apply_attention_with_softcap,
    resolve_softcap_value,
    set_attention_softcapping,
    set_sample_softcap_override,
    set_sampling_mode,
)


@pytest.fixture(autouse=True)
def reset_wan_attn_state():
    """Reset the module-global softcap state around every test."""
    set_attention_softcapping(
        enabled=True,
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
    set_sample_softcap_override(None, None)
    set_sampling_mode(False)
    yield
    set_sample_softcap_override(None, None)
    set_sampling_mode(False)


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------

def test_sample_item_softcap_defaults():
    sample_config = SampleConfig(samples=[{'prompt': 'a cat'}])
    item = sample_config.samples[0]
    assert item.attention_tanh_softcap_enabled is None
    assert item.attention_tanh_softcap_value is None


def test_sample_item_softcap_parsing():
    sample_config = SampleConfig(samples=[{
        'prompt': 'a cat',
        'attention_tanh_softcap_enabled': True,
        'attention_tanh_softcap_value': 20.0,
    }])
    item = sample_config.samples[0]
    assert item.attention_tanh_softcap_enabled is True
    assert item.attention_tanh_softcap_value == 20.0


def test_sample_config_softcap_value_default():
    sample_config = SampleConfig(samples=[{'prompt': 'a cat'}])
    assert sample_config.attention_tanh_softcap_enabled is False
    assert sample_config.attention_tanh_softcap_value is None


def test_sample_config_softcap_value_parsing():
    sample_config = SampleConfig(
        samples=[{'prompt': 'a cat'}],
        attention_tanh_softcap_enabled=True,
        attention_tanh_softcap_value=25.0,
    )
    assert sample_config.attention_tanh_softcap_enabled is True
    assert sample_config.attention_tanh_softcap_value == 25.0


def test_generate_image_config_softcap_defaults():
    cfg = GenerateImageConfig(prompt='a cat', output_path='/tmp/test.jpg')
    assert cfg.attention_tanh_softcap_enabled is False
    assert cfg.attention_tanh_softcap_value is None


def test_generate_image_config_softcap_parsing():
    cfg = GenerateImageConfig(
        prompt='a cat',
        output_path='/tmp/test.jpg',
        attention_tanh_softcap_enabled=True,
        attention_tanh_softcap_value=15.0,
    )
    assert cfg.attention_tanh_softcap_enabled is True
    assert cfg.attention_tanh_softcap_value == 15.0


# ---------------------------------------------------------------------------
# 2. Value resolution hierarchy
# ---------------------------------------------------------------------------

def test_resolve_training_global_by_default():
    assert resolve_softcap_value('self_attn', 'single') == 30.0


def test_resolve_sampling_inherits_training_value():
    set_sampling_mode(True)
    assert resolve_softcap_value('self_attn', 'single') == 30.0
    set_sampling_mode(False)


def test_resolve_sampling_uses_sample_global_value():
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_soft_cap=20.0)
    set_sampling_mode(True)
    assert resolve_softcap_value('self_attn', 'single') == 20.0
    set_sampling_mode(False)
    # training value is unaffected
    assert resolve_softcap_value('self_attn', 'single') == 30.0


def test_resolve_per_sample_value_wins_in_sampling():
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_soft_cap=20.0)
    set_sampling_mode(True)
    set_sample_softcap_override(enabled=True, value=10.0)
    assert resolve_softcap_value('self_attn', 'single') == 10.0
    assert resolve_softcap_value('cross_attn', 'high_noise') == 10.0
    set_sample_softcap_override(None, None)
    set_sampling_mode(False)


def test_resolve_per_sample_value_ignored_in_training():
    set_sampling_mode(False)
    set_sample_softcap_override(enabled=True, value=10.0)
    # training must never see the per-sample override
    assert resolve_softcap_value('self_attn', 'single') == 30.0
    set_sample_softcap_override(None, None)


def test_resolve_per_type_override_beats_sample_global():
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_soft_cap=20.0,
                              soft_cap_cross_attn=25.0)
    set_sampling_mode(True)
    # per-type override (25) beats the sampling global (20)
    assert resolve_softcap_value('cross_attn', 'single') == 25.0
    # no cross_attn override for self_attn -> sampling global (20)
    assert resolve_softcap_value('self_attn', 'single') == 20.0
    set_sampling_mode(False)


def test_resolve_per_sample_beats_per_type():
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_soft_cap=20.0,
                              soft_cap_cross_attn=25.0)
    set_sampling_mode(True)
    set_sample_softcap_override(enabled=True, value=5.0)
    # per-sample (5) beats per-type (25)
    assert resolve_softcap_value('cross_attn', 'single') == 5.0
    set_sample_softcap_override(None, None)
    set_sampling_mode(False)


# ---------------------------------------------------------------------------
# 3. Override state handling
# ---------------------------------------------------------------------------

def test_set_sample_softcap_override_none_state():
    set_sample_softcap_override(None, None)
    cfg = wan_attn._attention_config
    assert cfg['sample_softcap_override_enabled'] is None
    assert cfg['sample_softcap_override_value'] is None


def test_set_sample_softcap_override_values():
    set_sample_softcap_override(True, 12.5)
    cfg = wan_attn._attention_config
    assert cfg['sample_softcap_override_enabled'] is True
    assert cfg['sample_softcap_override_value'] == 12.5
    set_sample_softcap_override(False, None)
    assert cfg['sample_softcap_override_enabled'] is False
    assert cfg['sample_softcap_override_value'] is None


def test_set_attention_softcapping_sample_value():
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_soft_cap=18.0)
    assert wan_attn._attention_config['softcap_sample_value'] == 18.0
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_soft_cap=None)
    assert wan_attn._attention_config['softcap_sample_value'] is None


# ---------------------------------------------------------------------------
# 4. _apply_attention_with_softcap - per-sample enabled override
# ---------------------------------------------------------------------------

def _make_qkv():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    v = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    return q, k, v


def _run_sampling_attention(backend='auto'):
    """Run one sampling attention call with the current override state."""
    set_sampling_mode(True)
    try:
        q, k, v = _make_qkv()
        return _apply_attention_with_softcap(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            soft_cap=resolve_softcap_value('self_attn', 'single'),
            attn_type='self_attn', expert='single', backend=backend,
        )
    finally:
        set_sampling_mode(False)


def test_sampling_softcap_off_by_default():
    # Global sampling toggle off, no per-sample override -> no cap applied.
    # With backend 'sdpa' and softcap off the output must equal plain SDPA.
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_enabled=False)
    set_sample_softcap_override(False, None)
    q, k, v = _make_qkv()
    set_sampling_mode(True)
    try:
        out = _apply_attention_with_softcap(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            soft_cap=30.0, attn_type='self_attn', expert='single', backend='sdpa',
        )
    finally:
        set_sampling_mode(False)
    expected = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, expected, atol=1e-5)


def test_per_sample_override_enables_softcap():
    # Global sampling toggle OFF, per-sample override ON -> cap IS applied.
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_enabled=False)
    set_sample_softcap_override(True, 30.0)
    # The override must be visible to the attention path.
    assert wan_attn._attention_config['sample_softcap_override_enabled'] is True
    # The actual call goes through the flex path (score_mod applied).
    # flex on CPU is slow-ish but fine for 16x16.
    out = _run_sampling_attention(backend='auto')
    assert out.shape == (1, 2, 16, 32)
    set_sample_softcap_override(None, None)


def test_per_sample_override_disables_softcap():
    # Global sampling toggle ON, per-sample override OFF -> no cap applied.
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_enabled=True)
    set_sample_softcap_override(False, None)
    q, k, v = _make_qkv()
    set_sampling_mode(True)
    try:
        # backend 'sdpa' + softcap off -> plain SDPA
        out = _apply_attention_with_softcap(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            soft_cap=30.0, attn_type='self_attn', expert='single', backend='sdpa',
        )
    finally:
        set_sampling_mode(False)
    expected = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, expected, atol=1e-5)
    set_sample_softcap_override(None, None)


def test_per_sample_value_used_in_call():
    # The per-sample value must be what the kernel sees. With a very small cap
    # (1.0) attention becomes much softer than with no cap; verify the output
    # differs from plain SDPA and matches a manual capped computation.
    soft_cap = 1.0
    set_attention_softcapping(enabled=True, soft_cap=30.0, sample_enabled=False)
    set_sample_softcap_override(True, soft_cap)
    q, k, v = _make_qkv()
    set_sampling_mode(True)
    try:
        out = _apply_attention_with_softcap(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            soft_cap=resolve_softcap_value('self_attn', 'single'),
            attn_type='self_attn', expert='single', backend='auto',
        )
    finally:
        set_sampling_mode(False)
    set_sample_softcap_override(None, None)
    # Plain SDPA on the same data is noticeably sharper (cap was applied).
    plain = F.scaled_dot_product_attention(q, k, v)
    assert not torch.allclose(out, plain, atol=1e-3)
    # Must match a manual capped-softmax computation with the per-sample value
    # (flex score_mod applies the same scaling internally: 1/sqrt(head_dim)).
    scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    capped = soft_cap * torch.tanh(scores / soft_cap)
    manual = torch.softmax(capped, dim=-1) @ v
    assert torch.allclose(out, manual, atol=1e-4)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
