# Thin extension of the official diffusers AutoencoderKLWan.
# All model / tiling / patchify logic comes from diffusers so it stays in sync
# with upstream. The only thing added here is gradient checkpointing support:
# the encoder/decoder forwards are monkeypatched with copies of the upstream
# forwards that add checkpointing branches, and the subclass re-enables
# _supports_gradient_checkpointing (upstream has it turned off).

import torch

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    CACHE_T,
    AutoencoderKLWan as AutoencoderKLWanBase,
    WanDecoder3d,
    WanEncoder3d,
)


# copied from diffusers WanEncoder3d.forward with gradient checkpointing added
def _wan_encoder_forward(self, x, feat_cache=None, feat_idx=[0]):
    use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None

    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)

    ## downsamples
    for layer in self.down_blocks:
        if use_ckpt:
            x = self._gradient_checkpointing_func(layer, x)
        elif feat_cache is not None:
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            x = layer(x)

    ## middle
    if use_ckpt:
        x = self._gradient_checkpointing_func(self.mid_block, x)
    else:
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)

    return x


# copied from diffusers WanDecoder3d.forward with gradient checkpointing added
def _wan_decoder_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing and feat_cache is None

    ## conv1
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)

    ## middle
    if use_ckpt:
        x = self._gradient_checkpointing_func(self.mid_block, x)
    else:
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

    ## upsamples
    for up_block in self.up_blocks:
        if use_ckpt:
            x = self._gradient_checkpointing_func(up_block, x, None, [0], first_chunk)
        else:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)
    return x


WanEncoder3d.forward = _wan_encoder_forward
WanDecoder3d.forward = _wan_decoder_forward


class AutoencoderKLWan(AutoencoderKLWanBase):
    _supports_gradient_checkpointing = True


# ---------------------------------------------------------------------------
# VAE State Dict Normalization
# ---------------------------------------------------------------------------
# Some community/custom Wan VAEs (e.g. the fp32 variant of the Wan 2.1 VAE)
# use a different naming convention for their tensors compared to the official
# bf16 VAE published by Wan-AI / ai-toolkit. The architecture is identical,
# but tensor names differ (e.g. "encoder.conv1" vs "encoder.conv_in",
# "encoder.downsamples" vs "encoder.down_blocks", etc.).
#
# This module provides:
#   1. `detect_alternative_vae_naming(state_dict)` — returns True if the
#      state dict uses the alternative naming scheme.
#   2. `normalize_vae_state_dict(state_dict)` — renames tensors in the
#      alternative scheme to match the standard AutoencoderKLWan naming so
#      that diffusers can load them via `from_pretrained` / `load_state_dict`.
#
# The mapping is derived from shape matching and the systematic rules
# documented in `wan22_VAE_findings.md`.

import re
from typing import Dict, List, Tuple


def _key_matches_alternative_naming(state_dict: Dict[str, torch.Tensor]) -> bool:
    """
    Heuristic: detect whether the state dict uses the alternative (fp32)
    naming convention instead of the standard AutoencoderKLWan one.

    Detection criteria (any one of the following is sufficient):
      - Contains `encoder.conv1` or `encoder.downsamples.` keys.
      - Contains `decoder.upsamples.` or `decoder.middle.` keys.
      - Contains top-level `conv1` / `conv2` keys (quantization layers).
      - Does NOT contain the canonical `encoder.conv_in` key but has
        `encoder.*` keys.

    Returns True if the VAE appears to use the alternative naming.
    """
    keys = set(state_dict.keys())
    if not keys:
        return False

    alt_markers = [
        "encoder.conv1.",
        "encoder.downsamples.",
        "encoder.middle.",
        "decoder.conv1.",
        "decoder.upsamples.",
        "decoder.middle.",
    ]
    for marker in alt_markers:
        if any(k.startswith(marker) for k in keys):
            return True

    # Top-level quantization layers use "conv1"/"conv2" rather than
    # "quant_conv"/"post_quant_conv".
    if "conv1.bias" in keys or "conv2.bias" in keys:
        return True

    # If the canonical conv_in is missing and we have encoder.* keys, it is
    # likely the alternative scheme.
    if "encoder.conv_in.bias" not in keys and any(k.startswith("encoder.") for k in keys):
        return True

    return False


def _sub_path_to_bf16(sub_path: str) -> str:
    """
    Map a residual-block sub-path from the fp32 scheme to the bf16 scheme.

    The fp32 scheme uses sub-indices 0, 2, 3, 6 inside residual blocks while
    the bf16 scheme uses named layers (norm1, conv1, norm2, conv2).  This
    function handles that translation.

    Examples:
        "residual.0.gamma" -> "norm1.gamma"
        "residual.2.weight" -> "conv1.weight"
        "residual.3.gamma" -> "norm2.gamma"
        "residual.6.bias" -> "conv2.bias"
        "shortcut.bias" -> "conv_shortcut.bias"
    """
    if m := re.match(r"^residual\.0\.gamma$", sub_path):
        return "norm1.gamma"
    if m := re.match(r"^residual\.2\.(bias|weight)$", sub_path):
        return f"conv1.{m.group(1)}"
    if m := re.match(r"^residual\.3\.gamma$", sub_path):
        return "norm2.gamma"
    if m := re.match(r"^residual\.6\.(bias|weight)$", sub_path):
        return f"conv2.{m.group(1)}"
    if m := re.match(r"^shortcut\.(bias|weight)$", sub_path):
        return f"conv_shortcut.{m.group(1)}"
    return sub_path


def _decoder_upsamples_group_for_idx(idx: int) -> Tuple[int, int, bool]:
    """
    Return the (up_block_index, resnets_sub_index, is_upsampler) for a given
    fp32 `decoder.upsamples` index.

    Mapping (from `wan22_VAE_findings.md`):

        fp32 idx  |  up_block  |  resnets sub  |  is_upsampler
        ----------+------------+---------------+--------------
        0         |     0      |       0       |     False
        1         |     0      |       1       |     False
        2         |     0      |       2       |     False
        3         |     0      |      n/a      |      True
        4         |     1      |       0       |     False
        5         |     1      |       1       |     False
        6         |     1      |       2       |     False
        7         |     1      |      n/a      |      True
        8         |     2      |       0       |     False
        9         |     2      |       1       |     False
        10        |     2      |       2       |     False
        11        |     2      |      n/a      |      True
        12        |     3      |       0       |     False
        13        |     3      |       1       |     False
        14        |     3      |       2       |     False
    """
    # Upsampler indices: 3, 7, 11
    upsampler_map = {3: 0, 7: 1, 11: 2}
    if idx in upsampler_map:
        return (upsampler_map[idx], -1, True)

    # Residual groups: each entry is (start_idx, end_idx, up_block, sub_start)
    residual_groups: List[Tuple[int, int, int, int]] = [
        (0, 2, 0, 0),
        (4, 4, 1, 0),
        (5, 6, 1, 1),
        (8, 10, 2, 0),
        (12, 14, 3, 0),
    ]
    for start, end, block, sub_start in residual_groups:
        if start <= idx <= end:
            return (block, idx - start + sub_start, False)

    return (-1, -1, False)


def _fp32_to_bf16_rename(key: str) -> str:
    """
    Apply the fp32 -> bf16 tensor name mapping to a single key.

    The mapping is based on the systematic rules documented in
    `wan22_VAE_findings.md`.  Returns the renamed key.  If no rule matches,
    the original key is returned unchanged.
    """
    # ---- 1. Top-level heads & quantization layers ----
    if m := re.match(r"^encoder\.conv1\.(bias|weight)$", key):
        return f"encoder.conv_in.{m.group(1)}"
    if key == "encoder.head.0.gamma":
        return "encoder.norm_out.gamma"
    if m := re.match(r"^encoder\.head\.2\.(bias|weight)$", key):
        return f"encoder.conv_out.{m.group(1)}"
    if m := re.match(r"^decoder\.conv1\.(bias|weight)$", key):
        return f"decoder.conv_in.{m.group(1)}"
    if key == "decoder.head.0.gamma":
        return "decoder.norm_out.gamma"
    if m := re.match(r"^decoder\.head\.2\.(bias|weight)$", key):
        return f"decoder.conv_out.{m.group(1)}"
    if key == "conv1.bias":
        return "quant_conv.bias"
    if key == "conv1.weight":
        return "quant_conv.weight"
    if key == "conv2.bias":
        return "post_quant_conv.bias"
    if key == "conv2.weight":
        return "post_quant_conv.weight"

    # ---- 2. Encoder downsample indices (0..10) ----
    if m := re.match(r"^encoder\.downsamples\.(\d+)\.residual\.0\.gamma$", key):
        return f"encoder.down_blocks.{m.group(1)}.norm1.gamma"
    if m := re.match(r"^encoder\.downsamples\.(\d+)\.residual\.2\.(bias|weight)$", key):
        return f"encoder.down_blocks.{m.group(1)}.conv1.{m.group(2)}"
    if m := re.match(r"^encoder\.downsamples\.(\d+)\.residual\.3\.gamma$", key):
        return f"encoder.down_blocks.{m.group(1)}.norm2.gamma"
    if m := re.match(r"^encoder\.downsamples\.(\d+)\.residual\.6\.(bias|weight)$", key):
        return f"encoder.down_blocks.{m.group(1)}.conv2.{m.group(2)}"
    if m := re.match(r"^encoder\.downsamples\.(\d+)\.shortcut\.(bias|weight)$", key):
        return f"encoder.down_blocks.{m.group(1)}.conv_shortcut.{m.group(2)}"
    if m := re.match(r"^encoder\.downsamples\.(\d+)\.resample\.1\.(bias|weight)$", key):
        return f"encoder.down_blocks.{m.group(1)}.resample.1.{m.group(2)}"
    if m := re.match(r"^encoder\.downsamples\.(\d+)\.time_conv\.(bias|weight)$", key):
        return f"encoder.down_blocks.{m.group(1)}.time_conv.{m.group(2)}"

    # ---- 3. Encoder mid-block
    if m := re.match(r"^encoder\.middle\.0\.residual\.0\.gamma$", key):
        return "encoder.mid_block.resnets.0.norm1.gamma"
    if m := re.match(r"^encoder\.middle\.0\.residual\.2\.(bias|weight)$", key):
        return f"encoder.mid_block.resnets.0.conv1.{m.group(2)}"
    if m := re.match(r"^encoder\.middle\.0\.residual\.3\.gamma$", key):
        return "encoder.mid_block.resnets.0.norm2.gamma"
    if m := re.match(r"^encoder\.middle\.0\.residual\.6\.(bias|weight)$", key):
        return f"encoder.mid_block.resnets.0.conv2.{m.group(2)}"
    if key == "encoder.middle.1.norm.gamma":
        return "encoder.mid_block.attentions.0.norm.gamma"
    if m := re.match(r"^encoder\.middle\.1\.proj\.(bias|weight)$", key):
        return f"encoder.mid_block.attentions.0.proj.{m.group(1)}"
    if m := re.match(r"^encoder\.middle\.1\.to_qkv\.(bias|weight)$", key):
        return f"encoder.mid_block.attentions.0.to_qkv.{m.group(1)}"
    if m := re.match(r"^encoder\.middle\.2\.residual\.0\.gamma$", key):
        return "encoder.mid_block.resnets.1.norm1.gamma"
    if m := re.match(r"^encoder\.middle\.2\.residual\.2\.(bias|weight)$", key):
        return f"encoder.mid_block.resnets.1.conv1.{m.group(2)}"
    if m := re.match(r"^encoder\.middle\.2\.residual\.3\.gamma$", key):
        return "encoder.mid_block.resnets.1.norm2.gamma"
    if m := re.match(r"^encoder\.middle\.2\.residual\.6\.(bias|weight)$", key):
        return f"encoder.mid_block.resnets.1.conv2.{m.group(2)}"

    # ---- 4. Decoder mid-block (same structure as encoder) ----
    if m := re.match(r"^decoder\.middle\.0\.residual\.0\.gamma$", key):
        return "decoder.mid_block.resnets.0.norm1.gamma"
    if m := re.match(r"^decoder\.middle\.0\.residual\.2\.(bias|weight)$", key):
        return f"decoder.mid_block.resnets.0.conv1.{m.group(2)}"
    if m := re.match(r"^decoder\.middle\.0\.residual\.3\.gamma$", key):
        return "decoder.mid_block.resnets.0.norm2.gamma"
    if m := re.match(r"^decoder\.middle\.0\.residual\.6\.(bias|weight)$", key):
        return f"decoder.mid_block.resnets.0.conv2.{m.group(2)}"
    if key == "decoder.middle.1.norm.gamma":
        return "decoder.mid_block.attentions.0.norm.gamma"
    if m := re.match(r"^decoder\.middle\.1\.proj\.(bias|weight)$", key):
        return f"decoder.mid_block.attentions.0.proj.{m.group(1)}"
    if m := re.match(r"^decoder\.middle\.1\.to_qkv\.(bias|weight)$", key):
        return f"decoder.mid_block.attentions.0.to_qkv.{m.group(1)}"
    if m := re.match(r"^decoder\.middle\.2\.residual\.0\.gamma$", key):
        return "decoder.mid_block.resnets.1.norm1.gamma"
    if m := re.match(r"^decoder\.middle\.2\.residual\.2\.(bias|weight)$", key):
        return f"decoder.mid_block.resnets.1.conv1.{m.group(2)}"
    if m := re.match(r"^decoder\.middle\.2\.residual\.3\.gamma$", key):
        return "decoder.mid_block.resnets.1.norm2.gamma"
    if m := re.match(r"^decoder\.middle\.2\.residual\.6\.(bias|weight)$", key):
        return f"decoder.mid_block.resnets.1.conv2.{m.group(2)}"

    # ---- 5. Decoder upsamples ----
    if m := re.match(r"^decoder\.upsamples\.(\d+)\.(.+)$", key):
        idx = int(m.group(1))
        rest = m.group(2)
        up_block, sub_idx, is_upsampler = _decoder_upsamples_group_for_idx(idx)
        if is_upsampler:
            # `resample.1.*` or `time_conv.*` stays under `upsamplers.0`.
            return f"decoder.up_blocks.{up_block}.upsamplers.0.{rest}"
        elif up_block >= 0:
            # residual / shortcut -> `up_blocks.N.resnets.{sub_idx}.{bf16-sub-path}`
            new_rest = _sub_path_to_bf16(rest)
            return f"decoder.up_blocks.{up_block}.resnets.{sub_idx}.{new_rest}"
        # Fallback: no group matched, return unchanged.
        return key

    return key


def normalize_vae_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Return a new state dict whose keys follow the standard
    AutoencoderKLWan naming convention.

    If the input already uses the standard naming, it is returned unchanged
    (by reference).  Otherwise the fp32 -> bf16 mapping (see
    `wan22_VAE_findings.md`) is applied and a new dict is returned.
    Tensor values are preserved as-is.
    """
    if not _key_matches_alternative_naming(state_dict):
        return state_dict

    renamed: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        renamed[_fp32_to_bf16_rename(key)] = value

    return renamed


def detect_alternative_vae_naming(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Public alias for `_key_matches_alternative_naming`."""
    return _key_matches_alternative_naming(state_dict)
