"""
TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training
(https://arxiv.org/abs/2501.04765)

Backend-only port for the Wan 2.2 14B (diffusers ``WanTransformer3DModel``) used by
ai-toolkit. Assumes bf16/fp32 (non-quantized) weights.

Routing
-------
A *route* ``r_i -> j`` is defined by a start layer ``i`` and an end layer ``j``. During
training, a fixed fraction of the tokens is randomly *routed* (bypassed) so it skips blocks
``i .. j`` (inclusive); the remaining ("kept") tokens are processed normally through that
window. At layer ``j`` the routed tokens (which still carry their pre-window activations)
are merged back in, so the loss is still computed uniformly over the whole sequence - no
extra parameters, no architectural change, training-only.

RoPE correctness
----------------
Wan applies rotary embeddings from a *per-position* table (``rotary_emb`` is a
``(freqs_cos, freqs_sin)`` pair of shape ``(1, L, 1, head_dim)``). Because we select the
matching rows of that table together with the kept hidden states, every kept token keeps
its *true* ``(t, h, w)`` position. That makes routing correct for **any** selection mode
(``random`` / ``stride`` / ``contiguous``) and either granularity (token or whole-frame) -
unlike implementations that renumber the kept tokens from 0, which silently compress the
temporal spacing for strided selections.

fp32 front / tail layers
------------------------
Two independent precision options, both applied during training **and** sampling (they are
pure precision choices, not a training trick):

* ``fp32_front`` - keep the "front" of the model in fp32: the patch embedding, the
  condition embedder (time/text/image), the RoPE buffers and transformer block 0.
* ``fp32_last_layers`` - keep the final ``N`` transformer blocks **and** the output head
  (``norm_out`` / ``proj_out`` / ``scale_shift_table``) in fp32. This addresses the paper's
  note about instability when very few blocks remain after the route.

The parameters of the selected layers are upcast to fp32 **once at install time** (never
downcasting the rest of the bf16 model). fp32 attention cannot use the fp16/bf16 flash
kernel, so every fp32 layer's attention is run through torch
``scaled_dot_product_attention`` (SDPA), which auto-selects the memory-efficient kernel
for fp32 (O(N) memory, no full ``seq x seq`` score matrix) and is torch.compile-safe -
see :func:`_fp32_attention_ctx`. This avoids the OOM that torch ``flex_attention`` hits
when it is nested inside a whole-model compile and falls back to its un-fused dense
(math) kernel, which materialises the full score matrix.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace as _dataclass_replace
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn

# diffusers pieces we reuse so the TREAD forward stays a faithful copy of the stock
# ``WanTransformer3DModel.forward``.
try:
    from diffusers.models.attention_dispatch import (
        attention_backend as _attention_backend_ctx,
        _CAN_USE_FLEX_ATTN as _FLEX_AVAILABLE,
    )
except Exception:  # pragma: no cover - very old diffusers
    _attention_backend_ctx = None
    _FLEX_AVAILABLE = False

try:
    from diffusers.models.transformers.transformer_wan import Transformer2DModelOutput
except Exception:  # pragma: no cover
    Transformer2DModelOutput = None


@dataclass
class TREADConfig:
    """Resolved TREAD settings for a single transformer.

    ``end_layer`` may be ``None`` until it is resolved against the actual number of blocks
    (default: ``num_layers - 4``), matching the paper's "absolute" extrapolation guideline.
    """

    enabled: bool = False
    start_layer: int = 2
    end_layer: Optional[int] = None  # resolved to num_layers - 4 if None
    # Fraction of tokens/frames KEPT (processed) inside the window. The routed fraction is
    # ``1 - keep_ratio``. Paper default selection rate is 0.5 (== keep_ratio 0.5).
    keep_ratio: float = 0.5
    # Selection pattern: 'random' (paper-faithful per-step variance) | 'contiguous' | 'stride'
    mode: str = "random"
    # Routing granularity: 'token' (individual sequence tokens, as in the paper) or
    # 'frame' (route whole frames = all spatial tokens of a video frame together).
    granularity: str = "token"
    # Keep the front of the model (patch_embedding + condition_embedder + rope + block 0)
    # in fp32.
    fp32_front: bool = False
    # Number of final transformer blocks (plus the output head) to run in fp32 (0 = off).
    fp32_last_layers: int = 0
    # Optional explicit list of additional block indices to run in fp32 (in addition to
    # the front/tail). Lets you keep an arbitrary set of layers in fp32.
    fp32_layers: Tuple[int, ...] = ()

    def resolved_end_layer(self, num_layers: int) -> int:
        if self.end_layer is None:
            return max(self.start_layer, num_layers - 4)
        return min(max(self.end_layer, self.start_layer), num_layers - 1)

    def fp32_block_indices(self, num_layers: int) -> set:
        """Set of transformer block indices that run in fp32 (front/tail + explicit list)."""
        idx = set()
        if self.fp32_front:
            idx.add(0)
        if self.fp32_last_layers > 0:
            idx.update(range(max(0, num_layers - self.fp32_last_layers), num_layers))
        for i in self.fp32_layers:
            if 0 <= i < num_layers:
                idx.add(i)
        return idx

    @property
    def active(self) -> bool:
        return (
            self.enabled
            or self.fp32_front
            or self.fp32_last_layers > 0
            or len(self.fp32_layers) > 0
        )

    @property
    def has_fp32(self) -> bool:
        # True if any layer runs in fp32 (front / tail / explicit list). Used at load time
        # to decide whether to apply the selective fp32 cast + weight refill. (The fp32
        # attention itself is run via SDPA, not flex - see :func:`_fp32_attention_ctx`.)
        return (
            self.fp32_front
            or self.fp32_last_layers > 0
            or len(self.fp32_layers) > 0
        )


def _parse_fp32_layers(value) -> Tuple[int, ...]:
    """Coerce an ``fp32_layers`` config value into a tuple of block indices.

    Accepts a list/tuple of ints, a single int, or a comma-separated string (``"0,17,38"``).
    """
    if value is None:
        return ()
    if isinstance(value, str):
        value = [v for v in value.replace(";", ",").split(",") if v.strip() != ""]
    if isinstance(value, (int,)):
        return (int(value),)
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return tuple(out)
    return ()


def parse_tread_config(model_kwargs: Optional[dict], expert: Optional[str] = None) -> TREADConfig:
    """Build a :class:`TREADConfig` from ``model.model_kwargs``.

    Accepts either a nested ``tread:`` mapping or flat ``tread_*`` keys::

        model_kwargs:
          tread:
            enabled: true
            start_layer: 2
            end_layer: 36         # optional; defaults to num_layers - 4
            keep_ratio: 0.5
            mode: random          # random | contiguous | stride
            granularity: token    # token | frame
            fp32_front: true      # keep patch_embedding + condition_embedder + block 0 in fp32
            fp32_last_layers: 2   # run the last N blocks (+ output head) in fp32
            fp32_layers: [0, 17]  # optional explicit block indices to run in fp32

    Per-expert overrides
    ---------------------
    For dual-expert (high/low-noise) models, any parameter may be overridden per expert by
    passing ``expert="high"`` or ``expert="low"``. Per-expert values are looked up in this
    order (first match wins), falling back to the global setting when undefined::

        1. model_kwargs.tread_<expert>          (flat dict, e.g. tread_high: {...})
        2. model_kwargs.tread_<expert>_<key>    (flat key,  e.g. tread_high_keep_ratio: 0.7)
        3. model_kwargs.tread.<expert>.<key>    (nested,    e.g. tread: {high: {keep_ratio: 0.7}})
        4. model_kwargs.tread.<key>             (global nested)
        5. model_kwargs.tread_<key>             (global flat)

    So a per-expert block only needs to define the parameters it wants to differ; everything
    else is inherited from the global ``tread`` settings.
    """
    # NOTE: this parses the *base* routing settings only. Per-timestep overrides live in
    # ``timestep_overrides`` (see :func:`parse_tread_ranges`) and are resolved at forward
    # time from the batch's global timestep.
    mk = model_kwargs or {}
    g_nested = mk.get("tread", None)
    g_nested = g_nested if isinstance(g_nested, dict) else {}

    e_nested = {}
    e_flat_prefix = None
    if expert:
        e_nested = mk.get(f"tread_{expert}", None)
        e_nested = e_nested if isinstance(e_nested, dict) else {}
        # nested per-expert block: tread.<expert>
        sub = g_nested.get(expert, None)
        if isinstance(sub, dict):
            e_nested = {**sub, **e_nested}
        e_flat_prefix = f"tread_{expert}_"

    def get(key, default):
        # 1. expert nested (tread_<expert> / tread.<expert>)
        if key in e_nested:
            return e_nested[key]
        # 2. expert flat (tread_<expert>_<key>)
        if e_flat_prefix is not None:
            flat = e_flat_prefix + key
            if flat in mk:
                return mk[flat]
        # 3. global nested (tread.<key>)
        if key in g_nested:
            return g_nested[key]
        # 4. global flat (tread_<key>)
        flat = f"tread_{key}"
        if flat in mk:
            return mk[flat]
        return default

    enabled = bool(get("enabled", False))
    if "tread_enabled" in mk:
        enabled = bool(mk["tread_enabled"])

    mode = str(get("mode", "random")).lower()
    if mode not in ("random", "contiguous", "stride"):
        mode = "random"

    granularity = str(get("granularity", "token")).lower()
    if granularity not in ("token", "frame"):
        granularity = "token"

    end_layer = get("end_layer", None)
    if end_layer is not None:
        end_layer = int(end_layer)

    return TREADConfig(
        enabled=enabled,
        start_layer=int(get("start_layer", 2)),
        end_layer=end_layer,
        keep_ratio=float(get("keep_ratio", 0.5)),
        mode=mode,
        granularity=granularity,
        fp32_front=bool(get("fp32_front", False)),
        fp32_last_layers=int(get("fp32_last_layers", 0)),
        fp32_layers=_parse_fp32_layers(get("fp32_layers", None)),
    )


# Fields a per-timestep range entry may override (routing only - the fp32 front/tail
# settings are static precision choices applied at load time, not per-timestep).
TREAD_ROUTING_FIELDS = ("enabled", "start_layer", "end_layer", "keep_ratio", "mode", "granularity")


def parse_tread_ranges(model_kwargs: Optional[dict], expert: Optional[str] = None) -> list:
    """Parse the per-timestep TREAD overrides (``timestep_overrides``) for one expert scope.

    Looked up per expert first (``tread_<expert>.timestep_overrides``,
    ``tread_<expert>_timestep_overrides`` or ``tread.<expert>.timestep_overrides``);
    when the expert scope does not define the key, the global ``tread.timestep_overrides``
    (or flat ``tread_timestep_overrides``) is used. An expert that defines the key
    (even as an empty list) uses its own list exclusively.

    Each entry is a dict::

        timestep_overrides:
          - start_timestep: 1000   # GLOBAL timesteps (0-1000): high expert ~900-1000, low 0-900
            end_timestep: 950      # matches [start, end) (or (end, start] when descending)
            keep_ratio: 0.3        # any routing field may be overridden; omitted = inherit base
            enabled: true

    Ranges use **global** timesteps - the high-noise expert only ever sees ~900-1000 and the
    low-noise expert 0-900, so a range like 1000-950 can only match on the high expert.

    Ranges are **opt-in**: when a scope defines at least one range, TREAD routing only
    applies inside matched ranges - timesteps outside every range run with routing
    disabled. A scope with no ranges of its own falls back to the global list (so a
    global-only range set also restricts the other expert to those ranges).
    """
    mk = model_kwargs or {}
    g_nested = mk.get("tread", None)
    g_nested = g_nested if isinstance(g_nested, dict) else {}

    def _as_list(value):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            return None
        return [item for item in value if isinstance(item, dict)]

    if expert:
        e_nested = mk.get(f"tread_{expert}", None)
        e_nested = e_nested if isinstance(e_nested, dict) else {}
        sub = g_nested.get(expert, None)
        if isinstance(sub, dict):
            e_nested = {**sub, **e_nested}
        for candidate in (
            e_nested.get("timestep_overrides", None),
            mk.get(f"tread_{expert}_timestep_overrides", None),
        ):
            parsed = _as_list(candidate)
            if parsed is not None:
                return parsed

    # Global scope
    for candidate in (
        g_nested.get("timestep_overrides", None),
        mk.get("tread_timestep_overrides", None),
    ):
        parsed = _as_list(candidate)
        if parsed is not None:
            return parsed
    return []


def _timestep_in_range(t: float, start: float, end: float) -> bool:
    """Range semantics identical to the per-timestep range loss overrides."""
    if start >= end:
        # Descending range (e.g. 1000-950): matches (end, start]
        return t <= start and t > end
    # Ascending range (e.g. 0-500): matches [start, end)
    return t >= start and t < end


def resolve_tread_config_for_timestep(
    base_cfg: TREADConfig,
    ranges: list,
    timesteps: torch.Tensor,
) -> TREADConfig:
    """Resolve the effective :class:`TREADConfig` for a batch from its timesteps.

    ``timesteps`` are the model's input timesteps in **global** space (0-1000), one per
    batch item. The first range matching the batch timestep wins; its non-null routing
    fields override the base config (everything else is inherited).

    Ranges are **opt-in**: when at least one range is defined, TREAD routing only applies
    inside matched ranges. A timestep that matches NO range runs with routing disabled
    (the base config with ``enabled=False``) - e.g. a single range 1000-875 means TREAD is
    NOT used at all for 875-0. With no ranges defined, the base config applies unchanged
    at every timestep. The fp32 front/tail precision is unaffected either way (it is a
    static load-time choice, not part of the per-timestep routing).

    With TREAD routing active, ``train.force_same_timestep_per_batch`` guarantees every
    batch item shares one timestep, so a single resolution covers the whole batch. If the
    invariant is ever broken (different timesteps in one batch), a warning is printed and
    the first item's settings are used for the batch.
    """
    if base_cfg is None or not ranges:
        return base_cfg

    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps])
    ts = timesteps.detach().flatten().to(torch.float32).cpu()
    first = float(ts[0])

    resolved = base_cfg
    matched = False
    for r in ranges:
        start = float(r.get("start_timestep", 0) or 0)
        end = float(r.get("end_timestep", 0) or 0)
        if _timestep_in_range(first, start, end):
            matched = True
            overrides = {k: r[k] for k in TREAD_ROUTING_FIELDS if r.get(k) is not None}
            if overrides:
                resolved = _dataclass_replace(base_cfg, **overrides)
            break

    if not matched:
        # Opt-in semantics: with at least one range defined, timesteps outside every
        # range do not use TREAD routing (fp32 front/tail precision is still applied -
        # it is static and lives on the base config, which this replace preserves).
        resolved = _dataclass_replace(base_cfg, enabled=False)

    if ts.numel() > 1 and not bool(torch.all(ts == ts[0]).item()):
        print(
            "[TREAD] WARNING: batch items have different timesteps, but TREAD resolves its "
            "settings once per batch. Using the first item's settings for the whole batch. "
            "Enable train.force_same_timestep_per_batch for correct per-timestep routing."
        )
    return resolved


def tread_routing_may_be_active(model_kwargs: Optional[dict]) -> bool:
    """True if TREAD token routing can be active at any timestep.

    Checks the global and per-expert base configs plus every per-timestep range scope, since
    a range entry may enable routing even when the base config leaves it off. (fp32-only
    TREAD - no routing - does NOT count: it is static and needs no per-timestep handling.)
    """
    mk = model_kwargs or {}
    for expert in (None, "high", "low"):
        if parse_tread_config(mk, expert=expert).enabled:
            return True
        for r in parse_tread_ranges(mk, expert=expert):
            if r.get("enabled") is True:
                return True
    return False


def tread_fp32_param_prefixes(cfg: TREADConfig, num_layers: int):
    """Return the set of dotted module-name prefixes TREAD keeps in fp32.

    This is the single source of truth for *which* modules run in fp32. It is used both
    at load time (to avoid downcasting an fp32 checkpoint's front/tail weights to bf16)
    and by :func:`apply_tread` (the runtime upcast). The entries match diffusers module /
    state-dict names, e.g. ``patch_embedding``, ``condition_embedder``, ``rope``,
    ``blocks.0``, ``blocks.38``, ``norm_out``, ``proj_out`` and ``scale_shift_table``.
    """
    prefixes = set()
    # Every block that runs in fp32 (front block 0, the tail, and any explicit fp32_layers).
    for i in cfg.fp32_block_indices(num_layers):
        prefixes.add(f"blocks.{i}")
    if cfg.fp32_front:
        prefixes.update({"patch_embedding", "condition_embedder", "rope"})
    if cfg.fp32_last_layers > 0:
        prefixes.update({"norm_out", "proj_out", "scale_shift_table"})
    return prefixes


def is_tread_fp32_name(name: str, prefixes) -> bool:
    """True if a state-dict key / module name belongs to a TREAD fp32 module."""
    if not prefixes:
        return False
    for p in prefixes:
        if name == p or name.startswith(p + "."):
            return True
    return False


def selective_cast_model(transformer: nn.Module, cfg: TREADConfig, base_dtype: torch.dtype) -> nn.Module:
    """Upcast the TREAD fp32 modules to fp32 and every other param/buffer to ``base_dtype``.

    This is the load-time counterpart of :func:`apply_tread`'s upcast. Run it *after* the
    weights are in memory and *before* any blanket ``.to(base_dtype)`` downcast, so an fp32
    checkpoint's front/tail weights are never rounded to bf16. For a bf16 checkpoint the
    upcast of the fp32 modules is lossless (bf16 -> fp32).
    """
    num_layers = len(transformer.blocks)
    prefixes = tread_fp32_param_prefixes(cfg, num_layers)
    if not prefixes:
        transformer.to(base_dtype)
        return transformer
    for name, param in list(transformer.named_parameters()):
        param.data = param.data.to(torch.float32 if is_tread_fp32_name(name, prefixes) else base_dtype)
    for name, buf in list(transformer.named_buffers()):
        buf.data = buf.data.to(torch.float32 if is_tread_fp32_name(name, prefixes) else base_dtype)
    return transformer


def cast_state_dict_for_tread(state_dict: dict, cfg: TREADConfig, num_layers: int, base_dtype: torch.dtype) -> dict:
    """Cast a diffusers-named state dict: fp32 for TREAD front/tail keys, ``base_dtype`` else."""
    prefixes = tread_fp32_param_prefixes(cfg, num_layers)
    if not prefixes:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.to(torch.float32 if is_tread_fp32_name(k, prefixes) else base_dtype)
        else:
            out[k] = v
    return out


def load_fp32_target_tensors(safetensors_path: str, cfg: TREADConfig, num_layers: int) -> dict:
    """Read only the TREAD fp32 tensors from a diffusers-named safetensors file, as fp32.

    Used to refill the front/tail weights at full precision when the model was loaded in
    the base dtype (e.g. ``from_pretrained(torch_dtype=bf16)``), preserving an fp32
    checkpoint's precision for exactly the modules TREAD runs in fp32.
    """
    from safetensors import safe_open
    prefixes = tread_fp32_param_prefixes(cfg, num_layers)
    if not prefixes:
        return {}
    out = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            if key in ("class_version", "__metadata__"):
                continue
            if is_tread_fp32_name(key, prefixes):
                out[key] = f.get_tensor(key).to(torch.float32)
    return out


def refill_fp32_target_weights(transformer: nn.Module, tensors: dict) -> int:
    """Copy fp32 tensors (from :func:`load_fp32_target_tensors`) into the model in place.

    Assigning an fp32 tensor to a param/buffer's ``.data`` also rebinds that tensor to fp32,
    so the module stays in fp32 after the copy. Returns the number of tensors copied.
    """
    if not tensors:
        return 0
    params = dict(transformer.named_parameters())
    buffers = dict(transformer.named_buffers())
    copied = 0
    for key, value in tensors.items():
        target = params.get(key, buffers.get(key))
        if target is None:
            continue
        target.data = value.to(target.device)
        copied += 1
    return copied


# Default LayerNorm/attention-norm epsilon for blocks kept in fp32 by TREAD.
# fp32 resolves 1e-8 exactly; in bf16 any eps below ~1e-5 is rounded to 0 by the
# variance addition (``x / sqrt(var + eps)``) and can cause division-by-zero / NaNs.
TREAD_FP32_EPS_DEFAULT = 1e-8


def apply_wan_transformer_eps(transformer, global_eps, fp32_eps=None, label: str = "transformer") -> bool:
    """Apply a per-compute-dtype eps override to a Wan transformer's norm layers.

    Blocks whose parameters run in fp32 (TREAD ``fp32_front`` / ``fp32_last_layers`` /
    ``fp32_layers``) get ``fp32_eps`` (the explicit ``wan_transformer_fp32_eps`` if set,
    else the automatic :data:`TREAD_FP32_EPS_DEFAULT` = 1e-8, which fp32 resolves
    exactly); every other block gets ``global_eps`` (or keeps the model's default when
    ``global_eps`` is ``None``). A single global eps cannot serve both dtypes: it is too
    small to matter in bf16 and needlessly coarse in fp32.

    ``global_eps`` (``wan_transformer_eps``) is a bf16-oriented value and is deliberately
    NOT leaked into fp32 blocks - e.g. a global ``1e-4`` (needed for bf16 stability) must
    not coarsen the fp32 norms, which keep their automatic ``1e-8`` (override with
    ``wan_transformer_fp32_eps`` if you ever want a different fp32 value).

    Must be called *after* the TREAD selective fp32 cast, so each block's compute dtype
    is already final. LoRA has no norm layers and is not injected yet at load time; the
    dtype probe still skips any ``lora`` parameters for safety. Returns ``True`` if any
    block's eps was changed.
    """
    if fp32_eps is None:
        fp32_eps = TREAD_FP32_EPS_DEFAULT

    changed = False
    for block in transformer.blocks:
        try:
            param_dtype = next(
                p.dtype for n, p in block.named_parameters() if "lora" not in n
            )
        except StopIteration:
            param_dtype = torch.float32

        block_eps = fp32_eps if param_dtype == torch.float32 else global_eps
        if block_eps is None:
            continue
        # LayerNorm layers
        block.norm1.eps = block_eps
        if hasattr(block.norm2, "eps"):
            block.norm2.eps = block_eps
        block.norm3.eps = block_eps
        # Attention norms (self-attention)
        block.attn1.norm_q.eps = block_eps
        block.attn1.norm_k.eps = block_eps
        # Attention norms (cross-attention)
        block.attn2.norm_q.eps = block_eps
        block.attn2.norm_k.eps = block_eps
        changed = True

    if changed and global_eps is not None:
        # Report the bulk (non-fp32) value as the model's config eps.
        transformer.config.eps = global_eps
    return changed


# Cached handle to the ai-toolkit custom attention module (for the flex override).
_wan_attn_module = None


def _get_wan_attn():
    """Lazily import the ai-toolkit Wan attention module (avoids import cycles)."""
    global _wan_attn_module
    if _wan_attn_module is None:
        try:
            from toolkit.models.wan21 import wan_attn as m
            _wan_attn_module = m
        except Exception:
            _wan_attn_module = False
    return _wan_attn_module if _wan_attn_module else None


@contextmanager
def _fp32_attention_ctx():
    """Run the block(s) inside this context with SDPA attention (fp32-friendly).

    fp32 attention cannot use the fp16/bf16 flash kernel. We therefore route the fp32
    layers through ``torch.nn.functional.scaled_dot_product_attention`` (the diffusers
    "native" backend / the ai-toolkit "sdpa" backend), which auto-selects the
    *memory-efficient* kernel for fp32 (O(N) memory, no full ``seq x seq`` score matrix)
    and is torch.compile-safe.

    This is what fixes the OOM from the previous flex-based approach: torch
    ``flex_attention`` must be compiled to be efficient, and when it is nested inside a
    whole-model ``torch.compile`` (which is what happens when the per-block compile can't
    find the blocks) it falls back to its un-fused dense "math" kernel, which materialises
    the full score matrix and OOMs on long video sequences.

    WHY TWO THINGS ARE SET
    ----------------------
    ai-toolkit can run Wan self/cross-attention through two independent processors, each
    resolving its own kernel:

    1. **Stock diffusers ``WanAttnProcessor``** (the default for the standard Wan 2.2 14B
       training path). It calls ``dispatch_attention_fn(..., backend=None)``; when the
       backend is ``None`` diffusers falls back to the *active backend* in
       ``_AttentionBackendRegistry``. The ``attention_backend("native")`` context manager
       below sets exactly that active backend, so the stock processor is routed to SDPA
       while the context is open.

    2. **Optional ai-toolkit ``WanAttnProcessor2_0``** (installed for the softcapping /
       I2V-adapter paths). It does NOT use diffusers' dispatch; instead it reads the
       module-global ``_attention_config['train_backend'/'sample_backend']``. We therefore
       also set that dict to ``'sdpa'`` for the duration of the context. (If attention
       softcapping is active, that processor transparently upgrades to flex_attention,
       since softcapping is implemented as a flex ``score_mod``.)

    Both are restored on exit, so only the fp32 layers use SDPA while every other layer
    keeps the user-configured backend (flash / sdpa / ...).
    """
    aitk = _get_wan_attn()
    prev_aitk = None
    if aitk is not None:
        prev_aitk = (aitk._attention_config['train_backend'], aitk._attention_config['sample_backend'])
        aitk._attention_config['train_backend'] = 'sdpa'
        aitk._attention_config['sample_backend'] = 'sdpa'
    try:
        if _attention_backend_ctx is not None:
            with _attention_backend_ctx("native"):
                yield
        else:  # pragma: no cover - very old diffusers
            yield
    finally:
        if prev_aitk is not None:
            aitk._attention_config['train_backend'], aitk._attention_config['sample_backend'] = prev_aitk


def _select_keep_indices(
    total_tokens: int,
    keep_ratio: float,
    mode: str,
    device,
    frame_size: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Return the (sorted) indices of the tokens that are *kept* (processed) in the window.

    Returns ``None`` when nothing should be routed (keep_ratio >= 1).

    * ``frame_size is None``  -> token granularity: select ``total_tokens * keep_ratio``
      individual tokens.
    * ``frame_size is int``   -> frame granularity: select whole frames (each ``frame_size``
      consecutive tokens) so that ``total_tokens // frame_size * keep_ratio`` frames are kept.

    The selection is re-drawn on every forward call, which provides the per-step variance
    the paper relies on (for the common video LoRA case of batch size 1 this is exactly
    per-sample random selection).
    """
    if keep_ratio >= 1.0:
        return None

    if frame_size is not None:
        num_groups = total_tokens // frame_size
        num_keep = max(1, int(round(num_groups * keep_ratio)))
        if num_keep >= num_groups:
            return None
        if mode == "contiguous":
            start = (num_groups - num_keep) // 2
            groups = torch.arange(start, start + num_keep, device=device, dtype=torch.long)
        elif mode == "stride":
            stride = max(1, num_groups // num_keep)
            groups = torch.arange(0, num_groups, stride, device=device, dtype=torch.long)[:num_keep]
        else:  # random
            groups = torch.randperm(num_groups, device=device, dtype=torch.long)[:num_keep].sort().values
        # Expand each selected frame ``f`` to its contiguous token range
        # [f*frame_size, (f+1)*frame_size). The kept tokens thus form whole-frame blocks.
        offsets = torch.arange(frame_size, device=device, dtype=torch.long)
        return (groups.unsqueeze(1) * frame_size + offsets.unsqueeze(0)).flatten()

    num_keep = max(1, int(round(total_tokens * keep_ratio)))
    if num_keep >= total_tokens:
        return None
    if mode == "contiguous":
        start = (total_tokens - num_keep) // 2
        return torch.arange(start, start + num_keep, device=device, dtype=torch.long)
    elif mode == "stride":
        stride = max(1, total_tokens // num_keep)
        return torch.arange(0, total_tokens, stride, device=device, dtype=torch.long)[:num_keep]
    else:  # random
        return torch.randperm(total_tokens, device=device, dtype=torch.long)[:num_keep].sort().values


def _block_dtype(block: nn.Module) -> torch.dtype:
    return next(block.parameters()).dtype


def _bulk_dtype(model: nn.Module) -> torch.dtype:
    """The model's bulk compute dtype (the first block that is *not* an fp32 TREAD layer).

    This is the dtype the pipeline/loss expects for the final output, independent of the
    input's dtype (which may be fp32 when latents are cached in fp32) and independent of any
    fp32 front/tail upcast. Falls back to block 0's dtype if the whole model is fp32.
    """
    for blk in model.blocks:
        d = _block_dtype(blk)
        if d != torch.float32:
            return d
    return _block_dtype(model.blocks[0])


def _tread_block_loop(
    model,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    cfg: TREADConfig,
    keep_idx: Optional[torch.Tensor],
    model_dtype: torch.dtype,
):
    """Run the transformer blocks with TREAD routing and fp32 front/tail blocks.

    Mirrors the stock block loop but inserts the route at ``[start_layer, end_layer]``.
    A block runs in fp32 (with forced flex attention) exactly when its own parameters are
    fp32 - i.e. block 0 when ``fp32_front`` is set, the final ``fp32_last_layers`` blocks,
    and any explicit ``fp32_layers``. ``rotary_emb`` is ``(freqs_cos, freqs_sin)`` of shape
    ``(1, L, 1, D)``.

    The loop returns the stream in the *last executed block's* dtype (no downcast here). The
    output head in :func:`tread_forward` performs the single final cast, so an fp32 tail block
    hands its full-precision output straight to the fp32 head (no fp32 -> bf16 -> fp32 roundtrip).
    ``model_dtype`` is kept for signature stability but the loop no longer casts to it.
    """
    blocks = model.blocks
    num_layers = len(blocks)

    use_gc = torch.is_grad_enabled() and getattr(model, "gradient_checkpointing", False)
    gc_func = getattr(model, "_gradient_checkpointing_func", None)

    def invoke(block, hs, enc, temb, rope):
        if use_gc and gc_func is not None:
            return gc_func(block, hs, enc, temb, rope)
        return block(hs, enc, temb, rope)

    def run_block(hs, enc, temb, rope, i):
        """Run block ``i`` on the given (possibly reduced) stream, matching its dtype."""
        block = blocks[i]
        bd = _block_dtype(block)
        # Cast the stream to the block's dtype (no-op when they already match). fp32 blocks
        # therefore run fully in fp32; bf16 blocks run in bf16.
        hs = hs.to(bd)
        enc = enc.to(bd)
        temb = temb.to(bd)
        rope = (rope[0].to(bd), rope[1].to(bd))
        # fp32 blocks have their forward wrapped at install time (see ``apply_tread``) so
        # their attention always runs via SDPA - including during gradient-checkpointing
        # recompute. No per-call context needed here.
        return invoke(block, hs, enc, temb, rope)

    enc = encoder_hidden_states
    temb = timestep_proj
    rope = rotary_emb

    # Fast path: no routing this forward (fp32 front/tail may still apply). This matches
    # the stock loop block-for-block, so with no fp32 layers it is bit-exact. Return in the
    # last block's dtype so an fp32 tail feeds its full-precision output to the fp32 head.
    if keep_idx is None:
        hs = hidden_states
        for i in range(num_layers):
            hs = run_block(hs, enc, temb, rope, i)
        return hs

    num_tokens = hidden_states.shape[1]
    start = max(0, min(cfg.start_layer, num_layers - 1))
    end = cfg.resolved_end_layer(num_layers)

    # Phase 1: layers before the window operate on the full sequence.
    hs = hidden_states
    for i in range(0, start):
        hs = run_block(hs, enc, temb, rope, i)

    # Phase 2: the routed window [start, end]. Kept tokens (with their true RoPE rows) go
    # through the window; routed tokens keep their pre-window activations and are merged
    # back via index_copy.
    hs_kept = hs[:, keep_idx]
    rope_kept = (rope[0][:, keep_idx], rope[1][:, keep_idx])
    for i in range(start, end + 1):
        hs_kept = run_block(hs_kept, enc, temb, rope_kept, i)
    # The window blocks may run in a different dtype than the pre-window state (e.g. an
    # fp32 window end block vs a bf16 pre-window state). Merge in the *higher* of the two
    # dtypes so a kept token's fp32 window output is not downcast when it is merged back in.
    merge_dtype = torch.promote_types(hs.dtype, hs_kept.dtype)
    hs = torch.index_copy(hs.to(merge_dtype), 1, keep_idx, hs_kept.to(merge_dtype))

    # Phase 3: layers after the window operate on the full sequence again.
    for i in range(end + 1, num_layers):
        hs = run_block(hs, enc, temb, rope, i)

    # Return in the last executed block's dtype; the output head performs the final cast so
    # an fp32 tail block keeps full precision through to the fp32 head.
    return hs


def tread_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs=None,
    _tread_cfg: Optional[TREADConfig] = None,
):
    """Drop-in replacement for ``WanTransformer3DModel.forward`` with TREAD support.

    A faithful copy of the stock forward; the front (patch/condition embedding), the block
    loop and the output head are made TREAD-aware.
    """
    cfg = _tread_cfg

    # Per-timestep overrides: resolve the effective routing config from this batch's
    # (global-space) timesteps. Training-only - sampling runs with grad disabled, so the
    # static base config (fp32 front/tail) is what applies there.
    if cfg is not None and torch.is_grad_enabled() and getattr(self, "_tread_ranges", None):
        cfg = resolve_tread_config_for_timestep(cfg, self._tread_ranges, timestep)

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    # The pipeline/loss expects the model's bulk dtype for the output, regardless of the
    # input dtype (which may be fp32 when latents/text embeds are cached in fp32) or any
    # fp32 front/tail upcast. (Not the input dtype: a bf16 model fed fp32 latents still
    # outputs bf16.)
    model_dtype = _bulk_dtype(self)

    # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()
    else:
        ts_seq_len = None

    # --- Front: RoPE + patch embedding + condition embedding --------------------------
    # Run the "front" of the model (everything up to and including block 0's inputs) in the
    # front's own dtype. When fp32_front is set the params are fp32, so the inputs (which may
    # arrive as fp32-cached latents or bf16) are upcast to fp32. Otherwise the front is in the
    # base dtype and any fp32-cached input is downcast to it - so fp32 latents feed a bf16-front
    # expert correctly (the cache dtype is the highest across experts; each expert handles its
    # own precision here).
    front_fp32 = cfg is not None and cfg.fp32_front
    if front_fp32:
        hs = hidden_states.float()
        rotary_emb = self.rope(hs)
        hidden_states = self.patch_embedding(hs).flatten(2).transpose(1, 2)
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep.float(),
            encoder_hidden_states.float(),
            encoder_hidden_states_image.float() if encoder_hidden_states_image is not None else None,
            timestep_seq_len=ts_seq_len,
        )
    else:
        front_dtype = next(self.patch_embedding.parameters()).dtype
        hidden_states = hidden_states.to(front_dtype)
        encoder_hidden_states = encoder_hidden_states.to(front_dtype)
        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )

    if ts_seq_len is not None:
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # --- Select the routed (kept) tokens for this forward -----------------------------
    # Routing is a training-only strategy: only active when gradients are enabled.
    keep_idx = None
    if cfg is not None and cfg.enabled and torch.is_grad_enabled() and cfg.keep_ratio < 1.0:
        total_tokens = hidden_states.shape[1]
        if cfg.granularity == "frame":
            keep_idx = _select_keep_indices(
                total_tokens, cfg.keep_ratio, cfg.mode, hidden_states.device,
                frame_size=post_patch_height * post_patch_width,
            )
        else:
            keep_idx = _select_keep_indices(total_tokens, cfg.keep_ratio, cfg.mode, hidden_states.device)

    # --- 4. Transformer blocks (TREAD-aware) ------------------------------------------
    hidden_states = _tread_block_loop(
        self, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, cfg or TREADConfig(),
        keep_idx, model_dtype,
    )

    # --- 5. Output norm, projection & unpatchify --------------------------------------
    if temb.ndim == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    head_fp32 = cfg is not None and cfg.fp32_last_layers > 0
    if head_fp32:
        # Keep the whole output head in fp32 (norm_out/proj_out/scale_shift_table are
        # upcast at install time). Do NOT downcast to model_dtype before proj_out.
        hidden_states = self.norm_out(hidden_states.float())
        hidden_states = (hidden_states * (1 + scale) + shift)
        hidden_states = self.proj_out(hidden_states)
    else:
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    # The pipeline / loss expect the model's compute dtype.
    if output.dtype != model_dtype:
        output = output.to(model_dtype)

    if not return_dict:
        return (output,)
    if Transformer2DModelOutput is not None:
        return Transformer2DModelOutput(sample=output)
    return {"sample": output}


def apply_tread(
    transformer: nn.Module,
    cfg: TREADConfig,
    label: str = "transformer",
    ranges: Optional[list] = None,
) -> bool:
    """Install the TREAD forward on a ``WanTransformer3DModel``.

    Returns ``True`` if TREAD was actually installed (i.e. something is enabled).
    Safe to call multiple times (idempotent). Assumes bf16/fp32 (non-quantized) weights.

    ``ranges`` are per-timestep overrides (see :func:`parse_tread_ranges`); they are
    stored on the transformer and resolved at forward time from the batch's global
    timestep. A range may enable routing even when the base config leaves it off, so
    installation happens when the base config is active OR any range enables routing.
    """
    ranges = list(ranges) if ranges else []
    routing_possible = cfg.enabled or any(
        r.get("enabled") is True for r in ranges
    )
    if not (cfg.active or routing_possible):
        return False
    if getattr(transformer, "_tread_installed", False):
        return True

    num_layers = len(transformer.blocks)

    # fp32 attention is run via SDPA (F.scaled_dot_product_attention), which auto-selects
    # the memory-efficient kernel for fp32 (flash is fp16/bf16-only). SDPA is available in
    # PyTorch >= 2.0; guard just in case of a very old install.
    if cfg.has_fp32 and not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        raise RuntimeError(
            "TREAD fp32 front/tail layers require torch.nn.functional.scaled_dot_product_attention "
            "(PyTorch >= 2.0), which is not available in this environment. Disable fp32_front / "
            "fp32_last_layers or upgrade PyTorch."
        )

    # Upcast the selected layers to fp32 ONCE (never downcasting the rest of the bf16 model).
    # This is the runtime safety net: it is a no-op when the loader already applied the
    # selective fp32 cast (see ``selective_cast_model`` / ``cast_state_dict_for_tread``).
    prefixes = tread_fp32_param_prefixes(cfg, num_layers)
    for name, module in list(transformer.named_modules()):
        if name and is_tread_fp32_name(name, prefixes):
            module.to(torch.float32)
    # ``scale_shift_table`` is a buffer on the root module, not a submodule, so upcast it
    # explicitly (it is a no-op if it is already fp32 from the load-time cast).
    if is_tread_fp32_name("scale_shift_table", prefixes):
        transformer.scale_shift_table.data = transformer.scale_shift_table.data.to(torch.float32)

    # Wrap each fp32 block's forward so its attention ALWAYS runs via SDPA - including
    # during gradient-checkpointing recompute, which re-runs the block in eager mode
    # (dynamo disabled) OUTSIDE any forward-time context. A context manager set in the
    # forward loop would not be active on the recompute path, so the SDPA choice must be
    # a property of the block itself. (``nn.Module.__call__`` resolves ``self.forward`` at
    # call time, so this instance-attribute wrap is picked up by both the normal call and
    # the checkpoint recompute call.)
    if cfg.has_fp32:
        for i in cfg.fp32_block_indices(num_layers):
            block = transformer.blocks[i]
            if not getattr(block, "_tread_fp32_wrapped", False):
                _orig_forward = block.forward

                def _make_wrapped(orig):
                    def _wrapped(*args, **kwargs):
                        with _fp32_attention_ctx():
                            return orig(*args, **kwargs)
                    return _wrapped

                block.forward = _make_wrapped(_orig_forward)
                block._tread_fp32_wrapped = True

    # Preserve the stock LoRA-scale wrapping (no-op for ai-toolkit's own LoRA, but kept for
    # correctness if the PEFT backend is ever used).
    try:
        from diffusers.utils.peft_utils import apply_lora_scale
        wrapped = apply_lora_scale("attention_kwargs")(tread_forward)
    except Exception:
        wrapped = tread_forward

    # Bind the transformer as ``self`` (mirrors the ``condition_embedder.forward`` pattern).
    transformer.forward = partial(wrapped, transformer, _tread_cfg=cfg)
    # Per-timestep overrides, resolved in ``tread_forward`` from the batch's global timestep.
    transformer._tread_ranges = ranges
    transformer._tread_installed = True

    end = cfg.resolved_end_layer(num_layers)
    route = f"r{cfg.start_layer}->{end} keep={cfg.keep_ratio} {cfg.mode}/{cfg.granularity}" if cfg.enabled else "n/a"
    extras = []
    if cfg.fp32_front:
        extras.append("fp32 front (patch_embed+cond_embed+block0)")
    if cfg.fp32_last_layers > 0:
        extras.append(f"fp32 last {cfg.fp32_last_layers} blocks + head")
    if cfg.fp32_layers:
        extras.append(f"fp32 layers {sorted(cfg.fp32_layers)}")
    if ranges:
        extras.append(f"{len(ranges)} per-timestep range override(s)")
    extra_msg = f", {', '.join(extras)}" if extras else ""
    print(f"[TREAD] {label}: route={route}{extra_msg}")
    return True


def apply_tread_to_dual(model, model_kwargs: Optional[dict]) -> bool:
    """Apply TREAD to both experts of a ``DualWanTransformer3DModel``.

    Each expert gets its own :class:`TREADConfig`, resolved with per-expert overrides
    (``tread_high`` / ``tread_low``) falling back to the global ``tread`` settings - see
    :func:`parse_tread_config`. Returns ``True`` if TREAD was enabled on at least one expert.
    """
    transformers = []
    if hasattr(model, "transformer_1"):
        transformers.append(("transformer_1 (high-noise)", model.transformer_1, "high"))
    if hasattr(model, "transformer_2"):
        transformers.append(("transformer_2 (low-noise)", model.transformer_2, "low"))
    applied = False
    for label, tr, expert in transformers:
        cfg = parse_tread_config(model_kwargs, expert=expert)
        # Per-expert timestep ranges (fall back to the global list for this expert).
        ranges = parse_tread_ranges(model_kwargs, expert=expert)
        applied = apply_tread(tr, cfg, label, ranges=ranges) or applied
    return applied
