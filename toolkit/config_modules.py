import os
import time
import re
from typing import List, Optional, Literal, Tuple, Union, TYPE_CHECKING, Dict
import random

import torch
import torchaudio

from toolkit.audio.album_artwork import add_album_artwork
from toolkit.prompt_utils import PromptEmbeds
from torchao.quantization.quant_primitives import _DTYPE_TO_BIT_WIDTH

ImgExt = Literal['jpg', 'png', 'webp', 'jxl', 'mp4']

SaveFormat = Literal['safetensors', 'diffusers']

if TYPE_CHECKING:
    from toolkit.guidance import GuidanceType
    from toolkit.logging_aitk import EmptyLogger
else:
    EmptyLogger = None

# =============================================================================
# Layer Range Parsing Utility
# =============================================================================
def _parse_layer_range(s: str) -> List[int]:
    """Parse a layer range string like '0-2', '0..2', '0...2', '0,1,2', or '1-4,5-6' into a list of layer indices."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return []
    
    # Split by comma to handle multiple segments (e.g., '1-4,5-6' or '0,1,2')
    segments = [seg.strip() for seg in s.split(',') if seg.strip()]
    
    result = []
    for seg in segments:
        # Try range with 1 to 3 dots
        match = re.match(r'^(\d+)(?:\.{1,3})(\d+)$', seg)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if start <= end:
                result.extend(range(start, end + 1))
            else:
                result.extend(range(start, end - 1, -1))
            continue

        # Try range with hyphen
        match = re.match(r'^(\d+)-(\d+)$', seg)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if start <= end:
                result.extend(range(start, end + 1))
            else:
                result.extend(range(start, end - 1, -1))
            continue

        # Try single number
        try:
            result.append(int(seg))
        except ValueError:
            return []  # Invalid segment — return empty

    return result


# =============================================================================
# Wan 2.2 14B Tensor Type Configuration
# =============================================================================
# These define the tensor types in Wan 2.2 and their maximum LoRA ranks.
# The max_rank is the minimum dimension of the weight tensor, which limits
# the maximum effective rank of a LoRA adapter for that layer.
#
# Tensors are identified by their LOADED (diffusers) names, not their
# original checkpoint names. Wan 2.2 remaps tensor names during loading:
#   - self_attn -> attn1 with to_q, to_k, to_v, to_out.0
#   - cross_attn -> attn2 with to_q, to_k, to_v, to_out.0
#   - ffn.0 -> ffn.net.0.proj
#   - ffn.2 -> ffn.net.2
#   - text_embedding -> condition_embedder.text_embedder.linear_1/2
#   - time_embedding -> condition_embedder.time_embedder.linear_1/2
#   - head -> proj_out
#   - patch_embedding stays as patch_embedding
#
# A rank of -1 means the layer should use full weight training (no LoRA)
# because the tensor shape doesn't support low-rank decomposition well.
# =============================================================================

WAN22_TENSOR_TYPES: Dict[str, Dict[str, Union[int, str, List[str]]]] = {
    'self_attn': {
        'max_rank': 5120,
        'description': 'Self-attention (attn1) q/k/v/o projections',
        'name_patterns': [r'attn1\.to_q', r'attn1\.to_k', r'attn1\.to_v', r'attn1\.to_out\.0'],
        'sub_types': {
            'self_attn.q': { 'pattern': r'attn1\.to_q', 'max_rank': 5120, 'description': 'Self-attention Q projection' },
            'self_attn.k': { 'pattern': r'attn1\.to_k', 'max_rank': 5120, 'description': 'Self-attention K projection' },
            'self_attn.v': { 'pattern': r'attn1\.to_v', 'max_rank': 5120, 'description': 'Self-attention V projection' },
            'self_attn.o': { 'pattern': r'attn1\.to_out\.0', 'max_rank': 5120, 'description': 'Self-attention output projection' },
        }
    },
    'cross_attn': {
        'max_rank': 5120,
        'description': 'Cross-attention (attn2) q/k/v/o projections',
        'name_patterns': [r'attn2\.to_q', r'attn2\.to_k', r'attn2\.to_v', r'attn2\.to_out\.0'],
        'sub_types': {
            'cross_attn.q': { 'pattern': r'attn2\.to_q', 'max_rank': 5120, 'description': 'Cross-attention Q projection' },
            'cross_attn.k': { 'pattern': r'attn2\.to_k', 'max_rank': 5120, 'description': 'Cross-attention K projection' },
            'cross_attn.v': { 'pattern': r'attn2\.to_v', 'max_rank': 5120, 'description': 'Cross-attention V projection' },
            'cross_attn.o': { 'pattern': r'attn2\.to_out\.0', 'max_rank': 5120, 'description': 'Cross-attention output projection' },
        }
    },
    'ffn': {
        'max_rank': 5120,
        'description': 'Feed-forward network projections',
        'name_patterns': [r'ffn\.net\.0\.proj', r'ffn\.net\.2'],
        'sub_types': {
            'ffn.0': { 'pattern': r'ffn\.net\.0\.proj', 'max_rank': 5120, 'description': 'FFN up projection' },
            'ffn.2': { 'pattern': r'ffn\.net\.2', 'max_rank': 5120, 'description': 'FFN down projection' },
        }
    },
    'text_embedding': {
        'max_rank': 4096,
        'description': 'Text embedding projections (linear_1/linear_2)',
        'name_patterns': [r'condition_embedder\.text_embedder\.linear_1', r'condition_embedder\.text_embedder\.linear_2'],
        'sub_types': {
            'text_embedding.1': { 'pattern': r'condition_embedder\.text_embedder\.linear_1', 'max_rank': 4096, 'description': 'Text embedding linear_1' },
            'text_embedding.2': { 'pattern': r'condition_embedder\.text_embedder\.linear_2', 'max_rank': 4096, 'description': 'Text embedding linear_2' },
        }
    },
    'time_embedding': {
        'max_rank': 256,
        'description': 'Time embedding projections (linear_1/linear_2)',
        'name_patterns': [r'condition_embedder\.time_embedder\.linear_1', r'condition_embedder\.time_embedder\.linear_2'],
        'sub_types': {
            'time_embedding.1': { 'pattern': r'condition_embedder\.time_embedder\.linear_1', 'max_rank': 256, 'description': 'Time embedding linear_1' },
            'time_embedding.2': { 'pattern': r'condition_embedder\.time_embedder\.linear_2', 'max_rank': 256, 'description': 'Time embedding linear_2' },
        }
    },
    'head': {
        'max_rank': 64,
        'description': 'Output head projection',
        'name_patterns': [r'proj_out'],
        'sub_types': {}
    },
    # Layers that typically should be full weight, not LoRA
    'patch_embedding': {
        'max_rank': -1,  # -1 means full weight only
        'description': 'Patch embedding (conv-like, full weight recommended)',
        'name_patterns': [r'patch_embedding'],
        'sub_types': {}
    },
    'modulation': {
        'max_rank': -1,  # -1 means full weight only
        'description': 'Modulation / scale_shift_table parameters',
        'name_patterns': [r'scale_shift_table'],
        'sub_types': {}
    },
    'norm': {
        'max_rank': -1,  # -1 means full weight only
        'description': 'Normalization layer parameters',
        'name_patterns': [r'norm\d+\.weight', r'norm\d+\.bias'],
        'sub_types': {}
    },
    'time_projection': {
        'max_rank': -1,  # -1 means full weight only
        'description': 'Time projection parameters',
        'name_patterns': [r'condition_embedder\.time_proj'],
        'sub_types': {}
    },
}

# User-friendly aliases for tensor types (config accepts both formats)
# Includes short names (ffn.up), full layer paths, and alternative naming conventions
WAN22_TENSOR_TYPE_ALIASES: Dict[str, str] = {
    # FFN aliases
    'ffn.up': 'ffn.0',
    'ffn.down': 'ffn.2',
    'ffn.net.0.proj': 'ffn.0',
    'ffn.net.2': 'ffn.2',
    # Text embedding aliases (full layer paths)
    'condition_embedder.text_embedder.linear_1': 'text_embedding.1',
    'condition_embedder.text_embedder.linear_2': 'text_embedding.2',
    'text_embedder.linear_1': 'text_embedding.1',
    'text_embedder.linear_2': 'text_embedding.2',
    # Time embedding aliases (full layer paths)
    'condition_embedder.time_embedder.linear_1': 'time_embedding.1',
    'condition_embedder.time_embedder.linear_2': 'time_embedding.2',
    'time_embedder.linear_1': 'time_embedding.1',
    'time_embedder.linear_2': 'time_embedding.2',
    # Head aliases
    'proj_out': 'head',
    # Patch embedding aliases
    'patch_embed': 'patch_embedding',
    # Modulation aliases
    'scale_shift_table': 'modulation',
    'mod': 'modulation',
    # Time projection aliases
    'condition_embedder.time_proj': 'time_projection',
    'time_proj': 'time_projection',
    # Self attention sub-type aliases
    'attn1.to_q': 'self_attn.q',
    'attn1.to_k': 'self_attn.k',
    'attn1.to_v': 'self_attn.v',
    'attn1.to_out.0': 'self_attn.o',
    # Cross attention sub-type aliases
    'attn2.to_q': 'cross_attn.q',
    'attn2.to_k': 'cross_attn.k',
    'attn2.to_v': 'cross_attn.v',
    'attn2.to_out.0': 'cross_attn.o',
}

# Flatten sub-types into a single lookup dict
WAN22_ALL_TENSOR_TYPES: Dict[str, Dict[str, Union[int, str]]] = {}
for _parent_type, _info in WAN22_TENSOR_TYPES.items():
    WAN22_ALL_TENSOR_TYPES[_parent_type] = _info
    if 'sub_types' in _info:
        for _sub_name, _sub_info in _info['sub_types'].items():
            WAN22_ALL_TENSOR_TYPES[_sub_name] = _sub_info

WAN22_LINEAR_TENSOR_TYPES = [k for k, v in WAN22_TENSOR_TYPES.items() if v['max_rank'] > 0]
WAN22_FULL_TENSOR_TYPES = [k for k, v in WAN22_TENSOR_TYPES.items() if v['max_rank'] == -1]

# Tensor types that are single (not per-block/layer) - layer_range is ignored for these
WAN22_SINGLE_TENSOR_TYPES = {
    'patch_embedding', 'modulation', 'norm', 'time_projection', 'head',
    'text_embedding', 'text_embedding.1', 'text_embedding.2',
    'time_embedding', 'time_embedding.1', 'time_embedding.2',
}


def _resolve_tensor_type(tensor_type: str) -> str:
    """Resolve a tensor type to its canonical name using aliases.
    
    If the tensor type is already canonical, returns it unchanged.
    If it's an alias (e.g., 'ffn.up', 'condition_embedder.text_embedder.linear_1'),
    returns the canonical name (e.g., 'ffn.0', 'text_embedding.1').
    If not found in either, returns the input unchanged.
    """
    # Already canonical?
    if tensor_type in WAN22_ALL_TENSOR_TYPES:
        return tensor_type
    # Check aliases
    if tensor_type in WAN22_TENSOR_TYPE_ALIASES:
        return WAN22_TENSOR_TYPE_ALIASES[tensor_type]
    # Return as-is (might be validated later or might be invalid)
    return tensor_type


def _is_single_tensor_type(tensor_type: str) -> bool:
    """Check if a tensor type is a single tensor (not per-block).
    Resolves aliases first."""
    canonical = _resolve_tensor_type(tensor_type)
    return canonical in WAN22_SINGLE_TENSOR_TYPES


def get_wan22_tensor_type_from_name(layer_name: str) -> Optional[str]:
    """
    Determine the Wan 2.2 tensor type from a layer name (diffusers format).
    Returns the most specific tensor type key (e.g., 'self_attn.q', 'self_attn', 'ffn') or None if not matched.
    Prefers sub-types (e.g., 'self_attn.q') over parent types (e.g., 'self_attn').
    """
    # First try sub-types (more specific match)
    for tensor_type, config in WAN22_ALL_TENSOR_TYPES.items():
        if 'pattern' in config:
            if re.search(config['pattern'], layer_name):
                return tensor_type
    # Then try parent types (fallback)
    for tensor_type, config in WAN22_TENSOR_TYPES.items():
        for pattern in config['name_patterns']:
            if re.search(pattern, layer_name):
                return tensor_type
    return None


def get_wan22_max_rank_for_type(tensor_type: str) -> int:
    """
    Get the maximum LoRA rank for a Wan 2.2 tensor type.
    Returns -1 for types that should use full weight training.
    """
    if tensor_type in WAN22_ALL_TENSOR_TYPES:
        return WAN22_ALL_TENSOR_TYPES[tensor_type]['max_rank']
    if tensor_type in WAN22_TENSOR_TYPES:
        return WAN22_TENSOR_TYPES[tensor_type]['max_rank']
    return -1


def is_wan22_tensor_type_enabled(tensor_type: str, enabled_types: Optional[List[str]]) -> bool:
    """
    Check if a tensor type is enabled for training.
    If enabled_types is None or empty, all types are enabled by default.
    """
    if enabled_types is None or len(enabled_types) == 0:
        return True
    return tensor_type in enabled_types

class SaveConfig:
    def __init__(self, **kwargs):
        self.save_every: int = kwargs.get('save_every', 1000)
        self.dtype: str = kwargs.get('dtype', 'float16')
        self.max_step_saves_to_keep: int = kwargs.get('max_step_saves_to_keep', 5)
        self.save_format: SaveFormat = kwargs.get('save_format', 'safetensors')
        if self.save_format not in ['safetensors', 'diffusers']:
            raise ValueError(f"save_format must be safetensors or diffusers, got {self.save_format}")
        self.push_to_hub: bool = kwargs.get("push_to_hub", False)
        self.hf_repo_id: Optional[str] = kwargs.get("hf_repo_id", None)
        self.hf_private: Optional[str] = kwargs.get("hf_private", False)

class LoggingConfig:
    def __init__(self, **kwargs):
        self.log_every: int = kwargs.get('log_every', 100)
        self.verbose: bool = kwargs.get('verbose', False)
        self.use_wandb: bool = kwargs.get('use_wandb', False)
        self.use_ui_logger: bool = kwargs.get('use_ui_logger', False)
        self.project_name: str = kwargs.get('project_name', 'ai-toolkit')
        self.run_name: str = kwargs.get('run_name', None)

class SampleItem:
    def __init__(
        self,
        sample_config: 'SampleConfig',
        **kwargs
    ):
        # prompt should always be in the kwargs
        self.prompt = kwargs.get('prompt', None)
        self.width: int = kwargs.get('width', sample_config.width)
        self.height: int = kwargs.get('height', sample_config.height)
        self.neg: str = kwargs.get('neg', sample_config.neg)
        self.seed: Optional[int] = kwargs.get('seed', None) # if none, default to autogen seed
        self.guidance_scale: float = kwargs.get('guidance_scale', sample_config.guidance_scale)
        self.sample_steps: int = kwargs.get('sample_steps', sample_config.sample_steps)
        self.fps: int = kwargs.get('fps', sample_config.fps)
        self.num_frames: int = kwargs.get('num_frames', sample_config.num_frames)
        self.ctrl_img: Optional[str] = kwargs.get('ctrl_img', None)
        self.ctrl_idx: int = kwargs.get('ctrl_idx', 0)
        # for multi control image models
        self.ctrl_img_1: Optional[str] = kwargs.get('ctrl_img_1', self.ctrl_img)
        self.ctrl_img_2: Optional[str] = kwargs.get('ctrl_img_2', None)
        self.ctrl_img_3: Optional[str] = kwargs.get('ctrl_img_3', None)
        
        self.network_multiplier: float = kwargs.get('network_multiplier', sample_config.network_multiplier)
        # convert to a number if it is a string
        if isinstance(self.network_multiplier, str):
            try:
                self.network_multiplier = float(self.network_multiplier)
            except:
                print(f"Invalid network_multiplier {self.network_multiplier}, defaulting to 1.0")
                self.network_multiplier = 1.0
        
        # only for models that support it, (qwen image edit 2509 for now)
        self.do_cfg_norm: bool = kwargs.get('do_cfg_norm', False)

        # NAG (Negative Attention Guidance) parameters - per-sample override
        self.nag_scale: Optional[float] = kwargs.get('nag_scale', None)
        self.nag_alpha: Optional[float] = kwargs.get('nag_alpha', None)
        self.nag_tau: Optional[float] = kwargs.get('nag_tau', None)

        # Attention tanh softcap - per-sample override (Wan 2.x only).
        # None = follow the global "Apply Tanh Softcapping During Sampling"
        # toggle / inherit the sample-level (then training) soft cap value.
        self.attention_tanh_softcap_enabled: Optional[bool] = kwargs.get('attention_tanh_softcap_enabled', None)
        self.attention_tanh_softcap_value: Optional[float] = kwargs.get('attention_tanh_softcap_value', None)

class SampleConfig:
    def __init__(self, **kwargs):
        self.sampler: str = kwargs.get('sampler', 'ddpm')
        # Attention backend used while generating samples (Wan 2.x toolkit path).
        # Same values as TrainConfig.attention_backend:
        # native (default/auto), flex, sdpa, flash.
        self.attention_backend: str = kwargs.get('attention_backend', 'native')
        # Whether tanh softcapping is applied during sampling (Wan 2.x toolkit
        # path). Independent of train.attention_tanh_softcap_enabled; uses the
        # same soft_cap value/overrides. Off by default to match standard
        # inference; enabling it forces the sampling backend to flex_attention.
        self.attention_tanh_softcap_enabled: bool = kwargs.get('attention_tanh_softcap_enabled', False)
        # Sampling-specific soft cap value (Wan 2.x). Decoupled from the
        # training value so samples can be generated with a different cap.
        # None = inherit train.attention_tanh_softcap_value. Individual
        # samples (samples[i].attention_tanh_softcap_value) take precedence.
        self.attention_tanh_softcap_value: Optional[float] = kwargs.get('attention_tanh_softcap_value', None)
        self.sample_every: int = kwargs.get('sample_every', 100)
        self.width: int = kwargs.get('width', 512)
        self.height: int = kwargs.get('height', 512)
        self.neg = kwargs.get('neg', False)
        self.seed = kwargs.get('seed', 0)
        self.walk_seed = kwargs.get('walk_seed', False)
        self.guidance_scale = kwargs.get('guidance_scale', 7)
        self.sample_steps = kwargs.get('sample_steps', 20)
        self.network_multiplier = kwargs.get('network_multiplier', 1)
        self.guidance_rescale = kwargs.get('guidance_rescale', 0.0)
        self.ext: ImgExt = kwargs.get('format', 'jpg')
        self.adapter_conditioning_scale = kwargs.get('adapter_conditioning_scale', 1.0)
        self.refiner_start_at = kwargs.get('refiner_start_at',
                                           0.5)  # step to start using refiner on sample if it exists
        self.extra_values = kwargs.get('extra_values', [])
        self.num_frames = kwargs.get('num_frames', 1)
        self.fps: int = kwargs.get('fps', 16)
        # Shift value used during sampling/inference only. Does not affect training.
        # For Wan2.2 models trained with timestep_type: sigmoid, this controls the
        # flow matching shift applied during generation (e.g., 8.0 gives the
        # desired 1:1 high/low noise step ratio).
        self.sampling_flow_shift: Optional[float] = kwargs.get('sampling_flow_shift', None)
        if self.num_frames > 1 and self.ext not in ['webp', 'mp4']:
            print("Changing sample extension to mp4")
            self.ext = 'mp4'
        
        prompts: list[str] = kwargs.get('prompts', [])
        
        self.samples: Optional[List[SampleItem]] = None
        # use the legacy prompts if it is passed that way to get samples object
        default_samples_kwargs = [
            {"prompt": x} for x in prompts
        ]
        raw_samples = kwargs.get('samples', default_samples_kwargs)
        self.samples = [SampleItem(self, **item) for item in raw_samples]
        # only for models that support it, (qwen image edit 2509 for now)
        self.do_cfg_norm: bool = kwargs.get('do_cfg_norm', False)

        # NAG (Negative Attention Guidance) global defaults for all samples
        # nag_scale: 1 disables, >1 enables (typical range 1.0–20.0)
        # nag_tau:   threshold for similarity-based scaling (typical 1.0–5.0)
        # nag_alpha: blend factor between NAG-guided and original prediction (0.0–2.0)
        self.nag_scale: float = kwargs.get('nag_scale', 1.0)
        self.nag_alpha: float = kwargs.get('nag_alpha', 0.5)
        self.nag_tau: float = kwargs.get('nag_tau', 3.5)
        
    @property
    def prompts(self):
        # for backwards compatibility as this is checked for length frequently
        return [sample.prompt for sample in self.samples if sample.prompt is not None]
  
                


class LormModuleSettingsConfig:
    def __init__(self, **kwargs):
        self.contains: str = kwargs.get('contains', '4nt$3')
        self.extract_mode: str = kwargs.get('extract_mode', 'ratio')
        # min num parameters to attach to
        self.parameter_threshold: int = kwargs.get('parameter_threshold', 0)
        self.extract_mode_param: dict = kwargs.get('extract_mode_param', 0.25)


class LoRMConfig:
    def __init__(self, **kwargs):
        self.extract_mode: str = kwargs.get('extract_mode', 'ratio')
        self.do_conv: bool = kwargs.get('do_conv', False)
        self.extract_mode_param: dict = kwargs.get('extract_mode_param', 0.25)
        self.parameter_threshold: int = kwargs.get('parameter_threshold', 0)
        module_settings = kwargs.get('module_settings', [])
        default_module_settings = {
            'extract_mode': self.extract_mode,
            'extract_mode_param': self.extract_mode_param,
            'parameter_threshold': self.parameter_threshold,
        }
        module_settings = [{**default_module_settings, **module_setting, } for module_setting in module_settings]
        self.module_settings: List[LormModuleSettingsConfig] = [LormModuleSettingsConfig(**module_setting) for
                                                                module_setting in module_settings]

    def get_config_for_module(self, block_name):
        for setting in self.module_settings:
            contain_pieces = setting.contains.split('|')
            if all(contain_piece in block_name for contain_piece in contain_pieces):
                return setting
            # try replacing the . with _
            contain_pieces = setting.contains.replace('.', '_').split('|')
            if all(contain_piece in block_name for contain_piece in contain_pieces):
                return setting
            # do default
        return LormModuleSettingsConfig(**{
            'extract_mode': self.extract_mode,
            'extract_mode_param': self.extract_mode_param,
            'parameter_threshold': self.parameter_threshold,
        })


NetworkType = Literal['lora', 'locon', 'lorm', 'lokr']


class RankGateConfig:
    """
    Configuration for SparseForge-inspired rank gate annealing.
    
    Implements soft, curvature-aware rank gating for LoRA adapters,
    allowing gradual elimination of redundant ranks during training.
    
    Based on SparseForge (2026) with adaptations for diffusion training.
    Default values tuned for Wan 2.2 14B I2V LoRA training.
    """
    def __init__(self, **kwargs):
        # Enable/disable rank gating. Default: True (enabled by default for
        # curvature-aware pruning to prevent rank collapse).
        self.enabled: bool = kwargs.get('enabled', True)

        # ------------------------------------------------------------------
        # TIMING MODE
        # ------------------------------------------------------------------
        # auto_timing=True (default): annealing start / end and the final
        # hardening window are all detected automatically from the ACTUAL
        # training dynamics (loss plateau + learning-rate decay) via the
        # per-expert LearningAwareScheduler. This is the recommended mode and
        # removes all hardcoded step percentages.
        #
        # auto_timing=False: fall back to the legacy manual schedule driven by
        # the absolute start_step / end_step / hardening_window values below.
        self.auto_timing: bool = kwargs.get('auto_timing', True)

        # Legacy MANUAL timing (only used when auto_timing is False).
        # start_step: when to begin annealing
        self.start_step: Optional[int] = kwargs.get('start_step', None)  # manual: required if auto_timing=False
        # end_step: when to complete annealing (before hardening window)
        self.end_step: Optional[int] = kwargs.get('end_step', None)

        # ------------------------------------------------------------------
        # LEARNING-AWARE AUTO TIMING (used when auto_timing is True)
        # ------------------------------------------------------------------
        # Delay the annealing start until AFTER the LR scheduler warmup so
        # that the first gate decisions (and the per-tensor rank budgets
        # computed at that moment) are based on stable, non-ramping gradients
        # and an already-primed Fisher EMA.
        self.start_after_warmup: bool = kwargs.get('start_after_warmup', True)

        # --- Plateau detection (annealing START trigger) ---
        # Annealing starts once the expert's loss has stopped improving
        # meaningfully. "Meaningful" is defined by comparing a fast loss EMA
        # to a slow loss EMA: relative improvement
        #   (slow - fast) / |slow|
        # below this threshold counts as a "flat" step.
        self.plateau_relative_threshold: float = kwargs.get('plateau_relative_threshold', 5e-3)
        # Number of CONSECUTIVE flat steps required to confirm a plateau before
        # annealing starts (debounce, avoids triggering on a single noisy dip).
        self.plateau_confirm_steps: int = kwargs.get('plateau_confirm_steps', 50)
        # Per-expert floor: annealing will never start before this many
        # per-expert steps (also raised to warmup+1 when start_after_warmup).
        self.min_anneal_steps: int = kwargs.get('min_anneal_steps', 200)

        # --- Annealing END trigger ---
        # When the LR scheduler is decaying, annealing progress is driven by
        # the LR itself: it completes when the LR has decayed below
        # end_lr_fraction * peak_lr (the model is converging, safe to prune).
        self.end_lr_fraction: float = kwargs.get('end_lr_fraction', 0.2)
        # Per-expert max duration (steps) of the annealing window, used as the
        # fallback clock when the LR is constant (no decay to track) and as a
        # hard cap on the LR-driven window.
        self.anneal_max_duration: int = kwargs.get('anneal_max_duration', 1500)

        # --- Final HARDENING trigger (soft gates -> binary) ---
        # Hardening starts automatically when the LR has decayed below
        # hardening_lr_fraction * peak_lr (learning is essentially finished).
        self.hardening_lr_fraction: float = kwargs.get('hardening_lr_fraction', 0.05)
        # Per-expert minimum length of the hardening window (steps). Also used
        # as the step-based trigger when the LR is constant.
        self.hardening_min_steps: int = kwargs.get('hardening_min_steps', 150)

        # --- Loss EMA rates for plateau detection ---
        self.loss_ema_fast: float = kwargs.get('loss_ema_fast', 0.10)
        self.loss_ema_slow: float = kwargs.get('loss_ema_slow', 0.02)

        # ------------------------------------------------------------------
        # TRUNCATED CHECKPOINT ("button")
        # ------------------------------------------------------------------
        # When True, every checkpoint save ALSO emits a fully-truncated
        # variant: the LoRA rank is physically reduced (dead rows of lora_down
        # and columns of lora_up are removed, alpha rescaled), not merely
        # zeroed via gates. The .diff tensors are folded as usual.
        self.save_truncated: bool = kwargs.get('save_truncated', False)
        # Gate value above which a rank is kept in the truncated checkpoint.
        self.truncation_threshold: float = kwargs.get('truncation_threshold', 0.5)

        # Manual hardening window size (only used when auto_timing is False).
        self._legacy_hardening_window: int = kwargs.get('hardening_window', 500)
        
        # Target minimal per-component rank contribution (fraction of the
        # tensor's total energy). At annealing start, EACH tensor's final
        # target is computed from its current energy spectrum (per-tensor
        # annealing, not a global ratio):
        #   - LoRA pairs:  S^2/sum(S^2) of the SVD of B@A (per rank)
        #   - .diff tensors (full finetune, e.g. 1D layer norms): x^2/sum(x^2) per element
        # Components contributing less than this fraction are annealed out.
        # Default 1e-4 (0.01% of total energy) matches the recommended-rank
        # threshold of the offline LoRA statistics tool.
        self.target_min_rank_contribution: float = kwargs.get('target_min_rank_contribution', 1e-4)
        
        # Target rank ratio: global FALLBACK final active fraction, used only
        # for tensors whose per-tensor budget could not be computed
        # (e.g. weight references unavailable).
        # 0.3 = keep 30% of components, kill 70% (aggressive pruning)
        self.target_rank_ratio: float = kwargs.get('target_rank_ratio', 0.3)
        
        # Temperature for sigmoid gating
        # Higher = softer decisions, lower = more decisive
        self.temperature: float = kwargs.get('temperature', 1.0)
        
        # Temperature decay per update: T ← γT
        # 0.95 = faster decay, sharpens sigmoid decisions more quickly
        self.gamma: float = kwargs.get('gamma', 0.95)
        
        # EMA update rate for gates: m ← (1-α)m + αG
        # 0.1 = faster gate updates, more responsive to curvature changes
        self.alpha: float = kwargs.get('alpha', 0.1)
        
        # Binary preference penalty max: L_mid = Σ m(1-m)
        # 0.01 = stronger penalty, pushes gates toward 0 or 1 more aggressively
        self.lambda_mid_max: float = kwargs.get('lambda_mid_max', 0.01)
        
        # Update gates every N steps
        # 15 = more frequent updates (67 updates per 1000 steps vs 40)
        self.update_every: int = kwargs.get('update_every', 15)
        
        # Fisher EMA decay
        self.fisher_decay: float = kwargs.get('fisher_decay', 0.999)
        
        # Include first-order term |g·w| in scoring (recommended for diffusion)
        self.use_first_order: bool = kwargs.get('use_first_order', True)
        
        # Legacy manual hardening window size (kept as a public attribute for
        # backward compatibility; auto_timing uses hardening_min_steps instead).
        self.hardening_window: int = self._legacy_hardening_window
        
        # Penalty coefficient for mid-preference nudge
        self.eta_pen: float = kwargs.get('eta_pen', 0.01)
        
        # Enable final hardening (binarize gates at end)
        self.final_hardening: bool = kwargs.get('final_hardening', True)


class NetworkConfig:
    def __init__(self, **kwargs):
        self.type: NetworkType = kwargs.get('type', 'lora')
        rank = kwargs.get('rank', None)
        linear = kwargs.get('linear', None)
        if rank is not None:
            self.rank: int = rank  # rank for backward compatibility
            self.linear: int = rank
        elif linear is not None:
            self.rank: int = linear
            self.linear: int = linear
        else:
            self.rank: int = 4
            self.linear: int = 4
        self.conv: int = kwargs.get('conv', None)
        self.alpha: float = kwargs.get('alpha', 1.0)
        self.linear_alpha: float = kwargs.get('linear_alpha', self.alpha)
        self.conv_alpha: float = kwargs.get('conv_alpha', self.conv)
        self.dropout: Union[float, None] = kwargs.get('dropout', None)
        self.network_kwargs: dict = kwargs.get('network_kwargs', {})

        self.lorm_config: Union[LoRMConfig, None] = None
        lorm = kwargs.get('lorm', None)
        if lorm is not None:
            self.lorm_config: LoRMConfig = LoRMConfig(**lorm)

        if self.type == 'lorm':
            # set linear to arbitrary values so it makes them
            self.linear = 4
            self.rank = 4
            if self.lorm_config.do_conv:
                self.conv = 4

        self.transformer_only = kwargs.get('transformer_only', True)
        
        self.lokr_full_rank = kwargs.get('lokr_full_rank', False)
        if self.lokr_full_rank and self.type.lower() == 'lokr':
            self.linear = 9999999999
            self.linear_alpha = 9999999999
            self.conv = 9999999999
            self.conv_alpha = 9999999999
        # -1 automatically finds the largest factor
        self.lokr_factor = kwargs.get('lokr_factor', -1)
        
        # Use the old lokr format
        self.old_lokr_format = kwargs.get('old_lokr_format', False)
        
        # for multi stage models
        self.split_multistage_loras = kwargs.get('split_multistage_loras', True)
        
        # LoRA initialization methods
        # Can be a string (e.g. 'gaussian_random') or dict with 'method' and optional 'std'
        self.lora_a_init = kwargs.get('lora_a_init', 'gaussian_random')
        self.lora_b_init = kwargs.get('lora_b_init', 'zeros')
        self.high_noise_lora_a_init = kwargs.get('high_noise_lora_a_init', None)
        self.high_noise_lora_b_init = kwargs.get('high_noise_lora_b_init', None)
        self.low_noise_lora_a_init = kwargs.get('low_noise_lora_a_init', None)
        self.low_noise_lora_b_init = kwargs.get('low_noise_lora_b_init', None)
        
        # ramtorch, doesn't work yet
        self.layer_offloading = kwargs.get('layer_offloading', False)
        
        # start from a pretrained lora
        self.pretrained_lora_path = kwargs.get('pretrained_lora_path', None)
        
        # will create diffirential full weight modules for layers not conv/linear
        # only useful in very special cases. 
        self.all_layers = kwargs.get('all_layers', False)
        
        # =====================================================================
        # Wan 2.2 Tensor-Type-Specific LoRA Configuration
        # =====================================================================
        # Allows fine-grained control over which tensor types are trained
        # and what rank/alpha each type should use. This is especially useful
        # for Wan 2.2 14B where different layer types have different optimal ranks.
        #
        # wan22_tensor_types:
        #   dict mapping tensor type -> config, e.g.:
        #     wan22_tensor_types:
        #       self_attn:
        #         rank: 256
        #         alpha: 256
        #         full: false  # use LoRA instead of full weight
        #       cross_attn:
        #         rank: 128
        #         alpha: 128
        #       ffn:
        #         rank: null  # null means skip (no training)
        #       text_embedding:
        #         rank: 128
        #         alpha: 128
        #       patch_embedding:
        #         rank: 128
        #         alpha: 128
        #
        # If wan22_tensor_types is specified, only the types listed are trained
        # (unless enabled_types is also used). Types not listed are skipped.
        # If wan22_tensor_types is not specified, the legacy `linear`, `conv`,
        # `all_layers`, and `full_if_contains` settings are used.
        # =====================================================================
        self.wan22_tensor_types: Optional[Dict[str, Dict]] = kwargs.get('wan22_tensor_types', None)
        
        # List of tensor type names to enable. If wan22_tensor_types is specified
        # but you want to only train a subset, use this.
        # Default: None means all types in wan22_tensor_types are enabled.
        self.wan22_enabled_types: Optional[List[str]] = kwargs.get('wan22_enabled_types', None)
        
        # =====================================================================
        # Rank Gate Annealing (SparseForge-inspired)
        # =====================================================================
        # Enables soft, curvature-aware rank gating for LoRA adapters.
        # ENABLED BY DEFAULT for better training quality.
        # Set rank_gates.enabled: false to disable.
        # =====================================================================
        rank_gates_input = kwargs.get('rank_gates', None)
        if rank_gates_input is not None:
            # Accept dict or bool
            if isinstance(rank_gates_input, bool):
                self.rank_gates: Optional[RankGateConfig] = RankGateConfig(enabled=rank_gates_input)
            elif isinstance(rank_gates_input, dict):
                self.rank_gates: Optional[RankGateConfig] = RankGateConfig(**rank_gates_input)
            else:
                self.rank_gates: Optional[RankGateConfig] = None
        else:
            # Default: ENABLED with sensible defaults
            self.rank_gates: Optional[RankGateConfig] = RankGateConfig(enabled=True)
        
        # =====================================================================
        # Per-Layer Rank Overrides
        # =====================================================================
        # Allows fine-grained rank control for specific layers within a tensor type.
        # Format: list of dicts with 'tensor_type', 'rank', 'layer_range'
        # e.g. [{'tensor_type': 'cross_attn', 'rank': 64, 'layer_range': '0-2'}]
        #      -> layers 0,1,2 of cross_attn get rank 64
        # These overrides take precedence over wan22_tensor_types and global rank.
        # =====================================================================
        layer_overrides_input = kwargs.get('layer_overrides', [])
        self.layer_overrides: List[Dict] = []
        for override in layer_overrides_input:
            if not isinstance(override, dict):
                continue
            tensor_type = override.get('tensor_type', 'cross_attn')
            # Resolve aliases (e.g., 'ffn.up' -> 'ffn.0', 'condition_embedder.text_embedder.linear_1' -> 'text_embedding.1')
            canonical_type = _resolve_tensor_type(tensor_type)
            if canonical_type != tensor_type:
                print(f"Resolved tensor type alias: '{tensor_type}' -> '{canonical_type}'")
            
            # For single tensors, layer_range is ignored
            if _is_single_tensor_type(canonical_type):
                self.layer_overrides.append({
                    'tensor_type': canonical_type,
                    'rank': override.get('rank', 32),
                    'layers': []  # Empty means "all instances" for single tensors
                })
            else:
                layer_range_str = override.get('layer_range', '')
                layers = _parse_layer_range(layer_range_str)
                if not layers:
                    print(f"Ignoring layer override with invalid/empty layer_range: {layer_range_str}")
                    continue
                self.layer_overrides.append({
                    'tensor_type': canonical_type,
                    'rank': override.get('rank', 32),
                    'layers': layers
                })
        
        # =====================================================================
        # Per-Expert Per-Layer Rank Overrides
        # =====================================================================
        # Same format as layer_overrides, but scoped to a specific transformer expert.
        # layer_overrides_high -> transformer_1 (high-noise expert)
        # layer_overrides_low  -> transformer_2 (low-noise expert)
        # Priority: per-expert overrides > global layer_overrides > wan22_tensor_types > global rank
        # =====================================================================
        def _parse_overrides(input_overrides: List[Dict], expert_name: str) -> List[Dict]:
            overrides = []
            for override in input_overrides:
                if not isinstance(override, dict):
                    continue
                tensor_type = override.get('tensor_type', 'cross_attn')
                # Resolve aliases
                canonical_type = _resolve_tensor_type(tensor_type)
                if canonical_type != tensor_type:
                    print(f"Resolved {expert_name} tensor type alias: '{tensor_type}' -> '{canonical_type}'")
                
                # For single tensors, layer_range is ignored
                if _is_single_tensor_type(canonical_type):
                    overrides.append({
                        'tensor_type': canonical_type,
                        'rank': override.get('rank', 32),
                        'layers': []  # Empty means "all instances" for single tensors
                    })
                else:
                    layer_range_str = override.get('layer_range', '')
                    layers = _parse_layer_range(layer_range_str)
                    if not layers:
                        print(f"Ignoring {expert_name} layer override with invalid/empty layer_range: {layer_range_str}")
                        continue
                    overrides.append({
                        'tensor_type': canonical_type,
                        'rank': override.get('rank', 32),
                        'layers': layers
                    })
            return overrides
        
        self.layer_overrides_high: List[Dict] = _parse_overrides(
            kwargs.get('layer_overrides_high', []),
            'layer_overrides_high (transformer_1)'
        )
        self.layer_overrides_low: List[Dict] = _parse_overrides(
            kwargs.get('layer_overrides_low', []),
            'layer_overrides_low (transformer_2)'
        )


AdapterTypes = Literal['t2i', 'ip', 'ip+', 'clip', 'ilora', 'photo_maker', 'control_net', 'control_lora', 'i2v']

CLIPLayer = Literal['penultimate_hidden_states', 'image_embeds', 'last_hidden_state']


class AdapterConfig:
    def __init__(self, **kwargs):
        self.type: AdapterTypes = kwargs.get('type', 't2i')  # t2i, ip, clip, control_net, i2v
        self.in_channels: int = kwargs.get('in_channels', 3)
        self.channels: List[int] = kwargs.get('channels', [320, 640, 1280, 1280])
        self.num_res_blocks: int = kwargs.get('num_res_blocks', 2)
        self.downscale_factor: int = kwargs.get('downscale_factor', 8)
        self.adapter_type: str = kwargs.get('adapter_type', 'full_adapter')
        self.image_dir: str = kwargs.get('image_dir', None)
        self.test_img_path: List[str] = kwargs.get('test_img_path', None)
        if self.test_img_path is not None:
            if isinstance(self.test_img_path, str):
                self.test_img_path = self.test_img_path.split(',')
                self.test_img_path = [p.strip() for p in self.test_img_path]
                self.test_img_path = [p for p in self.test_img_path if p != '']
                
        self.train: str = kwargs.get('train', False)
        self.image_encoder_path: str = kwargs.get('image_encoder_path', None)
        self.name_or_path = kwargs.get('name_or_path', None)

        num_tokens = kwargs.get('num_tokens', None)
        if num_tokens is None and self.type.startswith('ip'):
            if self.type == 'ip+':
                num_tokens = 16
                num_tokens = 16
            elif self.type == 'ip':
                num_tokens = 4

        self.num_tokens: int = num_tokens
        self.train_image_encoder: bool = kwargs.get('train_image_encoder', False)
        self.train_only_image_encoder: bool = kwargs.get('train_only_image_encoder', False)
        if self.train_only_image_encoder:
            self.train_image_encoder = True
        self.train_only_image_encoder_positional_embedding: bool = kwargs.get(
            'train_only_image_encoder_positional_embedding', False)
        self.image_encoder_arch: str = kwargs.get('image_encoder_arch', 'clip')  # clip vit vit_hybrid, safe
        self.safe_reducer_channels: int = kwargs.get('safe_reducer_channels', 512)
        self.safe_channels: int = kwargs.get('safe_channels', 2048)
        self.safe_tokens: int = kwargs.get('safe_tokens', 8)
        self.quad_image: bool = kwargs.get('quad_image', False)

        # clip vision
        self.trigger = kwargs.get('trigger', 'tri993r')
        self.trigger_class_name = kwargs.get('trigger_class_name', None)

        self.class_names = kwargs.get('class_names', [])

        self.clip_layer: CLIPLayer = kwargs.get('clip_layer', None)
        if self.clip_layer is None:
            if self.type.startswith('ip+'):
                self.clip_layer = 'penultimate_hidden_states'
            else:
                self.clip_layer = 'last_hidden_state'

        # text encoder
        self.text_encoder_path: str = kwargs.get('text_encoder_path', None)
        self.text_encoder_arch: str = kwargs.get('text_encoder_arch', 'clip')  # clip t5

        self.train_scaler: bool = kwargs.get('train_scaler', False)
        self.scaler_lr: Optional[float] = kwargs.get('scaler_lr', None)

        # trains with a scaler to easy channel bias but merges it in on save
        self.merge_scaler: bool = kwargs.get('merge_scaler', False)

        # for ilora
        self.head_dim: int = kwargs.get('head_dim', 1024)
        self.num_heads: int = kwargs.get('num_heads', 1)
        self.ilora_down: bool = kwargs.get('ilora_down', True)
        self.ilora_mid: bool = kwargs.get('ilora_mid', True)
        self.ilora_up: bool = kwargs.get('ilora_up', True)
        
        self.pixtral_max_image_size: int = kwargs.get('pixtral_max_image_size', 512)
        self.pixtral_random_image_size: int = kwargs.get('pixtral_random_image_size', False)

        self.flux_only_double: bool = kwargs.get('flux_only_double', False)
        
        # train and use a conv layer to pool the embedding
        self.conv_pooling: bool = kwargs.get('conv_pooling', False)
        self.conv_pooling_stacks: int = kwargs.get('conv_pooling_stacks', 1)
        self.sparse_autoencoder_dim: Optional[int] = kwargs.get('sparse_autoencoder_dim', None)
        
        # for llm adapter
        self.num_cloned_blocks: int = kwargs.get('num_cloned_blocks', 0)
        self.quantize_llm: bool = kwargs.get('quantize_llm', False)
        
        # for control lora only
        lora_config: dict = kwargs.get('lora_config', None)
        if lora_config is not None:
            self.lora_config: NetworkConfig = NetworkConfig(**lora_config)
        else:
            self.lora_config = None
        self.num_control_images: int = kwargs.get('num_control_images', 1)
        # decimal for how often the control is dropped out and replaced with noise 1.0 is 100%
        self.control_image_dropout: float = kwargs.get('control_image_dropout', 0.0)
        self.has_inpainting_input: bool = kwargs.get('has_inpainting_input', False)
        self.invert_inpaint_mask_chance: float = kwargs.get('invert_inpaint_mask_chance', 0.0)
        
        # for subpixel adapter
        self.subpixel_downscale_factor: int = kwargs.get('subpixel_downscale_factor', 8)
        
        # for i2v adapter
        # append the masked start frame. During pretraining we will only do the vision encoder
        self.i2v_do_start_frame: bool = kwargs.get('i2v_do_start_frame', False)


class EmbeddingConfig:
    def __init__(self, **kwargs):
        self.trigger = kwargs.get('trigger', 'custom_embedding')
        self.tokens = kwargs.get('tokens', 4)
        self.init_words = kwargs.get('init_words', '*')
        self.save_format = kwargs.get('save_format', 'safetensors')
        self.trigger_class_name = kwargs.get('trigger_class_name', None)  # used for inverted masked prior


class DecoratorConfig:
    def __init__(self, **kwargs):
        self.num_tokens: str = kwargs.get('num_tokens', 4)


ContentOrStyleType = Literal['balanced', 'style', 'content']
LossTarget = Literal['noise', 'source', 'unaugmented', 'differential_noise']


class TrainConfig:
    def __init__(self, **kwargs):
        self.noise_scheduler = kwargs.get('noise_scheduler', 'ddpm')
        self.content_or_style: ContentOrStyleType = kwargs.get('content_or_style', 'balanced')
        self.content_or_style_reg: ContentOrStyleType = kwargs.get('content_or_style', 'balanced')
        self.steps: int = kwargs.get('steps', 1000)
        self.lr = kwargs.get('lr', 1e-6)
        self.unet_lr = kwargs.get('unet_lr', self.lr)
        self.text_encoder_lr = kwargs.get('text_encoder_lr', self.lr)
        self.refiner_lr = kwargs.get('refiner_lr', self.lr)
        self.embedding_lr = kwargs.get('embedding_lr', self.lr)
        self.adapter_lr = kwargs.get('adapter_lr', self.lr)
        self.optimizer = kwargs.get('optimizer', 'adamw')
        self.optimizer_params = kwargs.get('optimizer_params', {})
        self.lr_scheduler = kwargs.get('lr_scheduler', 'none')
        self.lr_scheduler_params = kwargs.get('lr_scheduler_params', {})
        # Per-expert learning rate schedulers for dual-expert models (e.g., Wan 2.2 14B).
        # Each expert gets its own scheduler that tracks steps based on how many times
        # that expert was active (not global step count).
        # If not specified, each expert will get its own copy of the global scheduler
        # configured to work on expert-specific step counts.
        # If specified, the per-expert scheduler config overwrites the global scheduler for that expert.
        self.expert_1_lr_scheduler = kwargs.get('expert_1_lr_scheduler', None)  # high-noise expert
        self.expert_1_lr_scheduler_params = kwargs.get('expert_1_lr_scheduler_params', None)
        self.expert_2_lr_scheduler = kwargs.get('expert_2_lr_scheduler', None)  # low-noise expert
        self.expert_2_lr_scheduler_params = kwargs.get('expert_2_lr_scheduler_params', None)
        self.min_denoising_steps: int = kwargs.get('min_denoising_steps', 0)
        self.max_denoising_steps: int = kwargs.get('max_denoising_steps', 999)
        self.batch_size: int = kwargs.get('batch_size', 1)
        self.orig_batch_size: int = self.batch_size
        self.dtype: str = kwargs.get('dtype', 'fp32')
        self.xformers = kwargs.get('xformers', False)
        self.sdp = kwargs.get('sdp', False)
        # see https://huggingface.co/docs/diffusers/main/optimization/attention_backends#available-backends for options
        # Values used by the Wan 2.x toolkit attention path:
        #   native (default) = auto: flex_attention when softcapping is enabled, else SDPA
        #   flex = torch flex_attention (applies tanh softcapping via a score_mod)
        #   sdpa = PyTorch scaled_dot_product_attention (auto flash/mem-efficient kernels)
        #   flash = flash-attn v2 package (requires `pip install flash-attn`, fp16/bf16;
        #           applies tanh softcapping natively in-kernel, 2.8.3+)
        # NOTE: for Wan 2.x the transformer's custom attention processor honors this
        # setting directly; for other models it is forwarded to diffusers
        # set_attention_backend() on the VAE/text encoder where available.
        # NOTE: when attention_tanh_softcap_enabled is true, the cap is applied by the
        # selected kernel - flash natively (2.8.3+), or flex via score_mod for
        # native/sdpa/flex (SDPA itself has no score hook). fp32 layers skip the
        # cap under flash. See toolkit/models/wan21/wan_attn.py.
        self.attention_backend: str = kwargs.get('attention_backend', 'native')  # native, flash, _flash_3_hub, _flash_3, 
        self.train_unet = kwargs.get('train_unet', True)
        self.train_text_encoder = kwargs.get('train_text_encoder', False)
        self.train_refiner = kwargs.get('train_refiner', True)
        self.train_turbo = kwargs.get('train_turbo', False)
        self.show_turbo_outputs = kwargs.get('show_turbo_outputs', False)
        self.min_snr_gamma = kwargs.get('min_snr_gamma', None)
        self.snr_gamma = kwargs.get('snr_gamma', None)
        # trains a gamma, offset, and scale to adjust loss to adapt to timestep differentials
        # this should balance the learning rate across all timesteps over time
        self.learnable_snr_gos = kwargs.get('learnable_snr_gos', False)
        self.noise_offset = kwargs.get('noise_offset', 0.0)
        self.skip_first_sample = kwargs.get('skip_first_sample', False)
        self.force_first_sample = kwargs.get('force_first_sample', False)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
        self.weight_jitter = kwargs.get('weight_jitter', 0.0)
        self.merge_network_on_save = kwargs.get('merge_network_on_save', False)
        self.merge_network_on_save_strength = kwargs.get('merge_network_on_save_strength', 1.0)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.start_step = kwargs.get('start_step', None)
        self.free_u = kwargs.get('free_u', False)
        self.adapter_assist_name_or_path: Optional[str] = kwargs.get('adapter_assist_name_or_path', None)
        self.adapter_assist_type: Optional[str] = kwargs.get('adapter_assist_type', 't2i')  # t2i, control_net
        self.noise_multiplier = kwargs.get('noise_multiplier', 1.0)
        self.target_noise_multiplier = kwargs.get('target_noise_multiplier', 1.0)
        self.random_noise_multiplier = kwargs.get('random_noise_multiplier', 0.0)
        self.do_signal_correction_noise = kwargs.get('do_signal_correction_noise', False)
        # batch noise correction adds other images in the batch as noise to correct away from other images
        self.do_batch_noise_correction = kwargs.get('do_batch_noise_correction', False)
        self.batch_noise_correction_scale = kwargs.get('batch_noise_correction_scale', 0.1)
        self.do_signal_amplification = kwargs.get('do_signal_amplification', False)
        self.signal_amplification_strength = kwargs.get('signal_amplification_strength', 0.5)
        
        self.signal_correction_noise_scale = kwargs.get('signal_correction_noise_scale', 1.0)
        self.random_noise_shift = kwargs.get('random_noise_shift', 0.0)
        self.img_multiplier = kwargs.get('img_multiplier', 1.0)
        self.noisy_latent_multiplier = kwargs.get('noisy_latent_multiplier', 1.0)
        self.latent_multiplier = kwargs.get('latent_multiplier', 1.0)
        self.negative_prompt = kwargs.get('negative_prompt', None)
        self.max_negative_prompts = kwargs.get('max_negative_prompts', 1)
        # multiplier applied to loos on regularization images
        self.reg_weight = kwargs.get('reg_weight', 1.0)
        self.num_train_timesteps = kwargs.get('num_train_timesteps', 1000)
        # automatically adapte the vae scaling based on the image norm
        self.adaptive_scaling_factor = kwargs.get('adaptive_scaling_factor', False)

        # Attention tanh softcapping - prevents attention scores from becoming too extreme
        # Inspired by Gemma2 and Grok-1. Applies: soft_cap * tanh(score / soft_cap) before softmax
        # Helps with training stability by avoiding overly sharp attention distributions
        # Requires PyTorch 2.5+ with flex_attention support
        # Hierarchy: per-type-per-expert → per-type → per-expert → global
        self.attention_tanh_softcap_enabled = kwargs.get('attention_tanh_softcap_enabled', True)
        self.attention_tanh_softcap_value = kwargs.get('attention_tanh_softcap_value', 30.0)
        
        # Per-attention-type overrides (applies to both experts)
        self.attention_tanh_softcap_value_self_attn = kwargs.get('attention_tanh_softcap_value_self_attn', None)
        self.attention_tanh_softcap_value_cross_attn = kwargs.get('attention_tanh_softcap_value_cross_attn', None)
        
        # Per-expert overrides (applies to both attention types)
        self.attention_tanh_softcap_value_high_noise = kwargs.get('attention_tanh_softcap_value_high_noise', None)
        self.attention_tanh_softcap_value_low_noise = kwargs.get('attention_tanh_softcap_value_low_noise', None)
        
        # Per-type-per-expert overrides (most specific)
        self.attention_tanh_softcap_value_self_attn_high_noise = kwargs.get('attention_tanh_softcap_value_self_attn_high_noise', None)
        self.attention_tanh_softcap_value_self_attn_low_noise = kwargs.get('attention_tanh_softcap_value_self_attn_low_noise', None)
        self.attention_tanh_softcap_value_cross_attn_high_noise = kwargs.get('attention_tanh_softcap_value_cross_attn_high_noise', None)
        self.attention_tanh_softcap_value_cross_attn_low_noise = kwargs.get('attention_tanh_softcap_value_cross_attn_low_noise', None)

        # Attention F32 acceleration - use float32 instead of float64 for rotary embeddings
        # Default toolkit uses float64 for maximum precision (slow), diffusers uses input dtype (fast, less stable)
        # F32 is a good middle ground: faster than F64, more stable than BF16/FP16 for RoPE
        self.attention_f32_rope_enabled = kwargs.get('attention_f32_rope_enabled', True)

        # GELU acceleration for Wan 2.x FeedForward layers
        # Patches diffusers' GELU to use tanh.approx.f32 PTX instruction (~2-5% FF speedup)
        # NOTE: Global monkeypatch - only enable when training Wan 2.x models
        self.gelu_acceleration_enabled = kwargs.get('gelu_acceleration_enabled', True)

        # dropout that happens before encoding. It functions independently per text encoder
        self.prompt_dropout_prob = kwargs.get('prompt_dropout_prob', 0.0)

        # match the norm of the noise before computing loss. This will help the model maintain its
        # current understandin of the brightness of images.

        self.match_noise_norm = kwargs.get('match_noise_norm', False)

        # set to -1 to accumulate gradients for entire epoch
        # warning, only do this with a small dataset or you will run out of memory
        # This is legacy but left in for backwards compatibility
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)

        # this will do proper gradient accumulation where you will not see a step until the end of the accumulation
        # the method above will show a step every accumulation
        self.gradient_accumulation = kwargs.get('gradient_accumulation', 1)
        if self.gradient_accumulation > 1:
            if self.gradient_accumulation_steps != 1:
                raise ValueError("gradient_accumulation and gradient_accumulation_steps are mutually exclusive")

        # short long captions will double your batch size. This only works when a dataset is
        # prepared with a json caption file that has both short and long captions in it. It will
        # Double up every image and run it through with both short and long captions. The idea
        # is that the network will learn how to generate good images with both short and long captions
        self.short_and_long_captions = kwargs.get('short_and_long_captions', False)
        # if above is NOT true, this will make it so the long caption foes to te2 and the short caption goes to te1 for sdxl only
        self.short_and_long_captions_encoder_split = kwargs.get('short_and_long_captions_encoder_split', False)

        # basically gradient accumulation but we run just 1 item through the network
        # and accumulate gradients. This can be used as basic gradient accumulation but is very helpful
        # for training tricks that increase batch size but need a single gradient step
        self.single_item_batching = kwargs.get('single_item_batching', False)

        match_adapter_assist = kwargs.get('match_adapter_assist', False)
        self.match_adapter_chance = kwargs.get('match_adapter_chance', 0.0)
        self.loss_target: LossTarget = kwargs.get('loss_target',
                                                  'noise')  # noise, source, unaugmented, differential_noise

        # When a mask is passed in a dataset, and this is true,
        # we will predict noise without a the LoRa network and use the prediction as a target for
        # unmasked reign. It is unmasked regularization basically
        self.inverted_mask_prior = kwargs.get('inverted_mask_prior', False)
        self.inverted_mask_prior_multiplier = kwargs.get('inverted_mask_prior_multiplier', 0.5)
        
        # DOP will will run the same image and prompt through the network without the trigger word blank and use it as a target
        self.diff_output_preservation = kwargs.get('diff_output_preservation', False)
        self.diff_output_preservation_multiplier = kwargs.get('diff_output_preservation_multiplier', 1.0)
        # If the trigger word is in the prompt, we will use this class name to replace it eg. "sks woman" -> "woman"
        self.diff_output_preservation_class = kwargs.get('diff_output_preservation_class', '')
        
        # blank prompt preservation will preserve the model's knowledge of a blank prompt
        self.blank_prompt_preservation = kwargs.get('blank_prompt_preservation', False)
        self.blank_prompt_preservation_multiplier = kwargs.get('blank_prompt_preservation_multiplier', 1.0)
        
        # legacy
        if match_adapter_assist and self.match_adapter_chance == 0.0:
            self.match_adapter_chance = 1.0

        # standardize inputs to the meand std of the model knowledge
        self.standardize_images = kwargs.get('standardize_images', False)
        self.standardize_latents = kwargs.get('standardize_latents', False)

        # if self.train_turbo and not self.noise_scheduler.startswith("euler"):
        #     raise ValueError(f"train_turbo is only supported with euler and wuler_a noise schedulers")

        self.dynamic_noise_offset = kwargs.get('dynamic_noise_offset', False)
        self.do_cfg = kwargs.get('do_cfg', False)
        self.do_random_cfg = kwargs.get('do_random_cfg', False)
        # when True, the unconditional (negative) side of training-time CFG uses the
        # same prompt as the conditional (positive) side instead of the negative prompt
        self.cfg_same_prompt = kwargs.get('cfg_same_prompt', False)
        self.cfg_scale = kwargs.get('cfg_scale', 1.0)
        self.max_cfg_scale = kwargs.get('max_cfg_scale', self.cfg_scale)
        self.cfg_rescale = kwargs.get('cfg_rescale', None)
        if self.cfg_rescale is None:
            self.cfg_rescale = self.cfg_scale

        # ------------------------------------------------------------------
        # Conditioning dropout (text + image), positive & negative branches.
        #
        # These are GLOBAL rates. A per-dataset value (DatasetConfig) overrides
        # the global value when it is set (not None). All rates are in [0, 1] and
        # are applied independently per training item, per step.
        #
        #   * text_dropout_rate(_negative)     -> drop the prompt (-> blank) for the
        #                                          positive / negative text branch.
        #   * image_dropout_rate(_negative)    -> drop the I2V first-frame image
        #                                          conditioning for the positive / negative
        #                                          image branch (makes the item T2V).
        #
        # The negative branch only exists when CFG is active (do_cfg / do_random_cfg).
        # Without CFG only the positive rates apply.
        # ------------------------------------------------------------------
        # Positive-branch text drop rate (global). None = use per-dataset then legacy prompt_dropout_prob.
        self.text_dropout_rate: Optional[float] = kwargs.get('text_dropout_rate', None)
        # Negative-branch text drop rate (global). Only used when CFG is active.
        self.text_dropout_rate_negative: Optional[float] = kwargs.get('text_dropout_rate_negative', None)
        # Positive-branch image (I2V first frame) drop rate (global).
        self.image_dropout_rate: Optional[float] = kwargs.get('image_dropout_rate', None)
        # Negative-branch image drop rate (global). Only used when CFG is active.
        # When cfg_same_prompt is on and this is None, it defaults to 1.0 (always drop,
        # preserving the original cfg_same_prompt behavior of a fully-unconditional image side).
        self.image_dropout_rate_negative: Optional[float] = kwargs.get('image_dropout_rate_negative', None)

        # Force the negative branch to share the SAME drop STATE as the positive branch
        # for the same item/step (positive dropped  =>  negative dropped).
        self.sync_text_dropout: bool = kwargs.get('sync_text_dropout', False)
        self.sync_image_dropout: bool = kwargs.get('sync_image_dropout', False)
        # Invert the synced relationship: positive dropped  =>  negative NOT dropped
        # (only meaningful when the corresponding sync_* toggle is on).
        self.invert_text_dropout: bool = kwargs.get('invert_text_dropout', False)
        self.invert_image_dropout: bool = kwargs.get('invert_image_dropout', False)

        # applies the inverse of the prediction mean and std to the target to correct
        # for norm drift
        self.correct_pred_norm = kwargs.get('correct_pred_norm', False)
        self.correct_pred_norm_multiplier = kwargs.get('correct_pred_norm_multiplier', 1.0)

        self.loss_type = kwargs.get('loss_type', 'mse')  # mse, mae, wavelet, spectral, spectral_flow, pixelspace, mean_flow, pseudo_huber
        self.pseudo_huber_c = kwargs.get('pseudo_huber_c', 0.01)  # c value for pseudo_huber loss

        # Spectral loss config - frequency dissociation and balancing
        # Low freq = structure/motion, High freq = texture/details
        # Inspired by SSVAE research on latent spectral biasing for superior diffusability
        # GLOBAL weights (used for single-expert models or as fallback)
        self.spectral_low_weight = kwargs.get('spectral_low_weight', 1.0)   # structure/motion weight
        self.spectral_mid_weight = kwargs.get('spectral_mid_weight', 1.0)   # mid frequencies weight
        self.spectral_high_weight = kwargs.get('spectral_high_weight', 2.0) # texture/details weight (emphasize)
        
        # PER-EXPERT spectral weights (Wan 2.2 14B and other MoE models)
        # For high-noise expert (t > boundary): focus on structure/motion
        self.spectral_low_weight_high = kwargs.get('spectral_low_weight_high', None)
        self.spectral_mid_weight_high = kwargs.get('spectral_mid_weight_high', None)
        self.spectral_high_weight_high = kwargs.get('spectral_high_weight_high', None)
        # For low-noise expert (t <= boundary): focus on texture/details
        self.spectral_low_weight_low = kwargs.get('spectral_low_weight_low', None)
        self.spectral_mid_weight_low = kwargs.get('spectral_mid_weight_low', None)
        self.spectral_high_weight_low = kwargs.get('spectral_high_weight_low', None)
        
        # PER-EXPERT frequency cutoffs (optional override)
        # Allow different band boundaries per expert if desired
        self.spectral_low_cutoff_high = kwargs.get('spectral_low_cutoff_high', None)
        self.spectral_high_cutoff_high = kwargs.get('spectral_high_cutoff_high', None)
        self.spectral_low_cutoff_low = kwargs.get('spectral_low_cutoff_low', None)
        self.spectral_high_cutoff_low = kwargs.get('spectral_high_cutoff_low', None)
        
        # Global frequency cutoffs (fallback)
        self.spectral_low_cutoff = kwargs.get('spectral_low_cutoff', 0.15)  # radius separating low/mid freq
        self.spectral_high_cutoff = kwargs.get('spectral_high_cutoff', 0.5) # radius separating mid/high freq
        
        self.spectral_use_phase = kwargs.get('spectral_use_phase', True)    # use phase info (more accurate)
        self.spectral_lcr_weight = kwargs.get('spectral_lcr_weight', 0.0)   # SSVAE LCR weight (0.0 = disabled)
        self.spectral_transform = kwargs.get('spectral_transform', 'dct')   # 'dct' (default, SSVAE-compliant) or 'fft'
        self.prediction_target = kwargs.get('prediction_target', 'velocity')  # 'velocity' (default) or 'x0'
        
        # Spectral temporal scale for video: controls how much temporal frequency
        # contributes to the 3D frequency mask.
        # 1.0 = pure spherical (all dims equal, can cause motion artifacts)
        # 0.3 = recommended for video (temporal down-weighted)
        # 0.0 = spatial-only (ignores temporal frequency)
        # PER-EXPERT temporal scale (optional)
        self.spectral_temporal_scale_high = kwargs.get('spectral_temporal_scale_high', None)
        self.spectral_temporal_scale_low = kwargs.get('spectral_temporal_scale_low', None)
        # Global temporal scale (fallback)
        self.spectral_temporal_scale = kwargs.get('spectral_temporal_scale', 0.3)

        # Spectral loss weight for combined losses (spectral_flow, mse_spectral_flow)
        # Controls overall spectral component magnitude relative to MSE/flow
        # Note: spectral_low/mid/high_weight control internal frequency balance only
        # GLOBAL spectral weight (used for single-expert models or as fallback)
        self.spectral_weight = kwargs.get('spectral_weight', 1.0)  # global spectral weight (fallback)
        # PER-EXPERT spectral weights (MoE models like Wan 2.2 14B)
        self.spectral_weight_high = kwargs.get('spectral_weight_high', None)  # per-expert spectral weight for high noise
        self.spectral_weight_low = kwargs.get('spectral_weight_low', None)    # per-expert spectral weight for low noise

        # Spectral flow loss config - combines spectral (spatial frequency) + optical flow (temporal motion)
        # Flow loss weight: 0.05-0.15 recommended for Wan 2.2 I2V LoRA
        self.spectral_flow_weight = kwargs.get('spectral_flow_weight', 0.1)  # global flow loss weight (fallback)
        self.spectral_flow_weight_low = kwargs.get('spectral_flow_weight_low', None)  # per-expert weight for low noise
        self.spectral_flow_weight_high = kwargs.get('spectral_flow_weight_high', None)  # per-expert weight for high noise
        self.spectral_flow_max_timestep = kwargs.get('spectral_flow_max_timestep', 800)  # timestep gate for flow loss
        self.spectral_flow_reverse_gate = kwargs.get('spectral_flow_reverse_gate', False)  # reverse gate: more weight at high noise
        self.spectral_flow_motion_weighted = kwargs.get('spectral_flow_motion_weighted', True)  # weight by motion magnitude
        self.spectral_flow_adaptive = kwargs.get('spectral_flow_adaptive', False)  # dynamic weight adjustment
        self.spectral_flow_rejection_threshold = kwargs.get('spectral_flow_rejection_threshold', 5.0)  # deviation threshold
        self.spectral_flow_max_rejections = kwargs.get('spectral_flow_max_rejections', 100)  # per-expert rejection budget
        
        # Step loss rejection config - reject optimizer steps that exceed loss thresholds
        self.spectral_flow_loss_rejection_enabled = kwargs.get('spectral_flow_loss_rejection_enabled', False)
        self.spectral_flow_loss_rejection_max_low = kwargs.get('spectral_flow_loss_rejection_max_low', 7.0)  # max loss for low noise expert
        self.spectral_flow_loss_rejection_max_high = kwargs.get('spectral_flow_loss_rejection_max_high', 14.0)  # max loss for high noise expert
        self.spectral_flow_loss_rejection_max_increase_pct = kwargs.get('spectral_flow_loss_rejection_max_increase_pct', 20.0)  # max % increase from prev step
        self.spectral_flow_loss_rejection_max_retries = kwargs.get('spectral_flow_loss_rejection_max_retries', 5)  # max retries per step
        
        # Constraint-based rejection: require spectral loss decrease while bounding flow loss increase
        self.spectral_flow_constraint_rejection_enabled = kwargs.get('spectral_flow_constraint_rejection_enabled', False)
        self.spectral_flow_constraint_flow_max_increase_pct = kwargs.get('spectral_flow_constraint_flow_max_increase_pct', 5.0)  # max flow loss increase allowed
        self.spectral_flow_constraint_spectral_must_decrease = kwargs.get('spectral_flow_constraint_spectral_must_decrease', True)  # require spectral to decrease
        
        # Gradient projection (PCGrad-style): when spectral and flow gradients conflict,
        # project spectral gradient to remove component that worsens flow loss
        self.spectral_flow_gradient_projection_enabled = kwargs.get('spectral_flow_gradient_projection_enabled', False)
        
        # MSE + Spectral + Flow loss config - combines all three loss types
        # MSE weight: 0.5-2.0 recommended for Wan 2.2 I2V LoRA
        self.mse_spectral_flow_mse_weight = kwargs.get('mse_spectral_flow_mse_weight', 1.0)  # global MSE weight (fallback)
        self.mse_spectral_flow_mse_weight_low = kwargs.get('mse_spectral_flow_mse_weight_low', None)  # per-expert MSE weight for low noise
        self.mse_spectral_flow_mse_weight_high = kwargs.get('mse_spectral_flow_mse_weight_high', None)  # per-expert MSE weight for high noise
        
        # Gradient projection (PCGrad-style) for MSE+Spectral+Flow: when any gradients conflict,
        # project them to remove components that worsen other losses
        self.mse_spectral_flow_gradient_projection_enabled = kwargs.get('mse_spectral_flow_gradient_projection_enabled', False)
        
        # Force every item in a batch to share a single (randomly drawn) timestep instead of
        # each item getting its own random one. None (default) = automatic: enabled whenever
        # TREAD token routing is active (global/per-expert enabled, or a per-timestep TREAD
        # range that enables routing), because TREAD resolves its per-timestep settings from
        # the batch's global timestep - and forced on even if explicitly set to false while
        # TREAD routing is active. Set true manually to share one timestep without TREAD.
        self.force_same_timestep_per_batch = kwargs.get('force_same_timestep_per_batch', None)

        # Per-timestep range loss weight overrides
        # Allows specifying different loss weights for different timestep ranges per model.
        # Ranges are in absolute model timesteps (0-1000); no scaling is applied.
        # Multiple ranges can be specified; first matching range wins.
        # Ranges are in absolute model timesteps (0-1000). Each expert dynamically
        # checks if its current timestep falls within a range.
        # Format: list of dicts with keys:
        #   start_timestep: start of range (absolute, 0-1000)
        #   end_timestep: end of range (absolute, 0-1000)
        #   flow_weight: optional flow weight override
        #   spectral_weight: optional spectral weight override
        #   spectral_low_weight: optional low freq weight override
        #   spectral_mid_weight: optional mid freq weight override
        #   spectral_high_weight: optional high freq weight override
        #   mse_weight: optional MSE weight override
        #   spectral_low_cutoff: optional low frequency cutoff override
        #   spectral_high_cutoff: optional high frequency cutoff override
        #   spectral_lcr_weight: optional LCR (Low-Cut Ratio) weight override
        #   spectral_temporal_scale: optional temporal scale override
        self.timestep_range_overrides = kwargs.get('timestep_range_overrides', [])
        
        # do the loss on a timestep to 0 prediction
        self.t0_loss_target = kwargs.get('t0_loss_target', False)
        self.t0_velocity_equiv_weight = kwargs.get('t0_velocity_equiv_weight', False)
        
        # do additional fft loss
        self.do_fft_loss = kwargs.get('do_fft_loss', False)
        self.do_fft_velocity_equiv_weight = kwargs.get('do_fft_velocity_equiv_weight', False)

        # scale the prediction by this. Increase for more detail, decrease for less
        self.pred_scaler = kwargs.get('pred_scaler', 1.0)

        # repeats the prompt a few times to saturate the encoder
        self.prompt_saturation_chance = kwargs.get('prompt_saturation_chance', 0.0)

        # applies negative loss on the prior to encourage network to diverge from it
        self.do_prior_divergence = kwargs.get('do_prior_divergence', False)

        ema_config: Union[Dict, None] = kwargs.get('ema_config', None)
        # if ema config exists and use_ema is not false result to True, otherwise False (behaviour before fix -> if ema_config is not None use_ema is always True, even when set false, otherwise False)
        if isinstance(ema_config, dict):
            ema_config['use_ema'] = ema_config.get('use_ema', True)
        else:
            ema_config = {'use_ema': False}
        print(f"Using EMA: {ema_config['use_ema']}")

        self.ema_config: EMAConfig = EMAConfig(**ema_config)

        # adds an additional loss to the network to encourage it output a normalized standard deviation
        self.target_norm_std = kwargs.get('target_norm_std', None)
        self.target_norm_std_value = kwargs.get('target_norm_std_value', 1.0)
        self.timestep_type = kwargs.get('timestep_type', 'sigmoid')  # sigmoid, linear, lognorm_blend, next_sample, weighted, one_step
        self.next_sample_timesteps = kwargs.get('next_sample_timesteps', 8)
        self.linear_timesteps = kwargs.get('linear_timesteps', False)
        self.linear_timesteps2 = kwargs.get('linear_timesteps2', False)
        self.disable_sampling = kwargs.get('disable_sampling', False)

        # will cache a blank prompt or the trigger word, and unload the text encoder to cpu
        # will make training faster and use less vram
        self.unload_text_encoder = kwargs.get('unload_text_encoder', False)
        # will toggle all datasets to cache text embeddings
        self.cache_text_embeddings: bool = kwargs.get('cache_text_embeddings', False)
        # for swapping which parameters are trained during training
        self.do_paramiter_swapping = kwargs.get('do_paramiter_swapping', False)
        # 0.1 is 10% of the parameters active at a time lower is less vram, higher is more
        self.paramiter_swapping_factor = kwargs.get('paramiter_swapping_factor', 0.1)
        # bypass the guidance embedding for training. For open flux with guidance embedding
        self.bypass_guidance_embedding = kwargs.get('bypass_guidance_embedding', False)
        
        # diffusion feature extractor
        self.latent_feature_extractor_path = kwargs.get('latent_feature_extractor_path', None)
        self.latent_feature_loss_weight = kwargs.get('latent_feature_loss_weight', 1.0)
        
        # we use this in the code, but it really needs to be called latent_feature_extractor as that makes more sense with new architecture
        self.diffusion_feature_extractor_path = kwargs.get('diffusion_feature_extractor_path', self.latent_feature_extractor_path)
        self.diffusion_feature_extractor_weight = kwargs.get('diffusion_feature_extractor_weight', self.latent_feature_loss_weight)
        
        # optimal noise pairing
        self.optimal_noise_pairing_samples = kwargs.get('optimal_noise_pairing_samples', 1)
        
        # forces same noise for the same image at a given size.
        self.force_consistent_noise = kwargs.get('force_consistent_noise', False)
        self.blended_blur_noise = kwargs.get('blended_blur_noise', False)
        
        # contrastive loss
        self.do_guidance_loss = kwargs.get('do_guidance_loss', False)
        self.guidance_loss_target: Union[int, List[int, int]] = kwargs.get('guidance_loss_target', 3.0)
        self.do_guidance_loss_cfg_zero: bool = kwargs.get('do_guidance_loss_cfg_zero', False)
        self.unconditional_prompt: str = kwargs.get('unconditional_prompt', '')
        if isinstance(self.guidance_loss_target, tuple):
            self.guidance_loss_target = list(self.guidance_loss_target)

        self.do_differential_guidance = kwargs.get('do_differential_guidance', False)
        self.differential_guidance_scale = kwargs.get('differential_guidance_scale', 3.0)

        # for multi stage models, how often to switch the boundary
        self.switch_boundary_every: int = kwargs.get('switch_boundary_every', 1)

        # stabilizes empty prompts to be zeroed predictions
        self.do_blank_stabilization = kwargs.get('do_blank_stabilization', False)
        
        self.audio_loss_multiplier = kwargs.get("audio_loss_multiplier", 1.0)
        
        # will throw detailed error when it goes over
        self.max_loss_debug: bool = kwargs.get("max_loss_debug", False)
        # will clip the loss to this amount to prevent wild outliers
        self.max_loss: Optional[float] = kwargs.get("max_loss", None)


ModelArch = Literal['sd1', 'sd2', 'sd3', 'sdxl', 'pixart', 'pixart_sigma', 'auraflow', 'flux', 'flex1', 'flex2', 'lumina2', 'vega', 'ssd', 'wan21']


class ModelConfig:
    def __init__(self, **kwargs):
        self.name_or_path: str = kwargs.get('name_or_path', None)
        # name or path is updated on fine tuning. Keep a copy of the original
        self.name_or_path_original: str = self.name_or_path
        self.is_v2: bool = kwargs.get('is_v2', False)
        self.is_xl: bool = kwargs.get('is_xl', False)
        self.is_pixart: bool = kwargs.get('is_pixart', False)
        self.is_pixart_sigma: bool = kwargs.get('is_pixart_sigma', False)
        self.is_auraflow: bool = kwargs.get('is_auraflow', False)
        self.is_v3: bool = kwargs.get('is_v3', False)
        self.is_flux: bool = kwargs.get('is_flux', False)
        self.is_lumina2: bool = kwargs.get('is_lumina2', False)
        if self.is_pixart_sigma:
            self.is_pixart = True
        self.use_flux_cfg = kwargs.get('use_flux_cfg', False)
        self.is_ssd: bool = kwargs.get('is_ssd', False)
        self.is_vega: bool = kwargs.get('is_vega', False)
        self.is_v_pred: bool = kwargs.get('is_v_pred', False)
        self.dtype: str = kwargs.get('dtype', 'float16')
        self.vae_path = kwargs.get('vae_path', None)
        self.refiner_name_or_path = kwargs.get('refiner_name_or_path', None)
        self._original_refiner_name_or_path = self.refiner_name_or_path
        self.refiner_start_at = kwargs.get('refiner_start_at', 0.5)
        self.lora_path = kwargs.get('lora_path', None)
        # mainly for decompression loras for distilled models
        self.assistant_lora_path = kwargs.get('assistant_lora_path', None)
        self.inference_lora_path = kwargs.get('inference_lora_path', None)
        # a lora that stays inactive except during the unconditional (negative)
        # CFG pass -- used to learn the unconditional branch without a second model
        self.unconditional_lora_path = kwargs.get('unconditional_lora_path', None)
        self.latent_space_version = kwargs.get('latent_space_version', None)

        # only for SDXL models for now
        self.use_text_encoder_1: bool = kwargs.get('use_text_encoder_1', True)
        self.use_text_encoder_2: bool = kwargs.get('use_text_encoder_2', True)

        self.experimental_xl: bool = kwargs.get('experimental_xl', False)

        if self.name_or_path is None:
            raise ValueError('name_or_path must be specified')

        if self.is_ssd:
            # sed sdxl as true since it is mostly the same architecture
            self.is_xl = True

        if self.is_vega:
            self.is_xl = True

        # for text encoder quant. Only works with pixart currently
        self.text_encoder_bits = kwargs.get('text_encoder_bits', 16)  # 16, 8, 4
        self.unet_path = kwargs.get("unet_path", None)
        self.unet_sample_size = kwargs.get("unet_sample_size", None)
        self.vae_device = kwargs.get("vae_device", None)
        self.vae_dtype = kwargs.get("vae_dtype", self.dtype)
        self.te_device = kwargs.get("te_device", None)
        self.te_dtype = kwargs.get("te_dtype", self.dtype)

        # only for flux for now
        self.quantize = kwargs.get("quantize", False)
        self.quantize_te = kwargs.get("quantize_te", self.quantize)
        self.qtype = kwargs.get("qtype", "qfloat8")
        self.qtype_te = kwargs.get("qtype_te", "qfloat8")
        self.low_vram = kwargs.get("low_vram", False)
        self.attn_masking = kwargs.get("attn_masking", False)
        if self.attn_masking and not self.is_flux:
            raise ValueError("attn_masking is only supported with flux models currently")
        # for targeting a specific layers
        self.ignore_if_contains: Optional[List[str]] = kwargs.get("ignore_if_contains", None)
        self.only_if_contains: Optional[List[str]] = kwargs.get("only_if_contains", None)
        self.quantize_kwargs = kwargs.get("quantize_kwargs", {})
        
        # splits the model over the available gpus WIP
        self.split_model_over_gpus = kwargs.get("split_model_over_gpus", False)
        if self.split_model_over_gpus and not self.is_flux:
            raise ValueError("split_model_over_gpus is only supported with flux models currently")
        self.split_model_other_module_param_count_scale = kwargs.get("split_model_other_module_param_count_scale", 0.3)
        
        self.te_name_or_path = kwargs.get("te_name_or_path", None)
        
        self.arch: ModelArch = kwargs.get("arch", None)
        
        # auto memory management, only for some models
        self.auto_memory = kwargs.get("auto_memory", False)
        # auto memory is deprecated, use layer offloading instead
        if self.auto_memory:
            print("auto_memory is deprecated, use layer_offloading instead")
        self.layer_offloading = kwargs.get("layer_offloading", self.auto_memory )
        if self.layer_offloading and self.qtype == "qfloat8":
            self.qtype = "float8"
        if self.layer_offloading and self.qtype_te == "qfloat8":
            self.qtype_te = "float8"
            
        # Mac mps only works with torachao uint
        if torch.backends.mps.is_available() and self.qtype == "qfloat8":
            self.qtype = "int8"
        if torch.backends.mps.is_available() and self.qtype_te == "qfloat8":
            self.qtype_te = "int8"
        
        # 0 is off and 1.0 is 100% of the layers
        self.layer_offloading_transformer_percent = kwargs.get("layer_offloading_transformer_percent", 1.0)
        self.layer_offloading_text_encoder_percent = kwargs.get("layer_offloading_text_encoder_percent", 1.0)

        # can be used to load the extras like text encoder or vae from here
        # only setup for some models but will prevent having to download the te for
        # 20 different model variants
        self.extras_name_or_path = kwargs.get("extras_name_or_path", self.name_or_path)

        # Custom VAE path - overrides the default _wan_vae_path for Wan models.
        # Can be a HuggingFace repo ID (e.g. "ai-toolkit/wan2.1-vae") or a local
        # directory path. When set, it takes precedence over the model's
        # class-level _wan_vae_path attribute. Useful for loading custom or
        # alternative VAE variants (e.g. fp32 precision, community-trained VAEs).
        # The loaded VAE state dict is automatically normalized to the standard
        # AutoencoderKLWan naming scheme if it uses an alternative convention.
        self.custom_vae_name_or_path = kwargs.get("custom_vae_name_or_path", None)
        
        # Wan transformer eps override (for LayerNorm and attention norms)
        # Official config uses 1e-6 (for fp32 training)
        # For bf16 training, use larger eps like 1e-4 or 1e-5 (bf16 has ~2-3 decimal digits precision)
        # Set to None to use the model's default eps from config
        # This is a model architecture setting, not a training hyperparameter
        self.wan_transformer_eps = kwargs.get('wan_transformer_eps', None)
        
        # Eps for Wan transformer blocks kept in fp32 (TREAD fp32_front / fp32_last_layers /
        # fp32_layers). Applied automatically per block compute dtype: fp32 blocks get this
        # value (default 1e-8, which fp32 resolves exactly), all other blocks get
        # wan_transformer_eps. The global wan_transformer_eps does NOT leak into fp32
        # blocks, so a bf16-oriented global (e.g. 1e-4) keeps fp32 blocks at 1e-8 unless
        # you set this explicitly. This is a model architecture setting, not a training
        # hyperparameter
        self.wan_transformer_fp32_eps = kwargs.get('wan_transformer_fp32_eps', None)
        
        # path to an accuracy recovery adapter, either local or remote
        self.accuracy_recovery_adapter = kwargs.get("accuracy_recovery_adapter", None)
        
        # parse ARA from qtype
        if self.qtype is not None and "|" in self.qtype:
            self.qtype, self.accuracy_recovery_adapter = self.qtype.split('|')

        # compile the model with torch compile
        self.compile = kwargs.get("compile", False)

        if self.compile and self.quantize:
            print("Quantized model detected - allowing torch.compile (experimental)")
        self.block_compile = kwargs.get("block_compile", False)
        self.compile_mode = kwargs.get("compile_mode", "default")
        self.compile_fullgraph = kwargs.get("compile_fullgraph", False)
        self.compile_dynamic = kwargs.get("compile_dynamic", True)
        self.cache_size_limit = kwargs.get("cache_size_limit", None)
        
        # Configurable noise schedulers for training and sampling
        self.train_scheduler = kwargs.get("train_scheduler", None)
        self.sampling_scheduler = kwargs.get("sampling_scheduler", None)
        
        # kwargs to pass to the model
        self.model_kwargs = kwargs.get("model_kwargs", {})
        
        # model paths for models that support it
        self.model_paths = kwargs.get("model_paths", {})
        
        self.in_context = kwargs.get("in_context", False)
        
        # allow frontend to pass arch with a color like arch:tag
        # but remove the tag
        if self.arch is not None:
            if ':' in self.arch:
                self.arch = self.arch.split(':')[0]
        
        if self.arch == "flex1":
            self.arch = "flux"
            
        
        # handle migrating to new model arch
        if self.arch is not None:
            # reverse the arch to the old style
            if self.arch == 'sd2':
                self.is_v2 = True
            elif self.arch == 'sd3':
                self.is_v3 = True
            elif self.arch == 'sdxl':
                self.is_xl = True
            elif self.arch == 'pixart':
                self.is_pixart = True
            elif self.arch == 'pixart_sigma':
                self.is_pixart_sigma = True
            elif self.arch == 'auraflow':
                self.is_auraflow = True
            elif self.arch == 'flux':
                self.is_flux = True
            elif self.arch == 'lumina2':
                self.is_lumina2 = True
            elif self.arch == 'vega':
                self.is_vega = True
            elif self.arch == 'ssd':
                self.is_ssd = True
            else:
                pass
        if self.arch is None:
            if kwargs.get('is_v2', False):
                self.arch = 'sd2'
            elif kwargs.get('is_v3', False):
                self.arch = 'sd3'
            elif kwargs.get('is_xl', False):
                self.arch = 'sdxl'
            elif kwargs.get('is_pixart', False):
                self.arch = 'pixart'
            elif kwargs.get('is_pixart_sigma', False):
                self.arch = 'pixart_sigma'
            elif kwargs.get('is_auraflow', False):
                self.arch = 'auraflow'
            elif kwargs.get('is_flux', False):
                self.arch = 'flux'
            elif kwargs.get('is_lumina2', False):
                self.arch = 'lumina2'
            elif kwargs.get('is_vega', False):
                self.arch = 'vega'
            elif kwargs.get('is_ssd', False):
                self.arch = 'ssd'
            else:
                self.arch = 'sd1'
        


class EMAConfig:
    def __init__(self, **kwargs):
        self.use_ema: bool = kwargs.get('use_ema', False)
        self.ema_decay: float = kwargs.get('ema_decay', 0.999)
        # feeds back the decay difference into the parameter
        self.use_feedback: bool = kwargs.get('use_feedback', False)
        
        # every update, the params are multiplied by this amount
        # only use for things without a bias like lora
        # similar to a decay in an optimizer but the opposite
        self.param_multiplier: float = kwargs.get('param_multiplier', 1.0)


class ReferenceDatasetConfig:
    def __init__(self, **kwargs):
        # can pass with a side by side pait or a folder with pos and neg folder
        self.pair_folder: str = kwargs.get('pair_folder', None)
        self.pos_folder: str = kwargs.get('pos_folder', None)
        self.neg_folder: str = kwargs.get('neg_folder', None)

        self.network_weight: float = float(kwargs.get('network_weight', 1.0))
        self.pos_weight: float = float(kwargs.get('pos_weight', self.network_weight))
        self.neg_weight: float = float(kwargs.get('neg_weight', self.network_weight))
        # make sure they are all absolute values no negatives
        self.pos_weight = abs(self.pos_weight)
        self.neg_weight = abs(self.neg_weight)

        self.target_class: str = kwargs.get('target_class', '')
        self.size: int = kwargs.get('size', 512)


class SliderTargetConfig:
    def __init__(self, **kwargs):
        self.target_class: str = kwargs.get('target_class', '')
        self.positive: str = kwargs.get('positive', '')
        self.negative: str = kwargs.get('negative', '')
        self.multiplier: float = kwargs.get('multiplier', 1.0)
        self.weight: float = kwargs.get('weight', 1.0)
        self.shuffle: bool = kwargs.get('shuffle', False)


class GuidanceConfig:
    def __init__(self, **kwargs):
        self.target_class: str = kwargs.get('target_class', '')
        self.guidance_scale: float = kwargs.get('guidance_scale', 1.0)
        self.positive_prompt: str = kwargs.get('positive_prompt', '')
        self.negative_prompt: str = kwargs.get('negative_prompt', '')


class SliderConfigAnchors:
    def __init__(self, **kwargs):
        self.prompt = kwargs.get('prompt', '')
        self.neg_prompt = kwargs.get('neg_prompt', '')
        self.multiplier = kwargs.get('multiplier', 1.0)


class SliderConfig:
    def __init__(self, **kwargs):
        targets = kwargs.get('targets', [])
        anchors = kwargs.get('anchors', [])
        anchors = [SliderConfigAnchors(**anchor) for anchor in anchors]
        self.anchors: List[SliderConfigAnchors] = anchors
        self.resolutions: List[List[int]] = kwargs.get('resolutions', [[512, 512]])
        self.prompt_file: str = kwargs.get('prompt_file', None)
        self.prompt_tensors: str = kwargs.get('prompt_tensors', None)
        self.batch_full_slide: bool = kwargs.get('batch_full_slide', True)
        self.use_adapter: bool = kwargs.get('use_adapter', None)  # depth
        self.adapter_img_dir = kwargs.get('adapter_img_dir', None)
        self.low_ram = kwargs.get('low_ram', False)

        # expand targets if shuffling
        from toolkit.prompt_utils import get_slider_target_permutations
        self.targets: List[SliderTargetConfig] = []
        targets = [SliderTargetConfig(**target) for target in targets]
        # do permutations if shuffle is true
        print(f"Building slider targets")
        for target in targets:
            if target.shuffle:
                target_permutations = get_slider_target_permutations(target, max_permutations=8)
                self.targets = self.targets + target_permutations
            else:
                self.targets.append(target)
        print(f"Built {len(self.targets)} slider targets (with permutations)")

ControlTypes = Literal['depth', 'line', 'pose', 'inpaint', 'mask', 'sapiens2_mask']

class DatasetConfig:
    """
    Dataset config for sd-datasets

    """

    def __init__(self, **kwargs):
        self.type = kwargs.get('type', 'image')  # sd, slider, reference
        # will be legacy
        self.folder_path: str = kwargs.get('folder_path', None)
        # can be json or folder path
        self.dataset_path: str = kwargs.get('dataset_path', None)

        self.default_caption: str = kwargs.get('default_caption', None)
        # trigger word for just this dataset
        self.trigger_word: str = kwargs.get('trigger_word', None)
        random_triggers = kwargs.get('random_triggers', [])
        # if they are a string, load them from a file
        if isinstance(random_triggers, str) and os.path.exists(random_triggers):
            with open(random_triggers, 'r') as f:
                random_triggers = f.read().splitlines()
                # remove empty lines
                random_triggers = [line for line in random_triggers if line.strip() != '']
        self.random_triggers: List[str] = random_triggers
        self.random_triggers_max: int = kwargs.get('random_triggers_max', 1)
        self.caption_ext: str = kwargs.get('caption_ext', '.txt')
        # if caption_ext doesnt start with a dot, add it
        if self.caption_ext and not self.caption_ext.startswith('.'):
            self.caption_ext = '.' + self.caption_ext
        self.random_scale: bool = kwargs.get('random_scale', False)
        self.random_crop: bool = kwargs.get('random_crop', False)
        self.resolution: int = kwargs.get('resolution', 512)
        # PIL resample filter used for all image resize/scale ops on this dataset.
        # One of: 'bicubic', 'lanczos'
        self.resize_method: str = kwargs.get('resize_method', 'lanczos')
        self.scale: float = kwargs.get('scale', 1.0)
        self.buckets: bool = kwargs.get('buckets', True)
        self.bucket_tolerance: int = kwargs.get('bucket_tolerance', 64)
        self.is_reg: bool = kwargs.get('is_reg', False)
        self.prior_reg: bool = kwargs.get('prior_reg', False)
        self.network_weight: float = float(kwargs.get('network_weight', 1.0))
        self.token_dropout_rate: float = float(kwargs.get('token_dropout_rate', 0.0))
        self.shuffle_tokens: bool = kwargs.get('shuffle_tokens', False)
        self.caption_dropout_rate: float = float(kwargs.get('caption_dropout_rate', 0.0))
        self.caption_dropout_rate_t2v: float = float(kwargs.get('caption_dropout_rate_t2v', 0.0))
        # Per-dataset conditioning dropout rates. When set (not None) these OVERRIDE the
        # corresponding global TrainConfig rate for items in this dataset. When None the
        # global rate is used (falling back to caption_dropout_rate for text).
        #   text_dropout_rate(_negative)   -> prompt (-> blank) drop rate
        #   image_dropout_rate(_negative)  -> I2V first-frame image drop rate (-> T2V)
        self.text_dropout_rate: Optional[float] = kwargs.get('text_dropout_rate', None)
        self.text_dropout_rate_negative: Optional[float] = kwargs.get('text_dropout_rate_negative', None)
        self.image_dropout_rate: Optional[float] = kwargs.get('image_dropout_rate', None)
        self.image_dropout_rate_negative: Optional[float] = kwargs.get('image_dropout_rate_negative', None)
        self.keep_tokens: int = kwargs.get('keep_tokens', 0)  # #of first tokens to always keep unless caption dropped
        self.flip_x: bool = kwargs.get('flip_x', False)
        self.flip_y: bool = kwargs.get('flip_y', False)
        self.augments: List[str] = kwargs.get('augments', [])
        self.control_path: Union[str,List[str]] = kwargs.get('control_path', None)  # depth maps, etc
        # pull a random control image from the same folder as the image. Useful for folder grouped pairs.
        self.control_from_same_folder: bool = kwargs.get('control_from_same_folder', False)
        self.num_controls_from_same_folder: int = kwargs.get('num_controls_from_same_folder', 1)
        
        if self.control_path == '':
            self.control_path = None
        
        # handle multi control inputs from the ui. It is just easier to handle it here for a cleaner ui experience
        control_path_1 = kwargs.get('control_path_1', None)
        control_path_2 = kwargs.get('control_path_2', None)
        control_path_3 = kwargs.get('control_path_3', None)
        
        if any([control_path_1, control_path_2, control_path_3]):
            control_paths = []
            if control_path_1:
                control_paths.append(control_path_1)
            if control_path_2:
                control_paths.append(control_path_2)
            if control_path_3:
                control_paths.append(control_path_3)
            self.control_path = control_paths
        
        # color for transparent reigon of control images with transparency
        self.control_transparent_color: List[int] = kwargs.get('control_transparent_color', [0, 0, 0])
        # inpaint images should be webp/png images with alpha channel. The alpha 0 (invisible) section will
        # be the part conditioned to be inpainted. The alpha 1 (visible) section will be the part that is ignored
        self.inpaint_path: Union[str,List[str]] = kwargs.get('inpaint_path', None)
        # instead of cropping ot match image, it will serve the full size control image (clip images ie for ip adapters)
        self.full_size_control_images: bool = kwargs.get('full_size_control_images', True)
        self.alpha_mask: bool = kwargs.get('alpha_mask', False)  # if true, will use alpha channel as mask
        self.mask_path: str = kwargs.get('mask_path',
                                         None)  # focus mask (black and white. White has higher loss than black)
        self.unconditional_path: str = kwargs.get('unconditional_path',
                                                  None)  # path where matching unconditional images are located
        self.invert_mask: bool = kwargs.get('invert_mask', False)  # invert mask
        self.mask_min_value: float = kwargs.get('mask_min_value', 0.0)  # min value for . 0 - 1
        self.poi: Union[str, None] = kwargs.get('poi', None)
        if self.poi is not None:
            raise ValueError("poi is deprecated and is no longer supported")
        self.use_short_captions: bool = kwargs.get('use_short_captions', False)  # if true, will use 'caption_short' from json
        self.num_repeats: int = kwargs.get('num_repeats', 1)  # number of times to repeat dataset
        # cache latents will store them in memory
        self.cache_latents: bool = kwargs.get('cache_latents', False)
        # cache latents to disk will store them on disk. If both are true, it will save to disk, but keep in memory
        self.cache_latents_to_disk: bool = kwargs.get('cache_latents_to_disk', False)
        self.cache_clip_vision_to_disk: bool = kwargs.get('cache_clip_vision_to_disk', False)
        self.cache_text_embeddings: bool = kwargs.get('cache_text_embeddings', False)
        self.load_image_when_caching_latents: bool = kwargs.get('load_image_when_caching_latents', False)

        self.standardize_images: bool = kwargs.get('standardize_images', False)

        # https://albumentations.ai/docs/api_reference/augmentations/transforms
        # augmentations are returned as a separate image and cannot currently be cached
        self.augmentations: List[dict] = kwargs.get('augmentations', None)
        self.shuffle_augmentations: bool = kwargs.get('shuffle_augmentations', False)

        has_augmentations = self.augmentations is not None and len(self.augmentations) > 0

        if (len(self.augments) > 0 or has_augmentations) and (self.cache_latents or self.cache_latents_to_disk):
            print(f"WARNING: Augments are not supported with caching latents. Setting cache_latents to False")
            self.cache_latents = False
            self.cache_latents_to_disk = False

        # legacy compatability
        legacy_caption_type = kwargs.get('caption_type', None)
        if legacy_caption_type:
            self.caption_ext = legacy_caption_type
        self.caption_type = self.caption_ext
        self.guidance_type: GuidanceType = kwargs.get('guidance_type', 'targeted')

        # ip adapter / reference dataset
        self.clip_image_path: str = kwargs.get('clip_image_path', None)  # depth maps, etc
        # get the clip image randomly from the same folder as the image. Useful for folder grouped pairs.
        self.clip_image_from_same_folder: bool = kwargs.get('clip_image_from_same_folder', False)
        self.clip_image_augmentations: List[dict] = kwargs.get('clip_image_augmentations', None)
        self.clip_image_shuffle_augmentations: bool = kwargs.get('clip_image_shuffle_augmentations', False)
        self.replacements: List[str] = kwargs.get('replacements', [])
        self.loss_multiplier: float = kwargs.get('loss_multiplier', 1.0)

        self.num_workers: int = kwargs.get('num_workers', 2)
        self.prefetch_factor: int = kwargs.get('prefetch_factor', 2)
        # Number of optimizer steps the training dataloader keeps staged in
        # VRAM ahead of the current step (rotating prefetch buffer). The
        # buffer holds `prefetch_steps * batch_size * gradient_accumulation`
        # items. Set 0 to disable the prefetch stream.
        self.prefetch_steps: int = kwargs.get('prefetch_steps', 2)
        self.extra_values: List[float] = kwargs.get('extra_values', [])
        self.square_crop: bool = kwargs.get('square_crop', False)
        # apply same augmentations to control images. Usually want this true unless special case
        self.replay_transforms: bool = kwargs.get('replay_transforms', True)
        
        # for video
        # if num_frames is greater than 1, the dataloader will look for video files.
        # num_frames will be the number of frames in the training batch. If num_frames is 1, it will look for images
        self.num_frames: int = kwargs.get('num_frames', 1)
        # if true, will shrink video to our frames. For instance, if we have a video with 100 frames and num_frames is 10,
        # we would pull frame 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 so they are evenly spaced
        self.shrink_video_to_frames: bool = kwargs.get('shrink_video_to_frames', True)
        # fps is only used if shrink_video_to_frames is false. This will attempt to pull the num_frames at the given fps
        # it will select a random start frame and pull the frames at the given fps
        # this could have various issues with shorter videos and videos with variable fps
        # I recommend trimming your videos to the desired length and using shrink_video_to_frames(default)
        self.fps: int = kwargs.get('fps', 24)
        
        # auto_frame_count pull as many frames as in the video at given fps
        # Important, make sure fps for dataset is set correctly.
        # this wont work with bucketing for now until I can handle this before bucketing.
        self.auto_frame_count: bool = kwargs.get('auto_frame_count', False)
        
        # debug the frame count and frame selection. You dont need this. It is for debugging.
        self.debug: bool = kwargs.get('debug', False)
        
        # automatic controls
        self.controls: List[ControlTypes] = kwargs.get('controls', [])
        if isinstance(self.controls, str):
            self.controls = [self.controls]
        # remove empty strings
        self.controls = [control for control in self.controls if control.strip() != '']
        
        # if true, will use a fask method to get image sizes. This can result in errors. Do not use unless you know what you are doing
        self.fast_image_size: bool = kwargs.get('fast_image_size', False)
        
        self.do_i2v: bool = kwargs.get('do_i2v', True)  # do image to video on models that are both t2i and i2v capable
        self.do_t2v: bool = kwargs.get('do_t2v', False)  # do text to video (without first frame conditioning)
        self.do_audio: bool = kwargs.get('do_audio', False) # load audio from video files for models that support it
        self.audio_preserve_pitch: bool = kwargs.get('audio_preserve_pitch', False) # preserve pitch when stretching audio to fit num_frames
        self.audio_normalize: bool = kwargs.get('audio_normalize', False) # normalize audio volume levels when loading

        # Optical flow caching for video datasets (used by spectral_flow loss type)
        # Mirrors cache_latents_to_disk: stores precomputed flow in _flow_cache/ beside videos
        self.cache_optical_flow_to_disk: bool = kwargs.get('cache_optical_flow_to_disk', False)
        # Which pretrained flow model to use. Options: "sea-raft-m" (default), "sea-raft-s"
        self.optical_flow_model: str = kwargs.get('optical_flow_model', 'sea-raft-m')
        # Resolution at which flow is computed. "dataset" = bucketed training resolution (default)
        self.optical_flow_resolution: str = kwargs.get('optical_flow_resolution', 'dataset')

        # Validation: flow caching requires video data
        if self.cache_optical_flow_to_disk and self.num_frames <= 1 and not self.auto_frame_count:
            print(f"WARNING: cache_optical_flow_to_disk requires num_frames > 1 or auto_frame_count. Disabling.")
            self.cache_optical_flow_to_disk = False


def preprocess_dataset_raw_config(raw_config: List[dict]) -> List[dict]:
    """
    This just splits up the datasets by resolutions so you dont have to do it manually
    :param raw_config:
    :return:
    """
    # split up datasets by resolutions
    new_config = []
    for dataset in raw_config:
        resolution = dataset.get('resolution', 512)
        if isinstance(resolution, list):
            resolution_list = resolution
        else:
            resolution_list = [resolution]
        for res in resolution_list:
            dataset_copy = dataset.copy()
            dataset_copy['resolution'] = res
            new_config.append(dataset_copy)
    return new_config


class GenerateImageConfig:
    def __init__(
            self,
            prompt: str = '',
            prompt_2: Optional[str] = None,
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: str = '',
            negative_prompt_2: Optional[str] = None,
            seed: int = -1,
            network_multiplier: float = 1.0,
            guidance_rescale: float = 0.0,
            # the tag [time] will be replaced with milliseconds since epoch
            output_path: str = None,  # full image path
            output_folder: str = None,  # folder to save image in if output_path is not specified
            output_ext: str = ImgExt,  # extension to save image as if output_path is not specified
            output_tail: str = '',  # tail to add to output filename
            add_prompt_file: bool = False,  # add a prompt file with generated image
            adapter_image_path: str = None,  # path to adapter image
            adapter_conditioning_scale: float = 1.0,  # scale for adapter conditioning
            latents: Union[torch.Tensor | None] = None,  # input latent to start with,
            extra_kwargs: dict = None,  # extra data to save with prompt file
            refiner_start_at: float = 0.5,  # start at this percentage of a step. 0.0 to 1.0 . 1.0 is the end
            extra_values: List[float] = None,  # extra values to save with prompt file
            logger: Optional[EmptyLogger] = None,
            ctrl_img: Optional[str] = None,  # control image for controlnet
            ctrl_img_1: Optional[str] = None,  # first control image for multi control model
            ctrl_img_2: Optional[str] = None,  # second control image for multi control model
            ctrl_img_3: Optional[str] = None,  # third control image for multi control model
            num_frames: int = 1,
            fps: int = 15,
            ctrl_idx: int = 0,
            do_cfg_norm: bool = False,
            nag_scale: float = 1.0,
            nag_alpha: float = 0.5,
            nag_tau: float = 3.5,
            attention_tanh_softcap_enabled: bool = False,
            attention_tanh_softcap_value: Optional[float] = None,
    ):
        self.width: int = width
        self.height: int = height
        self.num_inference_steps: int = num_inference_steps
        self.guidance_scale: float = guidance_scale
        self.guidance_rescale: float = guidance_rescale
        self.prompt: str = prompt
        self.prompt_2: str = prompt_2
        self.negative_prompt: str = negative_prompt
        self.negative_prompt_2: str = negative_prompt_2
        self.latents: Union[torch.Tensor | None] = latents

        self.output_path: str = output_path
        self.seed: int = seed
        if self.seed == -1:
            # generate random one
            self.seed = random.randint(0, 2 ** 32 - 1)
        self.network_multiplier: float = network_multiplier
        self.output_folder: str = output_folder
        self.output_ext: str = output_ext
        self.add_prompt_file: bool = add_prompt_file
        self.output_tail: str = output_tail
        self.gen_time: int = int(time.time() * 1000)
        self.adapter_image_path: str = adapter_image_path
        self.adapter_conditioning_scale: float = adapter_conditioning_scale
        self.extra_kwargs = extra_kwargs if extra_kwargs is not None else {}
        self.refiner_start_at = refiner_start_at
        self.extra_values = extra_values if extra_values is not None else []
        self.num_frames = num_frames
        self.fps = fps
        self.ctrl_img = ctrl_img
        self.ctrl_idx = ctrl_idx
        
        if ctrl_img_1 is None and ctrl_img is not None:
            ctrl_img_1 = ctrl_img
        
        self.ctrl_img_1 = ctrl_img_1
        self.ctrl_img_2 = ctrl_img_2
        self.ctrl_img_3 = ctrl_img_3

        # prompt string will override any settings above
        self._process_prompt_string()

        # handle dual text encoder prompts if nothing passed
        if negative_prompt_2 is None:
            self.negative_prompt_2 = negative_prompt

        if prompt_2 is None:
            self.prompt_2 = self.prompt

        # parse prompt paths
        if self.output_path is None and self.output_folder is None:
            raise ValueError('output_path or output_folder must be specified')
        elif self.output_path is not None:
            self.output_folder = os.path.dirname(self.output_path)
            self.output_ext = os.path.splitext(self.output_path)[1][1:]
            self.output_filename_no_ext = os.path.splitext(os.path.basename(self.output_path))[0]

        else:
            self.output_filename_no_ext = '[time]_[count]'
            if len(self.output_tail) > 0:
                self.output_filename_no_ext += '_' + self.output_tail
            self.output_path = os.path.join(self.output_folder, self.output_filename_no_ext + '.' + self.output_ext)

        # adjust height
        self.height = max(64, self.height - self.height % 8)  # round to divisible by 8
        self.width = max(64, self.width - self.width % 8)  # round to divisible by 8

        self.logger = logger
        
        self.do_cfg_norm: bool = do_cfg_norm

        # NAG (Negative Attention Guidance) parameters
        self.nag_scale: float = nag_scale
        self.nag_alpha: float = nag_alpha
        self.nag_tau: float = nag_tau

        # Attention tanh softcap (Wan 2.x). The enabled flag is the per-sample
        # EFFECTIVE toggle (per-sample override -> sample-level toggle, already
        # resolved by the caller). None value = inherit (sample-level value,
        # then the training value).
        self.attention_tanh_softcap_enabled: bool = attention_tanh_softcap_enabled
        self.attention_tanh_softcap_value: Optional[float] = attention_tanh_softcap_value

    def set_gen_time(self, gen_time: int = None):
        if gen_time is not None:
            self.gen_time = gen_time
        else:
            self.gen_time = int(time.time() * 1000)

    def _get_path_no_ext(self, count: int = 0, max_count=0):
        # zero pad count
        count_str = str(count).zfill(len(str(max_count)))
        # replace [time] with gen time
        filename = self.output_filename_no_ext.replace('[time]', str(self.gen_time))
        # replace [count] with count
        filename = filename.replace('[count]', count_str)
        return filename

    def get_image_path(self, count: int = 0, max_count=0):
        filename = self._get_path_no_ext(count, max_count)
        ext = self.output_ext
        # if it does not start with a dot add one
        if ext[0] != '.':
            ext = '.' + ext
        filename += ext
        # join with folder
        return os.path.join(self.output_folder, filename)

    def get_prompt_path(self, count: int = 0, max_count=0):
        filename = self._get_path_no_ext(count, max_count)
        filename += '.txt'
        # join with folder
        return os.path.join(self.output_folder, filename)

    def save_image(self, image, count: int = 0, max_count=0):
        # make parent dirs
        os.makedirs(self.output_folder, exist_ok=True)
        self.set_gen_time()
        if isinstance(image, list):
            # video
            if self.num_frames == 1:
                raise ValueError(f"Expected 1 img but got a list {len(image)}")
            if self.num_frames > 1 and self.output_ext not in ['webp', 'mp4']:
                self.output_ext = 'mp4'
            if self.output_ext == 'mp4':
                # save as mp4 with libx264 crf 24
                self._save_video_mp4(image, count, max_count)
            elif self.output_ext == 'webp':
                # save as animated webp
                duration = 1000 // self.fps  # Convert fps to milliseconds per frame
                image[0].save(
                    self.get_image_path(count, max_count),
                    format='WEBP',
                    append_images=image[1:],
                    save_all=True,
                    duration=duration,  # Duration per frame in milliseconds
                    loop=0,  # 0 means loop forever
                    quality=80  # Quality setting (0-100)
                )
            else:
                raise ValueError(f"Unsupported video format {self.output_ext}")
        elif self.output_ext in ['wav', 'mp3', 'flac', 'ogg']:
            # save audio file
            audio_path = self.get_image_path(count, max_count)
            torchaudio.save(
                audio_path, 
                image[0].to('cpu'),
                sample_rate=48000, 
                format=None, 
                backend=None
            )
            if self.output_ext == 'mp3':
                add_album_artwork(audio_path)
        else:
            # TODO save image gen header info for A1111 and us, our seeds probably wont match
            image.save(self.get_image_path(count, max_count))
            # do prompt file
            if self.add_prompt_file:
                self.save_prompt_file(count, max_count)

    def encrypt_sample_if_enabled(self, count: int = 0, max_count=0):
        """Encrypt the just-written sample file in place if sample encryption is enabled.

        Called from the ``generate_images`` funnels right after ``save_image`` so
        that every generated sample (image / video / audio, including any
        model-specific ``save_image`` override) is encrypted. The sample keeps
        its extension (e.g. .png) but its bytes become an AITK encrypted sample
        blob (X25519 public key + AES-256-GCM). The public key was derived
        client-side from the user's password - the password itself never reaches
        the server. No-op when no sample public key is configured. Failures are
        logged, not raised, so a crypto problem never aborts sampling.
        """
        try:
            from toolkit import dataset_crypto
        except Exception:
            return
        try:
            if dataset_crypto.is_sample_encryption_enabled():
                dataset_crypto.encrypt_sample_file_in_place(self.get_image_path(count, max_count))
        except Exception as e:
            print(f"Error encrypting generated sample: {e}")

    def _save_video_mp4(self, frames, count: int = 0, max_count=0):
        """Save frames as MP4 video using libx264 with CRF 24."""
        try:
            import av
        except ImportError:
            raise ImportError(
                "PyAV is required for MP4 video output. Install with: pip install av"
            )
        import numpy as np
        from PIL import Image

        output_path = self.get_image_path(count, max_count)
        fps = max(1, self.fps)  # Ensure fps is at least 1

        # Get dimensions from first frame
        first_frame = frames[0]
        width, height = first_frame.size

        # Ensure dimensions are divisible by 2 (required by x264)
        width = width - (width % 2)
        height = height - (height % 2)

        # Resize frames if needed
        if width != first_frame.size[0] or height != first_frame.size[1]:
            frames = [f.resize((width, height), Image.Resampling.LANCZOS) for f in frames]

        # Open container for writing
        container = av.open(output_path, mode='w')

        # Add video stream with libx264
        stream = container.add_stream('libx264', rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'  # Required for compatibility
        stream.options = {'crf': '24'}  # Constant Rate Factor for quality

        # Encode each frame
        for frame_pil in frames:
            frame_np = np.array(frame_pil.convert('RGB'))
            frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)

        container.close()

    def save_prompt_file(self, count: int = 0, max_count=0):
        # save prompt file
        with open(self.get_prompt_path(count, max_count), 'w') as f:
            prompt = self.prompt
            if self.prompt_2 is not None:
                prompt += ' --p2 ' + self.prompt_2
            if self.negative_prompt is not None:
                prompt += ' --n ' + self.negative_prompt
            if self.negative_prompt_2 is not None:
                prompt += ' --n2 ' + self.negative_prompt_2
            prompt += ' --w ' + str(self.width)
            prompt += ' --h ' + str(self.height)
            prompt += ' --seed ' + str(self.seed)
            prompt += ' --cfg ' + str(self.guidance_scale)
            prompt += ' --steps ' + str(self.num_inference_steps)
            prompt += ' --m ' + str(self.network_multiplier)
            prompt += ' --gr ' + str(self.guidance_rescale)

            # get gen info
            try:
                f.write(self.prompt)
            except Exception as e:
                print(f"Error writing prompt file. Prompt contains non-unicode characters. {e}")

    def _process_prompt_string(self):
        # we will try to support all sd-scripts where we can

        # FROM SD-SCRIPTS
        # --n Treat everything until the next option as a negative prompt.
        # --w Specify the width of the generated image.
        # --h Specify the height of the generated image.
        # --d Specify the seed for the generated image.
        # --l Specify the CFG scale for the generated image.
        # --s Specify the number of steps during generation.

        # OURS and some QOL additions
        # --m Specify the network multiplier for the generated image.
        # --p2 Prompt for the second text encoder (SDXL only)
        # --n2 Negative prompt for the second text encoder (SDXL only)
        # --gr Specify the guidance rescale for the generated image (SDXL only)

        # --seed Specify the seed for the generated image same as --d
        # --cfg Specify the CFG scale for the generated image same as --l
        # --steps Specify the number of steps during generation same as --s
        # --network_multiplier Specify the network multiplier for the generated image same as --m

        # process prompt string and update values if it has some
        if self.prompt is not None and len(self.prompt) > 0:
            # process prompt string
            prompt = self.prompt
            prompt = prompt.strip()
            p_split = prompt.split('--')
            self.prompt = p_split[0].strip()

            if len(p_split) > 1:
                for split in p_split[1:]:
                    # allows multi char flags
                    flag = split.split(' ')[0].strip()
                    content = split[len(flag):].strip()
                    if flag == 'p2':
                        self.prompt_2 = content
                    elif flag == 'n':
                        self.negative_prompt = content
                    elif flag == 'n2':
                        self.negative_prompt_2 = content
                    elif flag == 'w':
                        self.width = int(content)
                    elif flag == 'h':
                        self.height = int(content)
                    elif flag == 'd':
                        self.seed = int(content)
                    elif flag == 'seed':
                        self.seed = int(content)
                    elif flag == 'l':
                        self.guidance_scale = float(content)
                    elif flag == 'cfg':
                        self.guidance_scale = float(content)
                    elif flag == 's':
                        self.num_inference_steps = int(content)
                    elif flag == 'steps':
                        self.num_inference_steps = int(content)
                    elif flag == 'm':
                        self.network_multiplier = float(content)
                    elif flag == 'network_multiplier':
                        self.network_multiplier = float(content)
                    elif flag == 'gr':
                        self.guidance_rescale = float(content)
                    elif flag == 'a':
                        self.adapter_conditioning_scale = float(content)
                    elif flag == 'ref':
                        self.refiner_start_at = float(content)
                    elif flag == 'ev':
                        # split by comma
                        self.extra_values = [float(val) for val in content.split(',')]
                    elif flag == 'extra_values':
                        # split by comma
                        self.extra_values = [float(val) for val in content.split(',')]
                    elif flag == 'frames':
                        self.num_frames = int(content)
                    elif flag == 'num_frames':
                        self.num_frames = int(content)
                    elif flag == 'fps':
                        self.fps = int(content)
                    elif flag == 'ctrl_img':
                        self.ctrl_img = content
                    elif flag == 'ctrl_idx':
                        self.ctrl_idx = int(content)

    def post_process_embeddings(
            self,
            conditional_prompt_embeds: PromptEmbeds,
            unconditional_prompt_embeds: Optional[PromptEmbeds] = None,
    ):
        # this is called after prompt embeds are encoded. We can override them in the future here
        pass
    
    def log_image(self, image, count: int = 0, max_count=0):
        if self.logger is None:
            return

        self.logger.log_image(image, count, self.prompt)
        
        
def validate_configs(
    train_config: TrainConfig,
    model_config: ModelConfig,
    save_config: SaveConfig,
    dataset_configs: List[DatasetConfig]
):
    if model_config.is_flux:
        if save_config.save_format != 'diffusers':
            # make it diffusers
            save_config.save_format = 'diffusers'
        if model_config.use_flux_cfg:
            # bypass the embedding
            train_config.bypass_guidance_embedding = True
    if train_config.bypass_guidance_embedding and train_config.do_guidance_loss:
        raise ValueError("Cannot bypass guidance embedding and do guidance loss at the same time. "
                         "Please set bypass_guidance_embedding to False or do_guidance_loss to False.")
        
    if model_config.accuracy_recovery_adapter is not None:
        if model_config.assistant_lora_path is not None:
            raise ValueError("Cannot use accuracy recovery adapter and assistant lora at the same time. "
                             "Please set one of them to None.")

    # see if any datasets are caching text embeddings
    is_caching_text_embeddings = any(dataset.cache_text_embeddings for dataset in dataset_configs)
    if is_caching_text_embeddings:
        
        # check if they are doing differential output preservation
        if train_config.diff_output_preservation:
            raise ValueError("Cannot use differential output preservation with caching text embeddings. Please set diff_output_preservation to False.")
    
        # make sure they are all cached
        for dataset in dataset_configs:
            if not dataset.cache_text_embeddings:
                raise ValueError("All datasets must have cache_text_embeddings set to True when caching text embeddings is enabled.")
    
    # qwen image edit cannot cache text embeddings
    if model_config.arch in ['qwen_image_edit', 'boogu_image_edit']:
        if train_config.unload_text_encoder:
            raise ValueError(f"Cannot cache unload text encoder with {model_config.arch} model. Control images are encoded with text embeddings. You can cache the text embeddings though")
    
    if train_config.diff_output_preservation and train_config.blank_prompt_preservation:
        raise ValueError("Cannot use both differential output preservation and blank prompt preservation at the same time. Please set one of them to False.")
    
    if train_config.batch_size > 1 and any(dataset_config.auto_frame_count for dataset_config in dataset_configs):
        raise ValueError("Cannot use batch size greater than 1 with auto_frame_count. Please set batch_size to 1 or auto_frame_count to False.")

    
