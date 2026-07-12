from functools import partial
import os
import json
import tempfile
from typing import Any, Dict, Optional, Union, List
from typing_extensions import Self
import torch
import yaml
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
from diffusers import UniPCMultistepScheduler
import torch
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import quantize_model
from .wan22_pipeline import Wan22Pipeline
from diffusers import WanTransformer3DModel
from huggingface_hub import hf_hub_download, list_repo_files

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from torchvision.transforms import functional as TF

from toolkit.models.wan21.wan21 import Wan21
from .wan22_5b_model import (
    scheduler_config,
    time_text_monkeypatch,
)
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file, save_file


boundary_ratio_t2v = 0.875
boundary_ratio_i2v = 0.9

scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}


def _process_state_dict_for_fp8(state_dict: Dict[str, torch.Tensor], target_dtype: torch.dtype = torch.bfloat16, debug: bool = True) -> Dict[str, torch.Tensor]:
	"""
	Process state dict to dequantize FP8 weights and map keys to diffusers format.
	Handles:
	- FP8 quantized weights with scale factors (scale_input, scale_weight)
	- Key prefix removal (model., diffusion_model., transformer.)
	- Key remapping for different naming conventions between FP8 and diffusers
	The FP8 format uses different key names than diffusers:
	- text_embedding -> condition_embedder.text_embedder.linear_*
	- head.head -> proj_out
	- scale_shift_table might be missing (needs to be initialized by diffusers)
	- self_attn/cross_attn -> attn1/attn2 + to_q/to_k/to_v/to_out.0
	- ffn.0/ffn.2 -> ffn.net.0.proj / ffn.net.2
	- norm3 -> norm2
	- modulation -> scale_shift_table (per-block + head)
	- 'scaled_fp8' is a ComfyUI-only flag (will be skipped)
	For non-FP8 (e.g., bf16) models, this function will still process key prefixes
	and remapping but skip the dequantization step.
	"""
	processed_state_dict = {}
	all_keys = list(state_dict.keys())
# Build set of all scale keys from the state dict
	scale_keys = set()
	for key in all_keys:
		if 'scale_input' in key or 'scale_weight' in key or '._scale' in key:
			scale_keys.add(key)
# Check if this is an FP8 quantized model
# FP8 models have keys with scale_input, scale_weight, or ._data suffixes
	is_fp8_model = any(
		'scale_input' in k or 'scale_weight' in k or k.endswith('._data')
		for k in all_keys
	)
	if not is_fp8_model:
# This is not an FP8 model (e.g., bf16) - just return
		print("This is not an FP8 model (e.g., bf16), but keys will be remapped")
		# return processed_state_dict
# FP8 model - proceed with full processing
# Debug: Print some key patterns to understand the structure
	print(f"DEBUG: FP8 model detected. Total keys in state_dict: {len(all_keys)}")
# Print ALL unique key patterns to understand the structure
	print("=== DEBUG: ALL KEY PATTERNS ===")
# Group keys by their prefix after removing common prefixes
	key_groups = {}
	for key in all_keys:
# Try removing each prefix
		new_key = key
		for prefix in ["model.", "diffusion_model.", "transformer."]:
			if new_key.startswith(prefix):
				new_key = new_key[len(prefix):]
				break
# Get the first part of the key
		parts = new_key.split('.')
		if len(parts) >= 2:
			group = parts[0] + "." + parts[1] if len(parts) > 1 else parts[0]
		else:
			group = parts[0] if parts else "unknown"
		if group not in key_groups:
			key_groups[group] = []
		key_groups[group].append(new_key)
	for group, keys in sorted(key_groups.items()):
		print(f"Group '{group}': {len(keys)} keys")
# Print first 3 keys from each group
		for k in keys[:3]:
			print(f" - {k}")
# Print keys that contain specific patterns
	print("\n=== DEBUG: KEYS WITH 'scale_input' ===")
	scale_input_keys = [k for k in all_keys if 'scale_input' in k]
	print(f"Found {len(scale_input_keys)} scale_input keys")
	for k in scale_input_keys[:5]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'scale_weight' ===")
	scale_weight_keys = [k for k in all_keys if 'scale_weight' in k]
	print(f"Found {len(scale_weight_keys)} scale_weight keys")
	for k in scale_weight_keys[:5]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'blocks' ===")
	blocks_keys = [k for k in all_keys if 'blocks' in k]
	print(f"Found {len(blocks_keys)} blocks keys")
# Group by block number
	block_nums = set()
	for k in blocks_keys:
		import re
		match = re.search(r'blocks\.(\d+)', k)
		if match:
			block_nums.add(int(match.group(1)))
	print(f"Block numbers: {sorted(block_nums)}")
# Print one example from block 0
	for k in blocks_keys[:5]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'weight' (no scale) ===")
	weight_keys = [k for k in all_keys if 'weight' in k.lower() and 'scale' not in k.lower()]
	print(f"Found {len(weight_keys)} weight keys")
	for k in weight_keys[:10]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'bias' ===")
	bias_keys = [k for k in all_keys if 'bias' in k.lower()]
	print(f"Found {len(bias_keys)} bias keys")
	for k in bias_keys[:10]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'text_embedding' ===")
	text_emb_keys = [k for k in all_keys if 'text_embedding' in k]
	for k in text_emb_keys[:10]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'head.head' ===")
	head_keys = [k for k in all_keys if 'head.head' in k or 'head.' in k]
	for k in head_keys[:10]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'patch_embedding' ===")
	patch_keys = [k for k in all_keys if 'patch_embedding' in k]
	for k in patch_keys[:10]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'modulation' ===")
	mod_keys = [k for k in all_keys if 'modulation' in k]
	for k in mod_keys[:10]:
		print(f" - {k}")
	print("\n=== DEBUG: SAMPLE KEYS (first 50) ===")
	for k in all_keys[:50]:
		print(f" - {k}")
# Additional debug prints for mapping-critical keys (as requested)
	print("\n=== DEBUG: KEYS WITH 'self_attn' ===")
	self_attn_keys = [k for k in all_keys if 'self_attn' in k]
	print(f"Found {len(self_attn_keys)} self_attn keys")
	for k in self_attn_keys[:8]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'cross_attn' ===")
	cross_attn_keys = [k for k in all_keys if 'cross_attn' in k]
	print(f"Found {len(cross_attn_keys)} cross_attn keys")
	for k in cross_attn_keys[:8]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'ffn' ===")
	ffn_keys = [k for k in all_keys if 'ffn' in k]
	print(f"Found {len(ffn_keys)} ffn keys")
	for k in ffn_keys[:8]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'norm3' ===")
	norm3_keys = [k for k in all_keys if 'norm3' in k]
	print(f"Found {len(norm3_keys)} norm3 keys")
	for k in norm3_keys[:5]:
		print(f" - {k}")
	print("\n=== DEBUG: KEYS WITH 'norm' (non-scale) ===")
	norm_keys = [k for k in all_keys if 'norm' in k.lower() and 'scale' not in k.lower()]
	print(f"Found {len(norm_keys)} norm keys")
	for k in sorted(norm_keys)[:15]:
		print(f" - {k}")
	print("=== DEBUG: Starting key remapping and dequantization ===")
# Define remapping function
	def remap_key(key: str) -> str:
# Strip prefixes (already done but safety)
		for prefix in ["model.", "diffusion_model.", "transformer."]:
			if key.startswith(prefix):
				key = key[len(prefix):]
				break
# Text embedding
		if key.startswith('text_embedding.0.'):
			return key.replace('text_embedding.0.', 'condition_embedder.text_embedder.linear_1.')
		if key.startswith('text_embedding.2.'):
			return key.replace('text_embedding.2.', 'condition_embedder.text_embedder.linear_2.')
# Time embedding
		if key.startswith('time_embedding.0.'):
			return key.replace('time_embedding.0.', 'condition_embedder.time_embedder.linear_1.')
		if key.startswith('time_embedding.2.'):
			return key.replace('time_embedding.2.', 'condition_embedder.time_embedder.linear_2.')
		if key.startswith('time_projection.1.'):
			return key.replace('time_projection.1.', 'condition_embedder.time_proj.')
# Head
		if key.startswith('head.head.'):
			return key.replace('head.head.', 'proj_out.')
		if key == 'head.modulation':
			return 'scale_shift_table'
# Blocks
		if 'blocks.' in key:
# attn remap + q/k/v/o
			key = key.replace('.self_attn.', '.attn1.')
			key = key.replace('.cross_attn.', '.attn2.')
			key = re.sub(r'\.(q|k|v)\.', r'.to_\1.', key)
			key = key.replace('.o.', '.to_out.0.')
# ffn
			key = key.replace('.ffn.0.', '.ffn.net.0.proj.')
			key = key.replace('.ffn.2.', '.ffn.net.2.')
# norm3
			key = key.replace('.norm3.', '.norm2.')
# modulation
			if '.modulation' in key:
				key = key.replace('.modulation', '.scale_shift_table')
		return key
	import re
# ComfyUI-specific keys to completely ignore (they don't exist in Diffusers)
	ignore_keys = {'scaled_fp8', 'model.scaled_fp8', 'diffusion_model.scaled_fp8', 'transformer.scaled_fp8'}
# Process each key in the state dict
	for key, value in state_dict.items():
# Skip scale keys (they're handled with their base weights)
		if key in scale_keys:
			continue
# Skip known ComfyUI-only flags
		if key in ignore_keys or key.replace('model.', '').replace('diffusion_model.', '').replace('transformer.', '') in ignore_keys:
			continue
# Strip common prefixes to get base for remapping
		stripped_key = key
		for prefix in ["model.", "diffusion_model.", "transformer."]:
			if stripped_key.startswith(prefix):
				stripped_key = stripped_key[len(prefix):]
				break
		target_key = remap_key(stripped_key)
# Check if this is an FP8 quantized weight - use FULL original key for scale lookup
		base = key
		if base.endswith(('.weight', '.bias')):
			base = '.'.join(base.split('.')[:-1])
# Get scale factors (they exist in original dict)
		scale_input = state_dict.get(base + '.scale_input')
		scale_weight = state_dict.get(base + '.scale_weight')
# Dequantize if needed
		if scale_input is not None or scale_weight is not None:
# Handle different scale formats (they might be float32 or bf16)
			if scale_input is not None:
				scale_input = scale_input.float()
			if scale_weight is not None:
				scale_weight = scale_weight.float()
# Dequantize: value * scale_input (if exists) * scale_weight (if exists)
			if hasattr(value, 'float'):
				value = value.float()
			dequantized = value
			if scale_input is not None:
				dequantized = dequantized * scale_input
			if scale_weight is not None:
				dequantized = dequantized * scale_weight
# Convert to target dtype
			if target_dtype and target_dtype != torch.float32:
				try:
					dequantized = dequantized.to(dtype=target_dtype)
				except:
					pass
			processed_state_dict[target_key] = dequantized
			continue
# Regular key (norms, biases, patch_embedding, etc.) - just store (with dtype conversion if needed)
		if target_dtype and value.dtype != target_dtype:
			value = value.to(target_dtype)
		processed_state_dict[target_key] = value
	print(f"DEBUG: Processed {len(processed_state_dict)} keys to Diffusers bf16 format")
	return processed_state_dict

class DualWanTransformer3DModel(torch.nn.Module):
    def __init__(
        self,
        transformer_1: WanTransformer3DModel,
        transformer_2: WanTransformer3DModel,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        boundary_ratio: float = boundary_ratio_t2v,
        low_vram: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_1: WanTransformer3DModel = transformer_1
        self.transformer_2: WanTransformer3DModel = transformer_2
        self.torch_dtype: torch.dtype = torch_dtype
        self.device_torch: torch.device = device
        self.boundary_ratio: float = boundary_ratio
        self.boundary: float = self.boundary_ratio * 1000
        self.low_vram: bool = low_vram
        self._active_transformer_name = "transformer_1"  # default to transformer_1

    @property
    def device(self) -> torch.device:
        return self.device_torch

    @property
    def dtype(self) -> torch.dtype:
        return self.torch_dtype

    @property
    def config(self):
        return self.transformer_1.config

    @property
    def transformer(self) -> WanTransformer3DModel:
        return getattr(self, self._active_transformer_name)

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for both transformers.
        """
        self.transformer_1.enable_gradient_checkpointing()
        self.transformer_2.enable_gradient_checkpointing()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # determine if doing high noise or low noise by meaning the timestep.
        # timesteps are in the range of 0 to 1000, so we can use a threshold
        with torch.no_grad():
            if timestep.float().mean().item() > self.boundary:
                t_name = "transformer_1"
            else:
                t_name = "transformer_2"

            # check if we are changing the active transformer, if so, we need to swap the one in
            # vram if low_vram is enabled
            # todo swap the loras as well
            if t_name != self._active_transformer_name:
                if self.low_vram:
                    getattr(self, self._active_transformer_name).to("cpu")
                    getattr(self, t_name).to(self.device_torch)
                    torch.cuda.empty_cache()
                self._active_transformer_name = t_name

        if self.transformer.device != hidden_states.device:
            if self.low_vram:
                # move other transformer to cpu
                other_tname = (
                    "transformer_1" if t_name == "transformer_2" else "transformer_2"
                )
                getattr(self, other_tname).to("cpu")

            self.transformer.to(hidden_states.device)

        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )

    def to(self, *args, **kwargs) -> Self:
        # do not do to, this will be handled separately
        return self


def find_safetensors_files_in_repo(repo_id: str, revision: str = "main") -> Dict[str, str]:
    """
    Search for .safetensors files in a HuggingFace repo.
    First searches in folders, then falls back to root.
    Returns dict with 'high' and 'low' keys pointing to file paths.
    """
    try:
        all_files = list_repo_files(repo_id, revision=revision)
    except Exception as e:
        raise ValueError(f"Could not list files in repo {repo_id}: {e}")
    
    # Collect all .safetensors files with their paths
    safetensor_files = {}
    
    # First, look for safetensors in subfolders
    folders_with_safetensors = {}
    root_safetensors = []
    
    for f in all_files:
        if f.endswith('.safetensors'):
            # Check if it's in a subfolder
            if '/' in f:
                folder = f.split('/')[0]
                if folder not in folders_with_safetensors:
                    folders_with_safetensors[folder] = []
                folders_with_safetensors[folder].append(f)
            else:
                root_safetensors.append(f)
    
    # Try to find HIGH and LOW in folder-based safetensors first
    for folder, files in folders_with_safetensors.items():
        for f in files:
            filename = os.path.basename(f).lower()
            if 'high' in filename:
                safetensor_files['high'] = f
            elif 'low' in filename:
                safetensor_files['low'] = f
    
    # If not found in folders, try root
    if 'high' not in safetensor_files or 'low' not in safetensor_files:
        for f in root_safetensors:
            filename = f.lower()
            if 'high' in filename and 'high' not in safetensor_files:
                safetensor_files['high'] = f
            elif 'low' in filename and 'low' not in safetensor_files:
                safetensor_files['low'] = f
    
    return safetensor_files


def find_safetensors_files_local(base_path: str) -> Dict[str, str]:
    """
    Search for .safetensors files in a local directory.
    First searches in subfolders, then falls back to root.
    Returns dict with 'high' and 'low' keys pointing to file paths.
    """
    safetensor_files = {}
    
    # First, look in subfolders
    if os.path.isdir(base_path):
        subfolders = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]
        
        for folder in subfolders:
            folder_path = os.path.join(base_path, folder)
            files = [f for f in os.listdir(folder_path) 
                    if f.endswith('.safetensors')]
            
            for f in files:
                filename = f.lower()
                if 'high' in filename:
                    safetensor_files['high'] = os.path.join(folder_path, f)
                elif 'low' in filename:
                    safetensor_files['low'] = os.path.join(folder_path, f)
    
    # If not found in subfolders, look in root
    if 'high' not in safetensor_files or 'low' not in safetensor_files:
        if os.path.isdir(base_path):
            files = [f for f in os.listdir(base_path) 
                    if f.endswith('.safetensors') and os.path.isfile(os.path.join(base_path, f))]
            
            for f in files:
                filename = f.lower()
                if 'high' in filename and 'high' not in safetensor_files:
                    safetensor_files['high'] = os.path.join(base_path, f)
                elif 'low' in filename and 'low' not in safetensor_files:
                    safetensor_files['low'] = os.path.join(base_path, f)
    
    return safetensor_files


def download_config_for_model(repo_id: str, filename: str = "config.json", 
                               revision: str = "main", cache_dir: Optional[str] = None) -> Dict:
    """
    Download config.json from a HuggingFace repo or use a default Wan2.2 config.
    """
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            force_download=False,
        )
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        # Return default Wan 2.2 14B config
        return {
            "in_channels": 16,
            "out_channels": 16,
            "hidden_size": 3072,
            "num_hidden_layers": 30,
            "num_attention_heads": 24,
            "num_key_value_heads": 24,
            "cross_attention_dim": 4096,
            "caption_projection_dim": 4096,
            "max_sequence_length": 512,
            "max_batch_size": 16,
            "activation": "gelu-approximate",
            "attention_head_dim": 128,
            "patch_size": [1, 2, 2],
            "moe_intermediate_dim": 6144,
            "num_experts": 1,
            "expert_capacity": 1.0,
            "use_moe": False,
            "qk_lora": False,
            "qk_norm": False,
            "attention_mode": "ta_flash",
        }


def download_safetensors_file(repo_id: str, filename: str, 
                               revision: str = "main", cache_dir: Optional[str] = None) -> str:
    """
    Download a safetensors file from a HuggingFace repo and return local path.
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        force_download=False,
    )
    return local_path


def load_transformer_from_safetensors(safetensors_path: str, config: Dict, 
                                        dtype: torch.dtype, device: torch.device,
                                        is_high_noise: bool = True) -> WanTransformer3DModel:
    """
    Load a WanTransformer3DModel from a safetensors file and config.
    Handles FP8 quantized weights by dequantizing them to the target dtype.
    """
    # Create model from config
    model = WanTransformer3DModel(**config)
    
    # Load weights
    state_dict = load_file(safetensors_path)

    # Debug: Print the actual dtype being used
    print(f"Target dtype: {dtype}")

    print("Processing state dict")
    processed_state_dict = _process_state_dict_for_fp8(state_dict, dtype)

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading transformer: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading transformer: {unexpected_keys[:5]}...")
    
    # Move to device and dtype
    model = model.to(dtype=dtype)
    if device != torch.device('cpu'):
        model = model.to(device)
    
    return model


def _is_fp8_quantized_key(key: str) -> bool:
    """Check if a key represents an FP8 quantized tensor."""
    return any(suffix in key for suffix in ['._data', '._scale', 'scale_input', 'scale_weight', '.scale'])


def _get_base_key(key: str) -> str:
    """Get the base key name by removing FP8 quantization suffixes."""
    for suffix in ['._data', '._scale', '.input_scale', '.output_scale']:
        if key.endswith(suffix):
            return key[:-len(suffix)]
    # Handle scale_input and scale_weight patterns
    if '.scale_input' in key:
        return key.replace('.scale_input', '')
    if '.scale_weight' in key:
        return key.replace('.scale_weight', '')
    return key


def _find_scale_key(base_key: str, all_keys: List[str]) -> str:
    """Find the scale key corresponding to a base key."""
    # Try different scale key patterns
    scale_patterns = [
        base_key + '._scale',
        base_key + '.scale',
        base_key + '.scale_weight',
        base_key + '.scale_input',
    ]
    
    for pattern in scale_patterns:
        if pattern in all_keys:
            return pattern
    
    # Try to find any key that starts with the base key and ends with scale
    for key in all_keys:
        if key.startswith(base_key) and ('scale' in key or '_scale' in key):
            return key
    
    return None


class Wan2214bModel(Wan21):
    arch = "wan22_14b"
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = False
    _wan_vae_path = "ai-toolkit/wan2.1-vae"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            model_config=model_config,
            dtype=dtype,
            custom_pipeline=custom_pipeline,
            noise_scheduler=noise_scheduler,
            **kwargs,
        )
        # target it so we can target both transformers
        self.target_lora_modules = ["DualWanTransformer3DModel"]
        self._wan_cache = None

        self.is_multistage = True
        # multistage boundaries split the models up when sampling timesteps
        # for wan 2.2 14b. the timesteps are 1000-875 for transformer 1 and 875-0 for transformer 2
        self.multistage_boundaries: List[float] = [0.875, 0.0]

        self.train_high_noise = model_config.model_kwargs.get("train_high_noise", True)
        self.train_low_noise = model_config.model_kwargs.get("train_low_noise", True)

        self.trainable_multistage_boundaries: List[int] = []
        if self.train_high_noise:
            self.trainable_multistage_boundaries.append(0)
        if self.train_low_noise:
            self.trainable_multistage_boundaries.append(1)

        if len(self.trainable_multistage_boundaries) == 0:
            raise ValueError(
                "At least one of train_high_noise or train_low_noise must be True in model.model_kwargs"
            )
        
        # if we are only training one or the other, the target LoRA modules will be the wan transformer class
        if not self.train_high_noise or not self.train_low_noise:
            self.target_lora_modules = ["WanTransformer3DModel"]

    def get_quantization_exclude_modules(self):
        # the timestep/text conditioning embedders and the final projection feed
        # every downstream modulation; keep them in full precision when quantizing.
        # names are relative to each individual transformer (they quantize separately)
        return ["condition_embedder*", "proj_out*"]

    @property
    def max_step_saves_to_keep_multiplier(self):
        # the cleanup mechanism checks this to see how many saves to keep
        # if we are training a LoRA, we need to set this to 2 so we keep both the high noise and low noise LoRAs at saves to keep
        if (
            self.network is not None
            and self.network.network_config.split_multistage_loras
        ):
            return 2
        return 1

    def load_model(self):
        # load model from patent parent. Wan21 not immediate parent
        # super().load_model()
        super().load_model()

        # we have to split up the model on the pipeline
        self.pipeline.transformer = self.model.transformer_1
        self.pipeline.transformer_2 = self.model.transformer_2

        # patch the condition embedder
        self.model.transformer_1.condition_embedder.forward = partial(
            time_text_monkeypatch, self.model.transformer_1.condition_embedder
        )
        self.model.transformer_2.condition_embedder.forward = partial(
            time_text_monkeypatch, self.model.transformer_2.condition_embedder
        )

    def get_bucket_divisibility(self):
        # 8x compression  and 2x2 patch size
        return 16

    def load_wan_transformer(self, transformer_path, subfolder=None):
        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.2 models"
            )

        if (
            self.model_config.assistant_lora_path is not None
            or self.model_config.inference_lora_path is not None
        ):
            raise ValueError(
                "Assistant LoRA is not supported for Wan2.2 models currently"
            )

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for Wan2.2 models currently"
            )

        # Determine if we're loading from HuggingFace or local path
        is_hf_path = '/' in transformer_path and not os.path.exists(transformer_path)
        
        # Check if standard transformer folders exist (for backward compatibility)
        has_standard_structure = False
        
        if is_hf_path:
            # Check if this is a HF repo with standard structure
            try:
                files = list_repo_files(transformer_path)
                has_standard_structure = 'transformer/config.json' in files or 'transformer_2/config.json' in files
            except:
                has_standard_structure = False
        else:
            # Check local paths
            transformer_1_path = transformer_path if subfolder else os.path.join(transformer_path, "transformer")
            # Check if this is a LoRA/custom path (contains .safetensors files with high/low names)
            if os.path.isdir(transformer_path):
                lora_files = [f for f in os.listdir(transformer_path) 
                             if f.endswith('.safetensors') and os.path.isfile(os.path.join(transformer_path, f))]
                has_lora_files = any('high' in f.lower() or 'low' in f.lower() for f in lora_files)
                if has_lora_files:
                    # LoRA format - use custom loader
                    has_standard_structure = False
                else:
                    # Check for standard structure
                    has_standard_structure = os.path.exists(transformer_1_path) and os.path.exists(os.path.join(transformer_1_path, "config.json"))
        
        if has_standard_structure:
            # Use original loading method for standard diffusers format
            return self._load_wan_transformer_standard(transformer_path, subfolder)
        else:
            # Use new method for custom safetensors format
            return self._load_wan_transformer_custom(transformer_path, subfolder, is_hf_path)

    def _load_wan_transformer_standard(self, transformer_path, subfolder=None):
        """
        Load transformers using the standard diffusers format with transformer/ and transformer_2/ folders.
        """
        # transformer path can be a directory that ends with /transformer or a hf path.

        transformer_path_1 = transformer_path
        subfolder_1 = subfolder

        transformer_path_2 = transformer_path
        subfolder_2 = subfolder
        if subfolder_2 is None:
            # we have a local path, replace it with transformer_2 folder
            transformer_path_2 = os.path.join(
                os.path.dirname(transformer_path_1), "transformer_2"
            )
        else:
            # we have a hf path, replace it with transformer_2 subfolder
            subfolder_2 = "transformer_2"
        
        # Load state dict for debug output using _process_state_dict_for_fp8
        # This works for both HuggingFace models (downloaded) and local models
        dtype = self.torch_dtype
        
        # Check if this is a HuggingFace path
        is_hf_path = '/' in transformer_path and not os.path.exists(transformer_path)
        
        state_dict = None
        if is_hf_path:
            # For HuggingFace models, we need to download the safetensors file
            try:
                # Try to find and download safetensors file from the model repo
                from huggingface_hub import list_repo_files
                files = list_repo_files(transformer_path)
                # Look for model-*.safetensors or diffusion_model.safetensors
                safetensors_files = [f for f in files if f.endswith('.safetensors')]
                if safetensors_files:
                    # Download the first safetensors file
                    model_file = safetensors_files[0]
                    local_path = hf_hub_download(
                        repo_id=transformer_path,
                        filename=model_file,
                    )
                    state_dict = load_file(local_path)
            except Exception as e:
                print(f"Could not load state dict from HF repo: {e}")
                state_dict = None
        else:
            # For local models, try to find safetensors file
            if os.path.exists(transformer_path_2):
                safetensors_files = [f for f in os.listdir(transformer_path_2) 
                                   if f.endswith('.safetensors')]
                if safetensors_files:
                    state_dict = load_file(os.path.join(transformer_path_2, safetensors_files[0]))
        
        # Process state dict for debug output (works for both FP8 and bf16 models)
        if state_dict is not None:
            processed_state_dict = _process_state_dict_for_fp8(state_dict, dtype)
        
        self.print_and_status_update("Loading transformer 1 (standard format)")
        dtype = self.torch_dtype
        transformer_1 = WanTransformer3DModel.from_pretrained(
            transformer_path_1,
            subfolder=subfolder_1,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        flush()

        if self.model_config.low_vram:
            # quantize on the device
            transformer_1.to('cpu', dtype=dtype)
            flush()
        else:
            transformer_1.to(self.device_torch, dtype=dtype)
            flush()

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            # todo handle two ARAs
            self.print_and_status_update("Quantizing Transformer 1")
            quantize_model(self, transformer_1)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer 1 to CPU")
            transformer_1.to("cpu")
        else:
            transformer_1.to(self.device_torch)

        self.print_and_status_update("Loading transformer 2 (standard format)")
        dtype = self.torch_dtype
        transformer_2 = WanTransformer3DModel.from_pretrained(
            transformer_path_2,
            subfolder=subfolder_2,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        flush()

        if self.model_config.low_vram:
            # quantize on the device
            transformer_2.to('cpu', dtype=dtype)
            flush()
        else:
            transformer_2.to(self.device_torch, dtype=dtype)
            flush()

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            # todo handle two ARAs
            self.print_and_status_update("Quantizing Transformer 2")
            quantize_model(self, transformer_2)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer 2 to CPU")
            transformer_2.to("cpu")
        else:
            transformer_2.to(self.device_torch)
        
        return self._create_dual_transformer(transformer_1, transformer_2)

    def _load_wan_transformer_custom(self, transformer_path, subfolder=None, is_hf_path: bool = True):
        """
        Load transformers from custom safetensors files (HIGH and LOW noise models).
        Searches for .safetensors files in folders or root of the repo/path.
        """
        dtype = self.torch_dtype
        device = self.device_torch
        
        self.print_and_status_update("Searching for safetensors files in repo")
        
        safetensor_files = {}
        
        if is_hf_path:
            # HuggingFace repo
            safetensor_files = find_safetensors_files_in_repo(transformer_path)
        else:
            # Local path
            safetensor_files = find_safetensors_files_local(transformer_path)
        
        if 'high' not in safetensor_files:
            raise ValueError(
                f"Could not find a .safetensors file with 'high' in the name in {transformer_path}. "
                f"Found files: {list(safetensor_files.keys())}"
            )
        
        if 'low' not in safetensor_files:
            raise ValueError(
                f"Could not find a .safetensors file with 'low' in the name in {transformer_path}. "
                f"Found files: {list(safetensor_files.keys())}"
            )
        
        self.print_and_status_update(f"Found HIGH noise model: {safetensor_files['high']}")
        self.print_and_status_update(f"Found LOW noise model: {safetensor_files['low']}")
        
        # Get config (try to download from repo or use default)
        config = None
        if is_hf_path:
            try:
                # Try to find config.json in the same folder as the safetensors
                high_folder = os.path.dirname(safetensor_files['high'])
                if high_folder:
                    config_filename = os.path.join(high_folder, "config.json")
                    config = download_config_for_model(transformer_path, config_filename)
                else:
                    # In root
                    config = download_config_for_model(transformer_path, "config.json")
            except Exception as e:
                self.print_and_status_update(f"Could not download config.json, using default: {e}")
                config = download_config_for_model(transformer_path, "config.json")
        else:
            # Local path - check for config.json in same folder as safetensors
            high_path = safetensor_files['high']
            config_path = os.path.join(os.path.dirname(high_path), "config.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Use default config
                config = download_config_for_model("", "config.json")
        
        # Download or load safetensors files
        self.print_and_status_update("Loading HIGH noise transformer")
        
        if is_hf_path:
            high_path = download_safetensors_file(transformer_path, safetensor_files['high'])
        else:
            high_path = safetensor_files['high']
        
        transformer_1 = load_transformer_from_safetensors(
            high_path, config, dtype, device, is_high_noise=True
        )
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving HIGH noise transformer to CPU")
            transformer_1.to('cpu')
            flush()
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing HIGH noise Transformer")
            quantize_model(self, transformer_1)
            flush()
        
        self.print_and_status_update("Loading LOW noise transformer")
        
        if is_hf_path:
            low_path = download_safetensors_file(transformer_path, safetensor_files['low'])
        else:
            low_path = safetensor_files['low']
        
        transformer_2 = load_transformer_from_safetensors(
            low_path, config, dtype, device, is_high_noise=False
        )
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving LOW noise transformer to CPU")
            transformer_2.to('cpu')
            flush()
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing LOW noise Transformer")
            quantize_model(self, transformer_2)
            flush()
        
        return self._create_dual_transformer(transformer_1, transformer_2)

    def _create_dual_transformer(self, transformer_1, transformer_2):
        """
        Create DualWanTransformer3DModel from two transformers.
        """
        layer_offloading_transformer = self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0
        # make the combined model
        self.print_and_status_update("Creating DualWanTransformer3DModel")
        transformer = DualWanTransformer3DModel(
            transformer_1=transformer_1,
            transformer_2=transformer_2,
            torch_dtype=self.torch_dtype,
            device=self.device_torch,
            boundary_ratio=boundary_ratio_t2v,
            low_vram=self.model_config.low_vram,
        )
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is not None:
            # apply the accuracy recovery adapter to both transformers
            self.print_and_status_update("Applying Accuracy Recovery Adapter to Transformers")
            quantize_model(self, transformer)
            flush()
            
        
        if layer_offloading_transformer:
            MemoryManager.attach(
                transformer_1,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_1.scale_shift_table] + [block.scale_shift_table for block in transformer_1.blocks]
            )
            MemoryManager.attach(
                transformer_2,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_2.scale_shift_table] + [block.scale_shift_table for block in transformer_2.blocks]
            )

        return transformer

    def get_generation_pipeline(self):
        # todo unipc got broken in a diffusers update. Use euler for now.
        # scheduler = UniPCMultistepScheduler(**self._wan_generation_scheduler_config)
        scheduler = self.get_train_scheduler()
        pipeline = Wan22Pipeline(
            vae=self.vae,
            transformer=self.model.transformer_1,
            transformer_2=self.model.transformer_2,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=scheduler,
            expand_timesteps=self._wan_expand_timesteps,
            device=self.device_torch,
            aggressive_offload=self.model_config.low_vram,
            # todo detect if it is i2v or t2v
            boundary_ratio=boundary_ratio_t2v,
        )

        # pipeline = pipeline.to(self.device_torch)

        return pipeline

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def get_base_model_version(self):
        return "wan_2.2_14b"

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs,
    ):
        # todo do we need to override this? Adjust timesteps?
        return super().get_noise_prediction(
            latent_model_input=latent_model_input,
            timestep=timestep,
            text_embeddings=text_embeddings,
            batch=batch,
            **kwargs,
        )

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer_combo: DualWanTransformer3DModel = unwrap_model(self.model)
        transformer_combo.transformer_1.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        transformer_combo.transformer_2.save_pretrained(
            save_directory=os.path.join(output_path, "transformer_2"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def save_lora(
        self,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.network.network_config.split_multistage_loras:
            # just save as a combo lora
            save_file(state_dict, output_path, metadata=metadata)
            return

        # we need to build out both dictionaries for high and low noise LoRAs
        high_noise_lora = {}
        low_noise_lora = {}
        
        only_train_high_noise = self.train_high_noise and not self.train_low_noise
        only_train_low_noise = self.train_low_noise and not self.train_high_noise

        for key in state_dict:
            if ".transformer_1." in key or only_train_high_noise:
                # this is a high noise LoRA
                new_key = key.replace(".transformer_1.", ".")
                high_noise_lora[new_key] = state_dict[key]
            elif ".transformer_2." in key or only_train_low_noise:
                # this is a low noise LoRA
                new_key = key.replace(".transformer_2.", ".")
                low_noise_lora[new_key] = state_dict[key]

        # loras have either LORA_MODEL_NAME_000005000.safetensors or LORA_MODEL_NAME.safetensors
        if len(high_noise_lora.keys()) > 0:
            # save the high noise LoRA
            high_noise_lora_path = output_path.replace(
                ".safetensors", "_high_noise.safetensors"
            )
            save_file(high_noise_lora, high_noise_lora_path, metadata=metadata)

        if len(low_noise_lora.keys()) > 0:
            # save the low noise LoRA
            low_noise_lora_path = output_path.replace(
                ".safetensors", "_low_noise.safetensors"
            )
            save_file(low_noise_lora, low_noise_lora_path, metadata=metadata)

    def load_lora(self, file: str):
        # if it doesnt have high_noise or low_noise, it is a combo LoRA
        if (
            "_high_noise.safetensors" not in file
            and "_low_noise.safetensors" not in file
        ):
            # this is a combined LoRA, we dont need to split it up
            sd = load_file(file)
            return sd

        # we may have been passed the high_noise or the low_noise LoRA path, but we need to load both
        high_noise_lora_path = file.replace(
            "_low_noise.safetensors", "_high_noise.safetensors"
        )
        low_noise_lora_path = file.replace(
            "_high_noise.safetensors", "_low_noise.safetensors"
        )

        combined_dict = {}

        if os.path.exists(high_noise_lora_path) and self.train_high_noise:
            # load the high noise LoRA
            high_noise_lora = load_file(high_noise_lora_path)
            for key in high_noise_lora:
                new_key = key.replace(
                    "diffusion_model.", "diffusion_model.transformer_1."
                )
                combined_dict[new_key] = high_noise_lora[key]
        if os.path.exists(low_noise_lora_path) and self.train_low_noise:
            # load the low noise LoRA
            low_noise_lora = load_file(low_noise_lora_path)
            for key in low_noise_lora:
                new_key = key.replace(
                    "diffusion_model.", "diffusion_model.transformer_2."
                )
                combined_dict[new_key] = low_noise_lora[key]
        
        # if we are not training both stages, we wont have transformer designations in the keys
        if not self.train_high_noise or not self.train_low_noise:
            new_dict = {}
            for key in combined_dict:
                if ".transformer_1." in key:
                    new_key = key.replace(".transformer_1.", ".")
                elif ".transformer_2." in key:
                    new_key = key.replace(".transformer_2.", ".")
                else:
                    new_key = key
                new_dict[new_key] = combined_dict[key]
            combined_dict = new_dict

        return combined_dict
    
    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)

        if self.use_vae_tiling:
            # set vae to tile decode
            pipeline.vae.enable_tiling()

        # todo, figure out how to do video
        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra
        )[0]

        if self.use_vae_tiling:
            # restore no tiling
            pipeline.vae.disable_tiling()

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def get_model_to_train(self):
        # todo, loras wont load right unless they have the transformer_1 or transformer_2 in the key.
        # called when setting up the LoRA. We only need to get the model for the stages we want to train.
        if self.train_high_noise and self.train_low_noise:
            # we are training both stages, return the unified model
            return self.model
        elif self.train_high_noise:
            # we are only training the high noise stage, return transformer_1
            return self.model.transformer_1
        elif self.train_low_noise:
            # we are only training the low noise stage, return transformer_2
            return self.model.transformer_2
        else:
            raise ValueError(
                "At least one of train_high_noise or train_low_noise must be True in model.model_kwargs"
            )
