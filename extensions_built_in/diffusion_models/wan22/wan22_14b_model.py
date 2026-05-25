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
from toolkit.train_tools import get_torch_dtype
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

# Conditional import of the custom FP8 extension
try:
	import fp8_ops
	FP8_OPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
	FP8_OPS_AVAILABLE = False
	print("fp8_ops import error!")

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

import torch
import torch.nn.functional as F

class FP8PatchEmbed(torch.nn.Module):
	def __init__(self, original_conv: torch.nn.Conv3d):
		super().__init__()
		self.original = original_conv
		self.in_channels = original_conv.in_channels
		self.embed_dim = original_conv.out_channels
		self.kernel_size = original_conv.kernel_size
		self.stride = original_conv.stride

	def forward(self, x):
        # x : [B, C_in, T, H, W]
		B, C, T, H, W = x.shape

        # Step 1: unfold spatial patches (2x2, stride 2) over each frame
        # Permute to [B, T, C, H, W] and merge batch & time
		x = x.permute(0, 2, 1, 3, 4)          # B, T, C, H, W
		x = x.reshape(B * T, C, H, W)          # (B*T), C, H, W

        # Step 2: apply unfold / im2col (spatial only)
		if x.dtype == torch.float8_e4m3fn and FP8_OPS_AVAILABLE:
            # custom fp8 im2col
			x = fp8_ops.fp8_im2col(
                x, kernel_h=2, kernel_w=2,
                stride_h=2, stride_w=2
            )                                   # (B*T), C*4, num_patches
		else:
			x = F.unfold(x, kernel_size=(2,2), stride=(2,2))   # same shape

        # Step 3: transpose to (batch*tokens, num_patches, in_features)
		x = x.transpose(1, 2)                  # (B*T), num_patches, C*4

        # Step 4: flatten batch and token dimensions for 2D matmul
		B_T, num_patches, in_features = x.shape
		x_flat = x.reshape(-1, in_features)    # (B*T * num_patches), in_features

        # Step 5: get weight and bias (convert to fp8 once, cache if possible)
        # Here we assume self.original is the original nn.Linear layer.
        # For efficiency, you might store fp8 versions in __init__.
		weight = self.original.weight.view(self.embed_dim, -1)
		bias = self.original.bias

		if x_flat.dtype == torch.float8_e4m3fn and FP8_OPS_AVAILABLE:
            # Convert weight to fp8 (do this once outside the loop in real code)
			weight_fp8 = weight.to(torch.float8_e4m3fn)
            # Matrix multiply using custom fp8 kernel
			out_flat = fp8_ops.fp8_matmul(x_flat, weight_fp8.t())  # (B*T*num_patches, embed_dim)
		else:
            # Fallback to regular matmul (will upcast if needed)
			out_flat = torch.matmul(x_flat, weight.t())             # (B*T*num_patches, embed_dim)

        # Step 6: reshape back to separate batch/time and patches
		out = out_flat.reshape(B_T, num_patches, self.embed_dim)    # (B*T), num_patches, embed_dim

        # Step 7: add bias (if any)
		# Debug flag – set to True when troubleshooting
		DEBUG_BIAS = True

		if bias is not None:
			if DEBUG_BIAS:
				print(f"[DEBUG] out shape: {out.shape}, dtype: {out.dtype}")
				print(f"[DEBUG] bias shape: {bias.shape}, dtype: {bias.dtype}")
				# Check for NaNs/Infs if tensors are float (not fp8)
				# if out.dtype.is_floating_point:
				# 	print(f"[DEBUG] out has nan: {torch.isnan(out).any()}, inf: {torch.isinf(out).any()}")
				# if bias.dtype.is_floating_point:
				# 	print(f"[DEBUG] bias has nan: {torch.isnan(bias).any()}, inf: {torch.isinf(bias).any()}")

            # Bias is added per output feature, so we can add directly
			if out.dtype == torch.float8_e4m3fn and FP8_OPS_AVAILABLE:
				bias_expanded = bias.view(1, 1, -1).expand_as(out)
				if DEBUG_BIAS:
					print(f"[DEBUG] bias_expanded shape: {bias_expanded.shape}, dtype: {bias_expanded.dtype}")
                    # Check contiguity – custom kernels may require contiguous inputs
					print(f"[DEBUG] out is contiguous: {out.is_contiguous()}")
					print(f"[DEBUG] bias_expanded is contiguous: {bias_expanded.is_contiguous()}")
				out = fp8_ops.fp8_add(out, bias_expanded)
			else:
                # For non‑fp8, use regular addition
				out = out + bias

			if DEBUG_BIAS:
				print(f"[DEBUG] after addition – out shape: {out.shape}, dtype: {out.dtype}")
				if out.dtype.is_floating_point:
					print(f"[DEBUG] out has nan: {torch.isnan(out).any()}, inf: {torch.isinf(out).any()}")

        # Step 8: final reshape to [B, embed_dim, T * num_patches]
        # This matches the expected input for the transformer blocks.
		out = out.view(B, -1, self.embed_dim).transpose(1, 2)       # B, embed_dim, T * num_patches

		return out


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
        # Convert string dtype to torch.dtype if needed
        self.torch_dtype: torch.dtype = get_torch_dtype(torch_dtype) if torch_dtype is not None else torch_dtype
        self.device_torch: torch.device = torch.device(device) if device is not None else device
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


def load_config_from_safetensors(safetensors_path: str) -> Optional[Dict]:
    """
    Try to load config from safetensors file metadata.
    Returns None if no config found.
    """
    try:
        # Try to load metadata from safetensors file
        from safetensors import safe_open
        with safe_open(safetensors_path, framework="pt") as f:
            metadata = f.metadata()
            if metadata is not None:
                # Check if there's config info in metadata
                # Some safetensors files have config in metadata
                if "config" in metadata:
                    try:
                        return json.loads(metadata["config"])
                    except:
                        pass
                
                # Try to infer config from common metadata fields
                # Wan models often have model_format or architecture info
                print(f"Safetensors metadata keys: {list(metadata.keys())}")
    except Exception as e:
        print(f"Could not load config from safetensors metadata: {e}")
    
    return None


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
            config = json.load(f)
            # Fix config: WanTransformer3DModel uses 'dim' not 'hidden_size'
            if "hidden_size" in config and "dim" not in config:
                config["dim"] = config.pop("hidden_size")
            return config
    except Exception as e:
        print(f"Could not download config.json: {e}")
        # Return default Wan 2.2 14B config
        # Note: WanTransformer3DModel uses 'dim' not 'hidden_size'
        return {
            "_class_name": "WanTransformer3DModel",
            "_diffusers_version": "0.35.0.dev0",
            "_name_or_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "added_kv_proj_dim": 'null',
            "attention_head_dim": 128,
            "cross_attn_norm": 'true',
            "eps": 1e-06,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "image_dim": 'null',
            "in_channels": 36,
            "num_attention_heads": 40,
            "num_layers": 40,
            "out_channels": 16,
            "patch_size": [
                1,
                2,
                2
            ],
            "pos_embed_seq_len": 'null',
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 1024,
            "text_dim": 4096
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


def get_torch_dtype_typed(dtype_str):
    """Get torch dtype with support for float8 types"""
    from toolkit.train_tools import get_torch_dtype
    return get_torch_dtype(dtype_str)


def load_transformer_from_safetensors(safetensors_path: str, config: Dict, 
                                        dtype: torch.dtype, device: torch.device,
                                        is_high_noise: bool = True) -> WanTransformer3DModel:
    """
    Load a WanTransformer3DModel from a safetensors file and config.
    """
    # Fix config: WanTransformer3DModel uses 'dim' not 'hidden_size'
    if "hidden_size" in config and "dim" not in config:
        config["dim"] = config.pop("hidden_size")
    
    # Create model from config
    model = WanTransformer3DModel(**config)
    
    # Load weights
    state_dict = load_file(safetensors_path)
    
    # Handle potential key prefix differences
    # Common prefixes: "model.", "diffusion_model.", ""
    processed_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes if present
        new_key = key
        for prefix in ["model.", "diffusion_model.", "transformer."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        processed_state_dict[new_key] = value
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading transformer: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading transformer: {unexpected_keys[:5]}...")
    
    # Move to device and dtype
    model = model.to(dtype=dtype)
    print("Moved model with dtype == ",dtype)
    if device != torch.device('cpu'):
        model = model.to(device)
    
    return model


class Wan2214bModel(Wan21):
    arch = "wan22_14b"
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = False
    _wan_vae_path = "ai-toolkit/wan2.1-vae"
    # Default flow shift values (can be overridden in config)
    _default_flow_shift = 5.0
    _default_inference_flow_shift = 5.0

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        train_flow_shift=None,
        sample_flow_shift=None,
        inference_sampler="unipc",  # Default to unipc for better performance
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

        # Store flow shift values from config
        # train_flow_shift: for training scheduler
        # sample_flow_shift: for inference scheduler
        self.train_flow_shift = train_flow_shift if train_flow_shift is not None else self._default_flow_shift
        self.sample_flow_shift = sample_flow_shift if sample_flow_shift is not None else self._default_inference_flow_shift
        self.inference_sampler = inference_sampler

        # Detect if this is I2V or T2V model
        self.is_i2v = 'i2v' in model_config.name_or_path.lower()
        self.boundary_ratio = boundary_ratio_i2v if self.is_i2v else boundary_ratio_t2v

        # multistage boundaries split the models up when sampling timesteps
        # for wan 2.2 14b I2V: timesteps 1000-900 for transformer 1 and 900-0 for transformer 2
        # for wan 2.2 14b T2V: timesteps 1000-875 for transformer 1 and 875-0 for transformer 2
        self.multistage_boundaries: List[float] = [self.boundary_ratio, 0.0]

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

        # Skip quantization if dtype is already float8 (pre-quantized model)
        is_already_quantized = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        
        if self.model_config.quantize and not is_already_quantized and self.model_config.accuracy_recovery_adapter is None:
            # todo handle two ARAs
            self.print_and_status_update("Quantizing Transformer 1")
            quantize_model(self, transformer_1)
            flush()
        elif is_already_quantized:
            self.print_and_status_update("Skipping quantization - model is already in float8 format")

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

        # Skip quantization if dtype is already float8 (Pre-quantized model)
        if self.model_config.quantize and not is_already_quantized and self.model_config.accuracy_recovery_adapter is None:
            # todo handle two ARAs
            self.print_and_status_update("Quantizing Transformer 2")
            quantize_model(self, transformer_2)
            flush()
        elif is_already_quantized and self.model_config.quantize:
            self.print_and_status_update("Skipping quantization - model is already in float8 format")

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer 2 to CPU")
            transformer_2.to("cpu")
        else:
            transformer_2.to(self.device_torch)
        
        # Use the actual dtype from the transformers
        actual_dtype = transformer_1.dtype
        
        return self._create_dual_transformer(transformer_1, transformer_2, actual_dtype)

    def _load_wan_transformer_custom(self, transformer_path, subfolder=None, is_hf_path: bool = True):
        """
        Load transformers from custom safetensors files (HIGH and LOW noise models).
        Searches for .safetensors files in folders or root of the repo/path.
        """
        dtype = self.torch_dtype
        device = self.device_torch
        
        model_dtype = dtype
        
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
        else:
            # model_dtype = torch.float8_e4m3fn
            pass
        
        if 'low' not in safetensor_files:
            raise ValueError(
                f"Could not find a .safetensors file with 'low' in the name in {transformer_path}. "
                f"Found files: {list(safetensor_files.keys())}"
            )
        else:
            # model_dtype = torch.float8_e4m3fn
            pass
        
        # Check if model is already in fp8 format
        is_already_quantized = model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        
        # Update self.torch_dtype to match the actual model dtype when loading fp8 models
        # This ensures consistency across the model class
        if is_already_quantized:
            self.torch_dtype = model_dtype
            self.print_and_status_update(f"Detected fp8 model, updating torch_dtype to {model_dtype}")
        
        self.print_and_status_update(f"Found HIGH noise model: {safetensor_files['high']}")
        self.print_and_status_update(f"Found LOW noise model: {safetensor_files['low']}")
        
        # Download safetensors files first so we can extract config from them
        self.print_and_status_update("Downloading HIGH noise model")
        
        if is_hf_path:
            high_path = download_safetensors_file(transformer_path, safetensor_files['high'])
        else:
            high_path = safetensor_files['high']
        
        # Try to load config from safetensors file first
        config = load_config_from_safetensors(high_path)
        
        if config is None:
            # Try to download config.json from repo
            self.print_and_status_update("No config in safetensors, trying to download config.json from repo")
            config = download_config_for_model(transformer_path, "config.json")
        
        self.print_and_status_update(f"Using config: {config}")
        
        # Now load the transformers using the already downloaded high_path
        self.print_and_status_update("Loading HIGH noise transformer")
        
        transformer_1 = load_transformer_from_safetensors(
            high_path, config, model_dtype, device, is_high_noise=True
        )
        if model_dtype == torch.float8_e4m3fn:
            transformer_1.patch_embedding = FP8PatchEmbed(transformer_1.patch_embedding)
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving HIGH noise transformer to CPU")
            transformer_1.to('cpu')
            flush()
        
        # Skip quantization if dtype is already float8 (pre-quantized model)
        if self.model_config.quantize and not is_already_quantized and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing HIGH noise Transformer")
            quantize_model(self, transformer_1)
            flush()
        elif is_already_quantized and self.model_config.quantize:
            self.print_and_status_update("Skipping quantization - model is already in float8 format")
        
        self.print_and_status_update("Loading LOW noise transformer")
        
        if is_hf_path:
            low_path = download_safetensors_file(transformer_path, safetensor_files['low'])
        else:
            low_path = safetensor_files['low']
        
        transformer_2 = load_transformer_from_safetensors(
            low_path, config, model_dtype, device, is_high_noise=False
        )
        
        if model_dtype == torch.float8_e4m3fn:
            transformer_2.patch_embedding = FP8PatchEmbed(transformer_2.patch_embedding)
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving LOW noise transformer to CPU")
            transformer_2.to('cpu')
            flush()
        
        if self.model_config.quantize and not is_already_quantized and self.model_config.accuracy_recovery_adapter is None:
            self.print_and_status_update("Quantizing LOW noise Transformer")
            quantize_model(self, transformer_2)
            flush()
        elif is_already_quantized and self.model_config.quantize:
            self.print_and_status_update("Skipping quantization - model is already in float8 format")
        
        # Use self.torch_dtype which has been updated to fp8 if needed
        # This ensures consistency between self.torch_dtype and the actual model dtype
        
        return self._create_dual_transformer(transformer_1, transformer_2, self.torch_dtype)

    def _create_dual_transformer(self, transformer_1, transformer_2, dtype=None):
        """
        Create DualWanTransformer3DModel from two transformers.
        """
        layer_offloading_transformer = self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0
        # make the combined model
        # Use provided dtype if available, otherwise fall back to self.torch_dtype
        # This ensures consistency when loading fp8 models directly
        actual_dtype = dtype if dtype is not None else self.torch_dtype
        
        self.print_and_status_update("Creating DualWanTransformer3DModel")
        transformer = DualWanTransformer3DModel(
            transformer_1=transformer_1,
            transformer_2=transformer_2,
            torch_dtype=actual_dtype,
            device=self.device_torch,
            boundary_ratio=self.boundary_ratio,
            low_vram=self.model_config.low_vram,
        )
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is not None:
            # Skip if already in fp8 format
            is_already_quantized = actual_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            if not is_already_quantized:
                # apply the accuracy recovery adapter to both transformers
                self.print_and_status_update("Applying Accuracy Recovery Adapter to Transformers")
                quantize_model(self, transformer)
                flush()
            else:
                self.print_and_status_update("Skipping ARA - model is already in float8 format")
            
        
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

    def get_generation_pipeline(self, inference_sampler=None, flow_shift=None):
        # Use instance values if not provided
        if inference_sampler is None:
            inference_sampler = self.inference_sampler
        if flow_shift is None:
            flow_shift = self.sample_flow_shift
        
        # Build scheduler config with custom flow_shift if provided
        sched_config = dict(scheduler_configUniPC)
        if flow_shift is not None:
            sched_config['flow_shift'] = flow_shift
        
        # Create scheduler based on inference_sampler setting
        if inference_sampler and inference_sampler.lower() == 'unipc':
            scheduler = UniPCMultistepScheduler(**sched_config)
        elif inference_sampler and inference_sampler.lower() == 'euler':
            from diffusers import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler(**sched_config)
        elif inference_sampler and inference_sampler.lower() == 'flowmatch':
            from diffusers import FlowMatchEulerDiscreteScheduler
            scheduler = FlowMatchEulerDiscreteScheduler(**sched_config)
        else:
            # Default to UniPC
            scheduler = UniPCMultistepScheduler(**sched_config)
        
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
            boundary_ratio=self.boundary_ratio,
        )

        # pipeline = pipeline.to(self.device_torch)

        return pipeline

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler(flow_shift=None, noise_scheduler=None):
        # Default flow shift for wan22 14b
        default_shift = 5.0
        shift = flow_shift if flow_shift is not None else default_shift
        
        # Use noise_scheduler from config if provided, default to unipc
        scheduler_type = noise_scheduler if noise_scheduler else "unipc"
        
        if scheduler_type.lower() == "unipc":
            train_config = {
                "num_train_timesteps": 1000,
                "use_dynamic_shifting": False,
                "flow_shift": shift,
                "predict_x0": True,
                "solver_order": 2,
                "solver_type": "bh2",
                "lower_order_final": True,
            }
            scheduler = UniPCMultistepScheduler(**train_config)
        else:
            # Default to flowmatch for backward compatibility
            train_config = {
                "num_train_timesteps": 1000,
                "shift": shift,
                "use_dynamic_shifting": False,
            }
            scheduler = CustomFlowMatchEulerDiscreteScheduler(**train_config)
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
            # NAG (Negative Attention Guidance) parameters
            nag_scale=gen_config.nag_scale,
            nag_alpha=gen_config.nag_alpha,
            nag_tau=gen_config.nag_tau,
            **extra
        )[0]

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
