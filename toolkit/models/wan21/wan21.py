# WIP, coming soon ish
from functools import partial
import torch
import yaml
from safetensors.torch import load_file as load_safetensors
from toolkit.accelerator import unwrap_model
from toolkit.train_tools import get_torch_dtype
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.memory_management.manager import MemoryManager
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import  WanPipeline, WanTransformer3DModel, AutoencoderKL
from .autoencoder_kl_wan import (
    AutoencoderKLWan,
    detect_alternative_vae_naming,
    normalize_vae_state_dict,
)
import os
import sys

import weakref
import torch
import yaml

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds

import os
import copy
from toolkit.config_modules import ModelConfig, GenerateImageConfig, ModelArch
import torch
from optimum.quanto import freeze, qfloat8, QTensor, qint4
from toolkit.util.quantize import quantize, get_qtype
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from typing import TYPE_CHECKING, List
from toolkit.accelerator import unwrap_model
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import XLA_AVAILABLE
# from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from typing import Any, Callable, Dict, List, Optional, Union
from toolkit.models.wan21.wan_lora_convert import convert_to_diffusers, convert_to_original
from toolkit.util.quantize import quantize_model
from toolkit.models.loaders.umt5 import get_umt5_encoder

# for generation only?
scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.33.0.dev0",
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
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False
}

# for training. I think it is right
scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False
}



def _get_vae_safetensor_file(source: str, subfolder: Optional[str] = None) -> str:
    """
    Return the path to the VAE safetensor file in `source`.

    `source` may be:
      - A local directory (with or without `subfolder`).
      - A HuggingFace repo ID — in this case we download the file listing
        first and pick the first `*.safetensors` file we find.
    """
    import os
    from huggingface_hub import hf_hub_download, list_repo_files

    def _in_dir(dir_path: str) -> str:
        for f in os.listdir(dir_path):
            if f.endswith(".safetensors"):
                return os.path.join(dir_path, f)
        # Try subfolder
        if subfolder:
            sub = os.path.join(dir_path, subfolder)
            if os.path.isdir(sub):
                for f in os.listdir(sub):
                    if f.endswith(".safetensors"):
                        return os.path.join(sub, f)
        raise FileNotFoundError(
            f"No .safetensors file found in {dir_path}"
            + (f"/{subfolder}" if subfolder else "")
        )

    if os.path.exists(source):
        return _in_dir(source)

    # HuggingFace repo — download the VAE config file to discover the
    # repository type, then fetch the first safetensors file.
    # We use `hf_hub_download` with `filename="diffusers_state_dict.safetensors"`
    # as a fallback, otherwise list the repo files.
    try:
        files = list_repo_files(source)
    except Exception:
        # If listing fails, try downloading the default filename.
        return hf_hub_download(source, filename="diffusers_state_dict.safetensors")

    # Prefer explicitly-named VAE files.
    vae_candidates = [
        f for f in files
        if f.endswith(".safetensors") and ("vae" in f.lower() or "diffusers_state_dict" in f)
    ]
    if vae_candidates:
        return hf_hub_download(source, filename=vae_candidates[0])

    # Fallback: first safetensors file in the repo.
    safetensors = [f for f in files if f.endswith(".safetensors")]
    if safetensors:
        return hf_hub_download(source, filename=safetensors[0])

    raise FileNotFoundError(
        f"No .safetensors file found in HF repo '{source}'"
    )




# Hardcoded VAE config for Wan 2.1/2.2 VAE loading.
# This eliminates the need to ship a diffusers model_index.json alongside
# the safetensors weight file and avoids the pickle-based torch.load() path
# that fails on safetensors files with
#   _pickle.UnpicklingError: invalid load key, '\x00'.
_WAN_VAE_CONFIG = {
    "_class_name": "AutoencoderKLWan",
    "_diffusers_version": "0.35.0.dev0",
    "_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "attn_scales": [],
    "base_dim": 96,
    "decoder_base_dim": None,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "in_channels": 3,
    "is_residual": False,
    "latents_mean": [
        -0.7571, -0.7089, -0.9113, 0.1075,
        -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632,
        -0.1922, -0.9497, 0.2503, -0.2921
    ],
    "latents_std": [
        2.8184, 1.4541, 2.3275, 2.6558,
        1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579,
        1.6382, 1.1253, 2.8251, 1.916
    ],
    "num_res_blocks": 2,
    "out_channels": 3,
    "patch_size": None,
    "scale_factor_spatial": 8,
    "scale_factor_temporal": 4,
    "temperal_downsample": [False, True, True],
    "z_dim": 16
}

class AggressiveWanUnloadPipeline(WanPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
            vae=vae,
            scheduler=scheduler,
        )
        self._exec_device = device
    @property
    def _execution_device(self):
        return self._exec_device
    
    def __call__(
        self: WanPipeline,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None],
                  PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # unload vae and transformer
        vae_device = self.vae.device
        transformer_device = self.transformer.device
        text_encoder_device = self.text_encoder.device
        device = self.transformer.device
        
        print("Unloading vae")
        self.vae.to("cpu")
        self.text_encoder.to(device)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # unload text encoder
        print("Unloading text encoder")
        self.text_encoder.to("cpu")

        self.transformer.to(device)

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(device, transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device, transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(device, transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * \
                        (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # unload transformer
        # load vae
        print("Loading Vae")
        self.vae.to(vae_device)

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


class Wan21(BaseModel):
    arch = 'wan21'
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = False
    _wan_vae_path = None
    
    _comfy_te_file = ['text_encoders/umt5_xxl_fp16.safetensors', 'text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors']
    def __init__(
            self,
            device,
            model_config: ModelConfig,
            dtype='bf16',
            custom_pipeline=None,
            noise_scheduler=None,
            **kwargs
    ):
        super().__init__(device, model_config, dtype,
                         custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ['WanTransformer3DModel']

        # cache for holding noise
        self.effective_noise = None
        
    def get_bucket_divisibility(self):
        return 16

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler(model_config=None):
        if model_config and getattr(model_config, 'train_scheduler', None):
            from toolkit.scheduler import build_noise_scheduler
            return build_noise_scheduler(model_config.train_scheduler)
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler
    
    def load_wan_transformer(self, transformer_path, subfolder=None):
        self.print_and_status_update("Loading transformer")
        dtype = self.torch_dtype
        transformer = WanTransformer3DModel.from_pretrained(
            transformer_path,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).to(dtype=dtype)

        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.1 models")

        if self.model_config.low_vram:
            # quantize on the device
            transformer.to('cpu', dtype=dtype)
            flush()
        else:
            transformer.to(self.device_torch, dtype=dtype)
            flush()

        if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
            raise ValueError(
                "Assistant LoRA is not supported for Wan2.1 models currently")

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for Wan2.1 models currently")

        flush()
        
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()
        
        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent
            )
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to('cpu')

        return transformer

    def load_model(self):
        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path

        self.print_and_status_update("Loading Wan model")
        subfolder = 'transformer'
        transformer_path = model_path
        if os.path.exists(transformer_path):
            # Check if this is a LoRA/custom path (contains .safetensors files directly)
            # If so, the path itself is the LoRA directory, not a parent of transformer/
            has_safetensors = any(f.endswith('.safetensors') for f in os.listdir(transformer_path) if os.path.isfile(os.path.join(transformer_path, f)))
            if has_safetensors:
                # LoRA/custom path - files are directly in this directory
                # Keep subfolder as 'transformer' so load_wan_transformer uses the path as-is
                pass  # subfolder remains 'transformer', transformer_path remains model_path
            else:
                # Standard transformer directory structure - transformer files are in transformer/ subfolder
                subfolder = None
                transformer_path = os.path.join(transformer_path, 'transformer')
        
        te_path = "ai-toolkit/umt5_xxl_encoder"
        if self.model_config.te_name_or_path is not None:
            # Check if it's a custom local path or repo
            te_path = self.model_config.te_name_or_path
            self.print_and_status_update(f"Using custom text encoder: {te_path}")
        elif os.path.exists(os.path.join(model_path, 'text_encoder')):
            te_path = model_path
        
        vae_path = self.model_config.extras_name_or_path
        if os.path.exists(os.path.join(model_path, 'vae')):
            vae_path = model_path

        transformer = self.load_wan_transformer(
            transformer_path,
            subfolder=subfolder,
        )

        flush()

        self.print_and_status_update("Loading UMT5EncoderModel")
        
        tokenizer, text_encoder = get_umt5_encoder(
            model_path=te_path,
            tokenizer_subfolder="tokenizer",
            encoder_subfolder="text_encoder",
            torch_dtype=dtype,
            comfy_files=self._comfy_te_file
        )

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing UMT5EncoderModel")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()
        
        if self.model_config.layer_offloading and self.model_config.layer_offloading_text_encoder_percent > 0:
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent
            )

        if self.model_config.low_vram:
            print("Moving transformer back to GPU")
            # we can move it back to the gpu now
            transformer.to(self.device_torch)

        scheduler = Wan21.get_train_scheduler()
        self.print_and_status_update("Loading VAE")

        # ---- Resolve VAE source path ----
        # Priority:
        #   1. `model_config.custom_vae_name_or_path` — user-specified custom
        #      VAE repo/path (e.g. a community fp32 VAE).
        #   2. Class-level `_wan_vae_path` — the default VAE repo for this
        #      Wan model family (e.g. "ai-toolkit/wan2.1-vae").
        #   3. `extras_name_or_path` / local `vae/` subfolder — the
        #      fallback used by non-wan2.1 models.
        vae_source = self.model_config.custom_vae_name_or_path
        if vae_source is None:
            vae_source = self._wan_vae_path
        if vae_source is None:
            vae_source = vae_path
            vae_subfolder = "vae"
        else:
            vae_subfolder = None

        # ---- Resolve VAE dtype ----
        # `model_config.vae_dtype` lets the user override the VAE precision
        # independently of the transformer dtype.  This is useful when a
        # custom fp32 VAE is provided (e.g. for improved first-frame encoding
        # quality in I2V).  Falls back to the model's torch_dtype.
        vae_dtype_str = self.model_config.vae_dtype
        vae_dtype = get_torch_dtype(vae_dtype_str)
        
        # Log VAE source and precision
        source_type = "local path" if os.path.exists(vae_source) else "HuggingFace repo"
        self.print_and_status_update(
            f"=== VAE Loading Configuration ===\n"
            f"  Source: {vae_source} ({source_type})\n"
            f"  Subfolder: {vae_subfolder or '(root)'}\n"
            f"  Target dtype: {vae_dtype} ({self.model_config.vae_dtype or 'model default'})\n"
            f"  Using hardcoded VAE config (_WAN_VAE_CONFIG)"
        )

        # Load the VAE by:
        #   1. Using the hardcoded VAE config (_WAN_VAE_CONFIG) so we do not
        #      rely on a diffusers model_index.json being present next to the
        #      weight file.
        #   2. Auto-detecting the weight file format (.safetensors, .pt, .pth,
        #      or .bin) and using the appropriate loader.
        #   3. Normalizing the state dict if the VAE uses an alternative
        #      tensor naming scheme, and applying it to the model.
        #
        # This approach works for both the official Wan VAE and custom fp32
        # VAEs (e.g. for improved first-frame encoding quality in I2V), and
        # handles both safetensors and PyTorch pickle formats.
        vae = AutoencoderKLWan.from_config(_WAN_VAE_CONFIG)
        self.print_and_status_update("  ✓ VAE model instantiated from hardcoded config")

        weight_file = _get_vae_safetensor_file(vae_source, vae_subfolder)
        
        # Log file discovery
        file_type = "safetensors" if weight_file.endswith(".safetensors") else "PyTorch"
        self.print_and_status_update(
            f"  ✓ Discovered weight file: {os.path.basename(weight_file)}\n"
            f"    Full path: {weight_file}\n"
            f"    Format: {file_type}"
        )
        
        # Auto-detect file format and use appropriate loader
        if weight_file.endswith(".safetensors"):
            self.print_and_status_update("  → Loading with safetensors library (safetensors.torch.load_file)")
            state_dict = load_safetensors(weight_file)
        elif weight_file.endswith((".pt", ".pth", ".bin")):
            self.print_and_status_update("  → Loading with PyTorch (torch.load, weights_only=False)")
            state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)
        else:
            # Fallback: try safetensors first, then torch.load
            ext = weight_file.split('.')[-1]
            self.print_and_status_update(
                f"  → Unknown extension (.{ext}), attempting safetensors loader first"
            )
            try:
                state_dict = load_safetensors(weight_file)
            except Exception as e:
                self.print_and_status_update(
                    f"  → Safetensors failed ({type(e).__name__}), falling back to torch.load"
                )
                state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)
        
        # Log state dict info
        num_tensors = len(state_dict)
        self.print_and_status_update(
            f"  ✓ Loaded state dict: {num_tensors} tensors"
        )
        
        if detect_alternative_vae_naming(state_dict):
            self.print_and_status_update(
                "  → Detected alternative VAE naming scheme — normalizing tensors"
            )
            state_dict = normalize_vae_state_dict(state_dict)
            self.print_and_status_update("  ✓ Tensor names normalized")
        
        vae.load_state_dict(state_dict)
        self.print_and_status_update("  ✓ State dict loaded into VAE model")

        vae = vae.to(dtype=vae_dtype)
        self.print_and_status_update(
            f"  ✓ VAE moved to dtype: {vae_dtype}"
        )
        flush()

        self.print_and_status_update("Making pipe")
        pipe: WanPipeline = WanPipeline(
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer

        pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        text_encoder.to(self.device_torch)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()
        self.pipeline = pipe
        self.model = transformer
        self.vae = vae
        self.vae.enable_tiling()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def get_generation_pipeline(self, sampling_flow_shift: float = None):
        # Use the training scheduler. For Wan2.2 models (5b/14b), the subclass
        # overrides this to use a separate sampling scheduler with configurable shift.
        # During training with timestep_type='sigmoid', the shift is ignored.
        # During sampling, set_timesteps() reads self.config.shift.
        if self.model_config.sampling_scheduler:
            from toolkit.scheduler import build_noise_scheduler
            scheduler = build_noise_scheduler(self.model_config.sampling_scheduler)
        else:
            scheduler = self.get_train_scheduler(self.model_config)
        if self.model_config.low_vram:
            pipeline = AggressiveWanUnloadPipeline(
                vae=self.vae,
                transformer=self.model,
                transformer_2=self.model,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=scheduler,
                expand_timesteps=self._wan_expand_timesteps,
                device=self.device_torch
            )
        else:
            pipeline = WanPipeline(
                vae=self.vae,
                transformer=self.unet,
                transformer_2=self.unet,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                expand_timesteps=self._wan_expand_timesteps,
                scheduler=scheduler,
            )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    @property
    def use_vae_tiling(self):
        # tile the vae decode when sampling if in low vram or explicitly enabled
        return self.model_config.low_vram or self.model_config.model_kwargs.get(
            "vae_tiling", False
        )

    def generate_single_image(
        self,
        pipeline: WanPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        pipeline = pipeline.to(self.device_torch)

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

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs
    ):
        # vae_scale_factor_spatial = 8
        # vae_scale_factor_temporal = 4
        # num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        # shape = (
        #     batch_size,
        #     num_channels_latents, # 16
        #     num_latent_frames,  # 81
        #     int(height) // self.vae_scale_factor_spatial,
        #     int(width) // self.vae_scale_factor_spatial,
        # )

        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=512,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )
        return PromptEmbeds(prompt_embeds)

    @torch.no_grad()
    def encode_images(
            self,
            image_list: List[torch.Tensor],
            device=None,
            dtype=None
    ):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            # Return latents in the training dtype by default (e.g., bf16)
            # Encodes still occur in the VAE's own dtype for correctness.
            dtype = self.torch_dtype

        if self.vae.device == torch.device('cpu'):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # Encode with VAE's native dtype, then convert latents to the desired output dtype
        vae_dtype = self.vae.dtype
        image_list = [image.to(device, dtype=vae_dtype) for image in image_list]

        # Normalize shapes
        norm_images = []
        for image in image_list:
            if image.ndim == 3:
                # (C, H, W) -> (C, 1, H, W)
                norm_images.append(image.unsqueeze(1))
            elif image.ndim == 4:
                # (T, C, H, W) -> (C, T, H, W)
                norm_images.append(image.permute(1, 0, 2, 3))
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

        # Stack to (B, C, T, H, W)
        images = torch.stack(norm_images)
        B, C, T, H, W = images.shape

        # Resize if needed (B * T, C, H, W)
        if H % 8 != 0 or W % 8 != 0:
            target_h = H // 8 * 8
            target_w = W // 8 * 8
            images = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            images = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
            images = images.view(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)

        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std

        return latents.to(device, dtype=dtype)
    
    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device('cpu'):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents * latents_std + latents_mean

        images = self.vae.decode(latents).sample

        return images.to(device, dtype=dtype)

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        # only save the unet
        transformer: Wan21 = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, 'transformer'),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, 'aitk_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get('noise')
        batch = kwargs.get('batch')
        if batch is None:
            raise ValueError("Batch is not provided")
        if noise is None:
            raise ValueError("Noise is not provided")
        return (noise - batch.latents).detach()

    def convert_lora_weights_before_save(self, state_dict):
        return convert_to_original(state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        return convert_to_diffusers(state_dict)
    
    def get_base_model_version(self):
        return "wan_2.1"
    
    def get_transformer_block_names(self):
        return ['blocks']
