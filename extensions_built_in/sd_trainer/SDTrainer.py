import os
import random
from collections import OrderedDict
from typing import Union, Literal, List, Optional, Dict

import numpy as np
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel

import torch.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, ConcatDataset

from toolkit import train_tools
from toolkit.basic import value_map, adain, get_mean_std
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GenerateImageConfig
from toolkit.data_loader import get_dataloader_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss, GuidanceType
from toolkit.image_utils import show_tensors, show_latents
from toolkit.ip_adapter import IPAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.memory_management import sync_grad_transfers
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
    apply_learnable_snr_gos, LearnableSNRGamma
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms
from diffusers import EMAModel
import math
from toolkit.train_tools import precondition_model_outputs_flow_match
from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor, load_dfe
from toolkit.util.losses import wavelet_loss, stepped_loss, spectral_loss, spectral_flow_loss, mse_spectral_flow_loss
import torch.nn.functional as F
from toolkit.unloader import unload_text_encoder
from PIL import Image
from torchvision.transforms import functional as TF
from toolkit.basic import flush
from toolkit.rank_gates import FisherTracker, QuenchSchedule, update_rank_gates, apply_hardening_interpolation, finalize_gates, log_gate_stats


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False
        self.taesd: Optional[AutoencoderTiny] = None

        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None

        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"
        
        # Disable donated buffers when using gradient projection with compiled models.
        # PyTorch's compiler uses donated buffers for memory optimization, but
        # retain_graph=True in backward() conflicts with this optimization.
        if (self.train_config.spectral_flow_gradient_projection_enabled or
            self.train_config.mse_spectral_flow_gradient_projection_enabled):
            torch._functorch.config.donated_buffer = False
        
        # Rank gate annealing (SparseForge-inspired)
        self.rank_gates_scheduler: Optional[QuenchSchedule] = None
        self.fisher_tracker: Optional[FisherTracker] = None
        self.rank_gates_is_per_expert_training: bool = False

        self.do_grad_scale = True
        if self.is_fine_tuning and self.is_bfloat:
            self.do_grad_scale = False
        if self.adapter_config is not None:
            if self.adapter_config.train:
                self.do_grad_scale = False

        # if self.train_config.dtype in ["fp16", "float16"]:
        #     # patch the scaler to allow fp16 training
        #     org_unscale_grads = self.scaler._unscale_grads_
        #     def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        #         return org_unscale_grads(optimizer, inv_scale, found_inf, True)
        #     self.scaler._unscale_grads_ = _unscale_grads_replacer

        self.cached_blank_embeds: Optional[PromptEmbeds] = None
        self.cached_trigger_embeds: Optional[PromptEmbeds] = None
        self.diff_output_preservation_embeds: Optional[PromptEmbeds] = None
        
        self.dfe: Optional[DiffusionFeatureExtractor] = None
        self.unconditional_embeds = None

        # Spectral flow loss state
        self.flow_loss_module = None
        # Per-expert spectral flow state (Issue #1 fix)
        self.flow_deviation_history = {}      # {expert_label: [...]}
        self.current_flow_weight = {}         # {expert_label: float}
        self.flow_rejection_count = {}        # {expert_label: int}
        # Step loss rejection state
        self.prev_expert_loss = {}            # {expert_label: float} - previous step loss per expert
        self.prev_expert_spatial_loss = {}    # {expert_label: float} - previous step spectral loss
        self.prev_expert_flow_loss = {}       # {expert_label: float} - previous step flow loss
        self.prev_expert_mse_loss = {}        # {expert_label: float} - previous step MSE loss
        self.step_rejection_count = {}        # {expert_label: int} - cumulative step rejections
        self.current_step_expert_loss = {}    # {expert_label: float} - accumulated loss for current step
        self.current_step_expert_spatial = {} # {expert_label: float} - accumulated spectral loss
        self.current_step_expert_flow = {}    # {expert_label: float} - accumulated flow loss
        self.current_step_expert_mse = {}     # {expert_label: float} - accumulated MSE loss
        # Gradient projection state
        self.gradient_projection_stats = {
            'total_conflicts': 0,      # cumulative conflicts detected
            'total_projections': 0,    # cumulative projections applied
            'step_conflicts': 0,       # conflicts in current step (reset each step)
            'step_projections': 0      # projections in current step (reset each step)
        }
        # Temporary storage for gradient projection
        self._mse_loss_tensor = None       # Loss tensor for MSE component (for gradient projection)
        self._spectral_loss_tensor = None  # Loss tensor for spectral component (for gradient projection)
        self._flow_loss_tensor = None      # Loss tensor for flow component (for gradient projection)
        # Running EMA of the per-batch flow gate mean (E[gate]) over the timestep
        # distribution, logged for monitoring the effective flow dilution factor.
        # 0.0 = not yet warmed up.
        self._flow_gate_ema = 0.0
        
        if self.train_config.diff_output_preservation:
            if self.trigger_word is None:
                raise ValueError("diff_output_preservation requires a trigger_word to be set")
            if self.network_config is None:
                raise ValueError("diff_output_preservation requires a network to be set")
            if self.train_config.train_text_encoder:
                raise ValueError("diff_output_preservation is not supported with train_text_encoder")
        
        if self.train_config.blank_prompt_preservation:
            if self.network_config is None:
                raise ValueError("blank_prompt_preservation requires a network to be set")
        
        if self.train_config.blank_prompt_preservation or self.train_config.diff_output_preservation:
            # always do a prior prediction when doing output preservation
            self.do_prior_prediction = True
        
        # store the loss target for a batch so we can use it in a loss
        self._guidance_loss_target_batch: float = 0.0
        if isinstance(self.train_config.guidance_loss_target, (int, float)):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target)
        elif isinstance(self.train_config.guidance_loss_target, list):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target[0])
        else:
            raise ValueError(f"Unknown guidance loss target type {type(self.train_config.guidance_loss_target)}")


    def before_model_load(self):
        pass

    def _calculate_grad_norm(self, params):
        if params is None or len(params) == 0:
            return None

        if isinstance(params[0], dict):
            param_iterable = (p for group in params for p in group.get('params', []))
        else:
            param_iterable = params

        total_norm_sq = None
        for param in param_iterable:
            if param is None:
                continue
            grad = getattr(param, 'grad', None)
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce()._values()
            grad_norm = grad.detach().float().norm(2)
            if total_norm_sq is None:
                total_norm_sq = grad_norm.pow(2)
            else:
                total_norm_sq = total_norm_sq + grad_norm.pow(2)

        if total_norm_sq is None:
            return None

        return total_norm_sq.sqrt()

    def _freeze_inactive_expert_loras(self, active_experts):
        """
        Freeze inactive expert LoRA params before optimizer.step().

        For Wan 2.2 multistage models, only the active expert's LoRAs should
        receive gradient updates and weight decay. This sets requires_grad=False
        on all params of inactive experts so they are completely ignored by the
        optimizer (AdamW, Adafactor, Automagic3, etc.).

        Args:
            active_experts: set of active expert IDs (e.g. {"transformer_1"}),
                           or None if not a multistage model.

        Returns:
            List of (LoRA_module, frozendict_of_params) tuples for unfreezing.
        """
        if active_experts is None:
            return []

        network = getattr(self, 'network', None)
        if network is None:
            return []

        frozen = []

        # Check transformer_1 expert
        e1_loras = getattr(network, 'unet_loras_expert1', [])
        if e1_loras and "transformer_1" not in active_experts:
            for lora in e1_loras:
                lora.requires_grad_(False)
                frozen.append((lora, None))

        # Check transformer_2 expert
        e2_loras = getattr(network, 'unet_loras_expert2', [])
        if e2_loras and "transformer_2" not in active_experts:
            for lora in e2_loras:
                lora.requires_grad_(False)
                frozen.append((lora, None))

        return frozen

    def _unfreeze_inactive_expert_loras(self, frozen_loras):
        """
        Unfreeze previously frozen inactive expert LoRA params after optimizer.step().

        Args:
            frozen_loras: List of (LoRA_module, ...) returned by _freeze_inactive_expert_loras.
        """
        for lora, _ in frozen_loras:
            lora.requires_grad_(True)

    def cache_sample_prompts(self):
        if self.train_config.disable_sampling:
            return
        if self.sample_config is not None and self.sample_config.samples is not None and len(self.sample_config.samples) > 0:
            # cache all the samples
            self.sd.sample_prompts_cache = []
            sample_folder = os.path.join(self.save_root, 'samples')
            output_path = os.path.join(sample_folder, 'test.jpg')
            for i in range(len(self.sample_config.prompts)):
                sample_item = self.sample_config.samples[i]
                prompt = self.sample_config.prompts[i]

                # needed so we can autoparse the prompt to handle flags
                gen_img_config = GenerateImageConfig(
                    prompt=prompt, # it will autoparse the prompt
                    negative_prompt=sample_item.neg,
                    output_path=output_path,
                    ctrl_img=sample_item.ctrl_img,
                    ctrl_img_1=sample_item.ctrl_img_1,
                    ctrl_img_2=sample_item.ctrl_img_2,
                    ctrl_img_3=sample_item.ctrl_img_3,
                )
                
                has_control_images = False
                if gen_img_config.ctrl_img is not None or gen_img_config.ctrl_img_1 is not None or gen_img_config.ctrl_img_2 is not None or gen_img_config.ctrl_img_3 is not None:
                    has_control_images = True
                # see if we need to encode the control images
                if self.sd.encode_control_in_text_embeddings and has_control_images:
                    
                    ctrl_img_list = []
                    
                    if gen_img_config.ctrl_img is not None:
                        ctrl_img = Image.open(gen_img_config.ctrl_img).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img = (
                            TF.to_tensor(ctrl_img)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img)
                    
                    if gen_img_config.ctrl_img_1 is not None:
                        ctrl_img_1 = Image.open(gen_img_config.ctrl_img_1).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_1 = (
                            TF.to_tensor(ctrl_img_1)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_1)
                    if gen_img_config.ctrl_img_2 is not None:
                        ctrl_img_2 = Image.open(gen_img_config.ctrl_img_2).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_2 = (
                            TF.to_tensor(ctrl_img_2)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_2)
                    if gen_img_config.ctrl_img_3 is not None:
                        ctrl_img_3 = Image.open(gen_img_config.ctrl_img_3).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_3 = (
                            TF.to_tensor(ctrl_img_3)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_3)
                    
                    if self.sd.has_multiple_control_images:
                        ctrl_img = ctrl_img_list
                    else:
                        ctrl_img = ctrl_img_list[0] if len(ctrl_img_list) > 0 else None
                    
                    
                    positive = self.sd.encode_prompt(
                        gen_img_config.prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                    negative = self.sd.encode_prompt(
                        gen_img_config.negative_prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                else:
                    positive = self.sd.encode_prompt(gen_img_config.prompt).to('cpu')
                    negative = self.sd.encode_prompt(gen_img_config.negative_prompt).to('cpu')
                
                self.sd.sample_prompts_cache.append({
                    'conditional': positive,
                    'unconditional': negative
                })
        

    def before_dataset_load(self):
        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            if self.train_config.adapter_assist_type == "t2i":
                # dont name this adapter since we are not training it
                self.assistant_adapter = T2IAdapter.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch)
            elif self.train_config.adapter_assist_type == "control_net":
                self.assistant_adapter = ControlNetModel.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
            else:
                raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")

            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()
        if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
            if self.model_config.is_xl:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            else:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
            self.taesd.eval()
            self.taesd.requires_grad_(False)

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        
        # Initialize rank gate annealing (SparseForge-inspired)
        if (self.network is not None and 
            hasattr(self.network, 'gated_loras') and 
            self.network.gated_loras and
            self.network_config is not None and
            self.network_config.rank_gates is not None and
            self.network_config.rank_gates.enabled):
            
            rg = self.network_config.rank_gates
            total_steps = self.train_config.steps
            
            # Determine effective steps per expert for auto-timing.
            # In per-expert training (Wan 2.2 14B I2V with both experts active),
            # each expert only sees ~half the batches. Auto-calculated annealing
            # timing (5% start, 75% end) should be based on per-expert steps,
            # not total steps, so each expert's gates anneal over their own
            # effective training duration.
            #
            # IMPORTANT: User-provided start_step/end_step are interpreted as GLOBAL
            # steps (not per-expert). This matches user intuition: "start annealing
            # at step 1000" means global step 1000, regardless of expert count.
            
            self.rank_gates_is_per_expert_training = False
            
            # Check if we're in per-expert training mode with both experts active
            if (hasattr(self.network, 'unet_loras_expert1') and 
                hasattr(self.network, 'unet_loras_expert2')):
                expert1_count = len(getattr(self.network, 'unet_loras_expert1', []))
                expert2_count = len(getattr(self.network, 'unet_loras_expert2', []))
                if expert1_count > 0 and expert2_count > 0:
                    # Both experts are being trained - each sees ~half the steps
                    self.rank_gates_is_per_expert_training = True
                    print(f"\n[SDTrainer] Per-expert training detected ({expert1_count} LoRAs x2). "
                          f"Each expert sees ~{total_steps // 2} effective steps "
                          f"(total: {total_steps}). Annealing timing uses global steps.")
            
            # Scale hardening_window relative to total steps.
            # Default 500 is reasonable for large runs but would be too large for
            # small jobs. Cap at 5% of total steps.
            effective_hardening_window = rg.hardening_window
            max_hardening_window = max(50, int(total_steps * 0.05))
            if effective_hardening_window > max_hardening_window:
                effective_hardening_window = max_hardening_window
                print(f"  [SDTrainer] Scaled hardening_window: {rg.hardening_window} → {effective_hardening_window} "
                      f"(5% of {total_steps} total steps)")
            
            # Resolve annealing timing based on TOTAL (global) steps.
            # User-provided values are global steps; auto values are percentages of total.
            start_step = rg.start_step
            end_step = rg.end_step
            if start_step is None:
                start_step = max(100, int(total_steps * 0.05))
            if end_step is None:
                end_step = min(total_steps - effective_hardening_window, int(total_steps * 0.75))
            
            # QuenchSchedule uses GLOBAL steps. In per-expert training, each expert's
            # gates are driven by the global schedule but only receive ~1/N of the
            # EMA updates and gate updates. This is intentional: it keeps the code
            # simple and ensures both experts reach their final state by total_steps.
            # Temperature decay is tracked by actual update count (per expert) to
            # avoid decaying too fast when experts alternate.
            self.rank_gates_scheduler = QuenchSchedule(
                total_steps=total_steps,
                start_step=start_step,
                end_step=end_step,
                target_rank_ratio=rg.target_rank_ratio,
                temperature=rg.temperature,
                gamma=rg.gamma,
                alpha=rg.alpha,
                lambda_mid_max=rg.lambda_mid_max,
                update_every=rg.update_every,
                hardening_window=effective_hardening_window,
                eta_pen=rg.eta_pen,
            )
            
            self.fisher_tracker = FisherTracker(
                decay=rg.fisher_decay,
                use_first_order=rg.use_first_order,
            )
            
            # Track per-expert update counts for correct temperature decay.
            # In per-expert training, each expert only gets updated ~half as often,
            # so temperature should decay based on actual updates received, not
            # global step count.
            self.rank_gates_expert_update_counts: Dict[str, int] = {}
            
            timing_note = f" (per-expert)" if self.rank_gates_is_per_expert_training else ""
            print(f"\n[SDTrainer] Rank gate annealing initialized{timing_note}:")
            print(f"  Start: global step {start_step}, End: global step {end_step}")
            print(f"  Hardening window: global steps {self.rank_gates_scheduler.hardening_start}-{self.rank_gates_scheduler.hardening_end}")
            print(f"  Target rank ratio: {rg.target_rank_ratio}")
            print(f"  Total gated ranks: {sum(gl.r for gl in self.network.gated_loras)}")
        else:
            # Rank gates not enabled or no gated LoRAs
            self.rank_gates_scheduler = None
            self.fisher_tracker = None
        
        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to('cpu')
        
        # cache unconditional embeds (blank prompt)
        with torch.no_grad():
            kwargs = {}
            if self.sd.encode_control_in_text_embeddings:
                # just do a blank image for unconditionals
                control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                if self.sd.has_multiple_control_images:
                    control_image = [control_image]
                
                kwargs['control_images'] = control_image
            self.unconditional_embeds = self.sd.encode_prompt(
                [self.train_config.unconditional_prompt],
                long_prompts=self.do_long_prompts,
                **kwargs
            ).to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.do_prior_divergence:
            self.do_prior_prediction = True
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)
        if self.adapter is not None:
            self.adapter.to(self.device_torch)

            # check if we have regs and using adapter and caching clip embeddings
            has_reg = self.datasets_reg is not None and len(self.datasets_reg) > 0
            is_caching_clip_embeddings = self.datasets is not None and any([self.datasets[i].cache_clip_vision_to_disk for i in range(len(self.datasets))])

            if has_reg and is_caching_clip_embeddings:
                # we need a list of unconditional clip image embeds from other datasets to handle regs
                unconditional_clip_image_embeds = []
                datasets = get_dataloader_datasets(self.data_loader)
                for i in range(len(datasets)):
                    unconditional_clip_image_embeds += datasets[i].clip_vision_unconditional_cache

                if len(unconditional_clip_image_embeds) == 0:
                    raise ValueError("No unconditional clip image embeds found. This should not happen")

                self._clip_image_embeds_unconditional = unconditional_clip_image_embeds

        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    # remove empty
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]
            else:
                # single prompt
                self.negative_prompt_pool = [self.train_config.negative_prompt]

        # handle unload text encoder
        if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
            print_acc("Caching embeddings and unloading text encoder")
            with torch.no_grad():
                if self.train_config.train_text_encoder:
                    raise ValueError("Cannot unload text encoder if training text encoder")
                # cache embeddings
                self.sd.text_encoder_to(self.device_torch)
                encode_kwargs = {}
                if self.sd.encode_control_in_text_embeddings:
                    # just do a blank image for unconditionals
                    control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.sd.has_multiple_control_images:
                        control_image = [control_image]
                    encode_kwargs['control_images'] = control_image
                self.cached_blank_embeds = self.sd.encode_prompt("", **encode_kwargs)
                if self.trigger_word is not None:
                    self.cached_trigger_embeds = self.sd.encode_prompt(self.trigger_word, **encode_kwargs)
                if self.train_config.diff_output_preservation:
                    self.diff_output_preservation_embeds = self.sd.encode_prompt(self.train_config.diff_output_preservation_class)
                
                self.cache_sample_prompts()
                
                print_acc("\n***** UNLOADING TEXT ENCODER *****")
                if self.is_caching_text_embeddings:
                    print_acc("Embeddings cached to disk. We dont need the text encoder anymore")
                else:
                    print_acc("This will train only with a blank prompt or trigger word, if set")
                    print_acc("If this is not what you want, remove the unload_text_encoder flag")
                print_acc("***********************************")
                print_acc("")

                # unload the text encoder
                if self.is_caching_text_embeddings:
                    unload_text_encoder(self.sd)
                else:
                    # todo once every model is tested to work, unload properly. Though, this will all be merged into one thing.
                    # keep legacy usage for now. 
                    self.sd.text_encoder_to("cpu")
                flush()
        
        if self.train_config.blank_prompt_preservation and self.cached_blank_embeds is None:
            # make sure we have this if not unloading
            self.cached_blank_embeds = self.sd.encode_prompt("").to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.diffusion_feature_extractor_path is not None:
            vae = self.sd.vae
            # if not (self.model_config.arch in ["flux"]) or self.sd.vae.__class__.__name__ == "AutoencoderPixelMixer":
            #     vae = self.sd.vae
            self.dfe = load_dfe(
                self.train_config.diffusion_feature_extractor_path, 
                vae=vae,
                sd=self.sd
            )
            self.dfe.to(self.device_torch)
            if hasattr(self.dfe, 'vision_encoder') and self.train_config.gradient_checkpointing:
                # must be set to train for gradient checkpointing to work
                self.dfe.vision_encoder.train()
                self.dfe.vision_encoder.gradient_checkpointing = True
            elif hasattr(self.dfe, 'model') and self.train_config.gradient_checkpointing:
                if hasattr(self.dfe.model, 'enable_gradient_checkpointing'): 
                    self.dfe.model.train()
                    self.dfe.model.enable_gradient_checkpointing()
                if hasattr(self.dfe.model, 'gradient_checkpointing_enable'): 
                    self.dfe.model.train()
                    self.dfe.model.gradient_checkpointing_enable()
                elif hasattr(self.dfe.model, 'gradient_checkpointing'):
                    self.dfe.model.train()
                    self.dfe.model.gradient_checkpointing = True
                else:
                    print_acc("Warning: Could not enable gradient checkpointing on diffusion feature extractor model.")
            else:
                self.dfe.eval()
                
            # enable gradient checkpointing on the vae
            if vae is not None and self.train_config.gradient_checkpointing:
                try:
                    vae.enable_gradient_checkpointing()
                    vae.train()
                except:
                    pass

        # Initialize spectral flow loss module if using spectral_flow loss type
        if self.train_config.loss_type == 'spectral_flow':
            from toolkit.optical_flow.flow_loss import load_flow_loss
            self.flow_loss_module = load_flow_loss(self.sd).to(self.device_torch, dtype=self.sd.torch_dtype)
            # FlowConsistencyLoss has no learnable parameters (static helpers only),
            # so .eval()/.parameters() are unnecessary but harmless
            print_acc(f"Spectral+Flow loss enabled. Base flow weight: {self.train_config.spectral_flow_weight}, "
                      f"max_timestep: {self.train_config.spectral_flow_max_timestep}")
            # TODO: SEA-RAFT vendored impl (sea_raft_impl/model.py) has different CorrBlock/AltCorrBlock
            # tensor layouts than the official SEA-RAFT. Loading official weights with strict=False will
            # silently drop or misalign keys, producing potentially inaccurate flow estimates.
            # Needs additional research against official SEA-RAFT implementation; fix deferred.

        # Initialize attention tanh softcapping (Gemma2/Grok-1 style)
        # Prevents attention scores from becoming too extreme, improving training stability
        self._setup_attention_softcapping()

    def _setup_attention_softcapping(self):
        """Set up attention tanh softcapping, F32 RoPE acceleration, and GELU acceleration."""
        try:
            from toolkit.models.wan21.wan_attn import (
                set_attention_softcapping, set_attention_f32_rope, configure_softcap_logging,
                enable_gelu_acceleration, is_gelu_acceleration_enabled, update_softcap_step,
                set_attention_backend_choice, get_effective_backend,
                check_flash_softcap_support
            )
            from toolkit.util.attention_softcapping import check_flex_attention_support

            # Tanh softcapping config
            softcap_enabled = getattr(self.train_config, 'attention_tanh_softcap_enabled', True)
            soft_cap = getattr(self.train_config, 'attention_tanh_softcap_value', 30.0)
            
            # Per-type overrides
            soft_cap_self = getattr(self.train_config, 'attention_tanh_softcap_value_self_attn', None)
            soft_cap_cross = getattr(self.train_config, 'attention_tanh_softcap_value_cross_attn', None)
            
            # Per-expert overrides
            soft_cap_high = getattr(self.train_config, 'attention_tanh_softcap_value_high_noise', None)
            soft_cap_low = getattr(self.train_config, 'attention_tanh_softcap_value_low_noise', None)
            
            # Per-type-per-expert overrides
            soft_cap_self_high = getattr(self.train_config, 'attention_tanh_softcap_value_self_attn_high_noise', None)
            soft_cap_self_low = getattr(self.train_config, 'attention_tanh_softcap_value_self_attn_low_noise', None)
            soft_cap_cross_high = getattr(self.train_config, 'attention_tanh_softcap_value_cross_attn_high_noise', None)
            soft_cap_cross_low = getattr(self.train_config, 'attention_tanh_softcap_value_cross_attn_low_noise', None)

            # Sampling softcapping is an independent toggle (off by default to
            # match standard inference). Uses the same soft_cap/overrides.
            sample_softcap_enabled = getattr(
                getattr(self, 'sample_config', None), 'attention_tanh_softcap_enabled', False)

            # Attention backend selection (separate for training and sampling).
            # train.attention_backend / sample.attention_backend:
            #   native (default) | flex | sdpa | flash
            # Read BEFORE the softcap gate so we can decide, per mode, whether the
            # selected kernel can actually apply softcapping (flex via score_mod,
            # or flash natively in 2.8.3+).
            train_backend = getattr(self.train_config, 'attention_backend', 'native')
            sample_backend = getattr(getattr(self, 'sample_config', None), 'attention_backend', 'native')
            train_backend_lc = str(train_backend).lower()
            sample_backend_lc = str(sample_backend).lower()

            if softcap_enabled or sample_softcap_enabled:
                # Probe kernel capabilities once (cheap; cached at the call sites too).
                flex_ok = check_flex_attention_support().get('available', False)
                flash_ok = check_flash_softcap_support()

                def _mode_softcap_ok(enabled, backend_lc):
                    # A mode can apply softcap if its selected kernel can do it:
                    #   flash          -> natively (2.8.3+) or defer to flex
                    #   auto/sdpa/flex -> flex score_mod
                    if not enabled:
                        return True
                    if backend_lc == 'flash':
                        return flash_ok or flex_ok
                    return flex_ok

                train_softcap_ok = _mode_softcap_ok(softcap_enabled, train_backend_lc)
                sample_softcap_ok = _mode_softcap_ok(sample_softcap_enabled, sample_backend_lc)

                if train_softcap_ok or sample_softcap_ok:
                    set_attention_softcapping(
                        enabled=train_softcap_ok,
                        soft_cap=soft_cap,
                        sample_enabled=sample_softcap_ok,
                        soft_cap_self_attn=soft_cap_self,
                        soft_cap_cross_attn=soft_cap_cross,
                        soft_cap_high_noise=soft_cap_high,
                        soft_cap_low_noise=soft_cap_low,
                        soft_cap_self_attn_high_noise=soft_cap_self_high,
                        soft_cap_self_attn_low_noise=soft_cap_self_low,
                        soft_cap_cross_attn_high_noise=soft_cap_cross_high,
                        soft_cap_cross_attn_low_noise=soft_cap_cross_low,
                    )
                    # Enable logging - sample every 10 training steps (not attention ops)
                    configure_softcap_logging(enabled=True, sample_every_n_steps=10)
                    print_acc(f"Attention tanh softcapping -> training: {'ON' if train_softcap_ok else 'off'}, "
                              f"sampling: {'ON' if sample_softcap_ok else 'off'}. "
                              f"flex_attention available: {flex_ok}, flash native softcap: {flash_ok}.")
                    print_acc("Softcapping is applied by the selected kernel: flash natively (2.8.3+) "
                              "or flex via score_mod (auto/sdpa/flex). fp32 layers skip it under flash.")
                    if softcap_enabled and not train_softcap_ok:
                        print_acc(f"  NOTE: training softcapping disabled - train backend '{train_backend}' "
                                  f"cannot apply it (flex unavailable and flash lacks native softcap).")
                    if sample_softcap_enabled and not sample_softcap_ok:
                        print_acc(f"  NOTE: sampling softcapping disabled - sample backend '{sample_backend}' "
                                  f"cannot apply it (flex unavailable and flash lacks native softcap).")

                    # Print active softcap configuration summary
                    has_overrides = any(v is not None for v in [
                        soft_cap_self, soft_cap_cross, soft_cap_high, soft_cap_low,
                        soft_cap_self_high, soft_cap_self_low, soft_cap_cross_high, soft_cap_cross_low
                    ])
                    
                    if has_overrides:
                        from toolkit.models.wan21.wan_attn import resolve_softcap_value
                        print_acc(f"  Per-type-per-expert softcap values (effective):")
                        for expert in ['single', 'high_noise', 'low_noise']:
                            self_val = resolve_softcap_value('self_attn', expert)
                            cross_val = resolve_softcap_value('cross_attn', expert)
                            expert_label = expert.replace('_noise', '')
                            print_acc(f"    {expert_label:8s} expert: self_attn={self_val:.0f}, cross_attn={cross_val:.0f}")
                    else:
                        print_acc(f"  Global soft_cap={soft_cap} for all attention types and experts.")
                    
                    print_acc(f"Softcapping stats logging enabled - will log per-type per-expert statistics every 10 steps. "
                              f"(Works with torch.compile enabled; sampled via a graph break.)")
                else:
                    print_acc("Attention tanh softcapping requested but no selected attention backend "
                              "can apply it (flex_attention unavailable and flash_attn lacks native "
                              "softcapping). Using standard attention.")
                    set_attention_softcapping(enabled=False, sample_enabled=False)
                    configure_softcap_logging(enabled=False)
            else:
                set_attention_softcapping(enabled=False, sample_enabled=False)
                configure_softcap_logging(enabled=False)
                print_acc("Attention tanh softcapping disabled.")

            try:
                set_attention_backend_choice(train_backend=train_backend, sample_backend=sample_backend)
            except ValueError as e:
                print_acc(f"Warning: {e} Using 'auto' for both.")
                set_attention_backend_choice(train_backend='auto', sample_backend='auto')
            print_acc(f"Attention backend -> training: {get_effective_backend(in_sampling=False)} "
                      f"(configured: '{train_backend}'), sampling: {get_effective_backend(in_sampling=True)} "
                      f"(configured: '{sample_backend}').")

            # F32 RoPE acceleration config
            f32_rope_enabled = getattr(self.train_config, 'attention_f32_rope_enabled', True)
            set_attention_f32_rope(enabled=f32_rope_enabled)
            rope_dtype = "float32" if f32_rope_enabled else "float64"
            print_acc(f"Attention RoPE computation using {rope_dtype}. "
                      f"F32 is faster than F64 while maintaining numerical stability.")

            # GELU acceleration for Wan 2.2 FeedForward layers (separate config flag)
            # Wan uses gelu-approximate in all FF layers - patch to use tanh.approx.f32
            # NOTE: This is a global monkeypatch - only enable if training Wan 2.x models
            gelu_accel_enabled = getattr(self.train_config, 'gelu_acceleration_enabled', True)
            if gelu_accel_enabled:
                if enable_gelu_acceleration():
                    print_acc("GELU acceleration enabled - using tanh.approx.f32 PTX instruction for FF layers.")
                else:
                    print_acc("GELU acceleration requested but not available - using standard PyTorch GELU.")
            else:
                print_acc("GELU acceleration disabled (set gelu_acceleration_enabled=True to enable for Wan 2.x).")

        except ImportError as e:
            print_acc(f"Attention config setup failed: {e}")

    def _log_attention_stats(self):
        """Log attention softcapping statistics with per-type breakdown."""
        try:
            from toolkit.models.wan21.wan_attn import get_softcap_stats

            stats = get_softcap_stats()
            if not stats['enabled'] or not stats['sample_stats']:
                return

            # Use grouped stats for per-type breakdown
            grouped = stats.get('grouped', {})
            step = stats.get('current_step', 'N/A')
            expert = stats.get('current_expert', 'single')

            print_acc(f"\n{'='*70}")
            print_acc(f"Attention Softcapping Stats (step {step}, expert: {expert})")
            if stats.get('fallback_count'):
                print_acc(f"WARNING: flex_attention fell back to SDPA {stats['fallback_count']} time(s) "
                          f"- softcapping was skipped on those calls.")
            print_acc(f"{'='*70}")

            # Log per attention type
            for attn_type in ['self_attn', 'cross_attn']:
                type_label = "Self-Attention (attn1: video→video)" if attn_type == 'self_attn' else "Cross-Attention (attn2: video→text/img)"
                type_stats = grouped.get(attn_type, {})
                
                # Find stats for current expert, or any if none for current expert
                expert_samples = type_stats.get(expert, [])
                if not expert_samples:
                    # Fall back to any expert's samples for this type
                    for e in ['single', 'high', 'low']:
                        if type_stats.get(e):
                            expert_samples = type_stats[e]
                            break
                
                if not expert_samples:
                    print_acc(f"\n  {type_label}: no samples")
                    continue
                
                # Aggregate from last few samples for stability
                samples = expert_samples[-3:]  # Last 3 samples
                avg_pct_capped = sum(s.get('pct_capped', 0) for s in samples) / len(samples)
                avg_reduction = sum(s.get('max_reduction_pct', 0) for s in samples) / len(samples)
                avg_lse = sum(s.get('avg_lse', 0) for s in samples) / len(samples)
                raw_min = min(s.get('raw_min', 0) for s in samples)
                raw_max = max(s.get('raw_max', 0) for s in samples)
                capped_min = min(s.get('capped_min', 0) for s in samples)
                capped_max = max(s.get('capped_max', 0) for s in samples)
                soft_cap_used = samples[-1].get('soft_cap_used', 30.0)
                
                print_acc(f"\n  {type_label} [soft_cap={soft_cap_used:.0f}]:")
                print_acc(f"    Scores capped:      {avg_pct_capped:.2f}%")
                print_acc(f"    Max reduction:      {avg_reduction:.1f}%")
                print_acc(f"    Attention sharpness (LSE): {avg_lse:.2f} (lower=softer)")
                print_acc(f"    Raw score range:    [{raw_min:.2f}, {raw_max:.2f}]")
                print_acc(f"    Capped score range: [{capped_min:.2f}, {capped_max:.2f}]")
                
                # Tuning hints
                if avg_pct_capped < 0.1:
                    print_acc(f"    → soft_cap may be too HIGH (no capping effect). Try lowering.")
                elif avg_pct_capped > 20:
                    print_acc(f"    → soft_cap may be too LOW (overly aggressive). Try raising.")
                elif avg_pct_capped > 1:
                    print_acc(f"    → soft_cap is actively capping. Distribution looks healthy.")

            print_acc(f"{'='*70}\n")

        except Exception as e:
            # Don't break training if logging fails
            pass

    def _get_active_expert_label(self):
        """Get the active expert label for per-expert training logging."""
        if hasattr(self.sd, 'model') and hasattr(self.sd.model, '_active_transformer_name'):
            active = self.sd.model._active_transformer_name
            if getattr(self.sd, 'train_high_noise', False) and getattr(self.sd, 'train_low_noise', False):
                return "high" if active == "transformer_1" else "low"
        return "single"

    def _get_expert_spectral_params(self):
        """
        Get spectral loss parameters for the currently active expert.
        
        Returns per-expert weights if configured, otherwise falls back to global weights.
        This allows different frequency emphasis for high-noise (structure) vs
        low-noise (texture) experts in MoE models like Wan 2.2 14B.
        
        Returns:
            dict with keys: low_weight, mid_weight, high_weight,
                           low_cutoff, high_cutoff, temporal_scale,
                           spectral_weight
        """
        expert = self._get_active_expert_label()
        tc = self.train_config
        
        # Determine overall spectral weight (scales entire spectral component)
        if expert == "low":
            spectral_w = tc.spectral_weight_low if tc.spectral_weight_low is not None else tc.spectral_weight
        elif expert == "high":
            spectral_w = tc.spectral_weight_high if tc.spectral_weight_high is not None else tc.spectral_weight
        else:  # single
            spectral_w = tc.spectral_weight
        
        # Determine per-band weights
        if expert == "low":
            low_w = tc.spectral_low_weight_low if tc.spectral_low_weight_low is not None else tc.spectral_low_weight
            mid_w = tc.spectral_mid_weight_low if tc.spectral_mid_weight_low is not None else tc.spectral_mid_weight
            high_w = tc.spectral_high_weight_low if tc.spectral_high_weight_low is not None else tc.spectral_high_weight
        elif expert == "high":
            low_w = tc.spectral_low_weight_high if tc.spectral_low_weight_high is not None else tc.spectral_low_weight
            mid_w = tc.spectral_mid_weight_high if tc.spectral_mid_weight_high is not None else tc.spectral_mid_weight
            high_w = tc.spectral_high_weight_high if tc.spectral_high_weight_high is not None else tc.spectral_high_weight
        else:  # single
            low_w = tc.spectral_low_weight
            mid_w = tc.spectral_mid_weight
            high_w = tc.spectral_high_weight
        
        # Determine cutoffs
        if expert == "low":
            low_c = tc.spectral_low_cutoff_low if tc.spectral_low_cutoff_low is not None else tc.spectral_low_cutoff
            high_c = tc.spectral_high_cutoff_low if tc.spectral_high_cutoff_low is not None else tc.spectral_high_cutoff
        elif expert == "high":
            low_c = tc.spectral_low_cutoff_high if tc.spectral_low_cutoff_high is not None else tc.spectral_low_cutoff
            high_c = tc.spectral_high_cutoff_high if tc.spectral_high_cutoff_high is not None else tc.spectral_high_cutoff
        else:  # single
            low_c = tc.spectral_low_cutoff
            high_c = tc.spectral_high_cutoff
        
        # Determine temporal scale
        if expert == "low":
            t_scale = tc.spectral_temporal_scale_low if tc.spectral_temporal_scale_low is not None else tc.spectral_temporal_scale
        elif expert == "high":
            t_scale = tc.spectral_temporal_scale_high if tc.spectral_temporal_scale_high is not None else tc.spectral_temporal_scale
        else:  # single
            t_scale = tc.spectral_temporal_scale
        
        return {
            'low_weight': low_w,
            'mid_weight': mid_w,
            'high_weight': high_w,
            'low_cutoff': low_c,
            'high_cutoff': high_c,
            'temporal_scale': t_scale,
            'spectral_weight': spectral_w,
        }

    def _get_timestep_range_override(self, current_timestep):
        """
        Get loss weight overrides for the given timestep based on configured timestep ranges.
        
        Ranges are specified in absolute model timesteps (0-1000). Each expert dynamically
        checks if its current timestep falls within a range. No scaling is applied.
        
        For dual-expert models (e.g., Wan 2.2 14B with boundary=900):
        - High-noise expert handles timesteps 900-1000
        - Low-noise expert handles timesteps 0-900
        
        Examples:
        - Range 1000-950: Only affects high-noise expert
        - Range 950-900: Only affects high-noise expert
        - Range 800-400: Only affects low-noise expert
        - Range 400-0: Only affects low-noise expert
        
        Note: Ranges crossing the boundary (e.g., 950-850) will apply to whichever expert
        is active at each timestep.
        
        Args:
            current_timestep: Current timestep in absolute model space (scalar or tensor)
        
        Returns:
            dict with override weights, or empty dict if no override applies.
        """
        tc = self.train_config
        if not tc.timestep_range_overrides:
            return {}
        
        # Convert current_timestep to a Python float for comparison
        if isinstance(current_timestep, torch.Tensor):
            current_t = current_timestep.item()
        else:
            current_t = float(current_timestep)
        
        # Find first matching range using absolute timesteps
        for override in tc.timestep_range_overrides:
            start = override.get('start_timestep', 0)
            end = override.get('end_timestep', 0)
            
            # Check if current timestep falls in this range
            # Range is [start, end) - inclusive of start, exclusive of end
            if start >= end:
                # Handle descending ranges (e.g., 1000-500)
                if current_t <= start and current_t > end:
                    return override
            else:
                # Handle ascending ranges (e.g., 0-500)
                if current_t >= start and current_t < end:
                    return override
        
        return {}

    def _get_expert_spectral_params_with_override(self, current_timestep=None):
        """
        Get spectral loss parameters for the currently active expert,
        with optional timestep range overrides applied.
        
        Args:
            current_timestep: Current timestep for range override lookup
        
        Returns:
            dict with keys: low_weight, mid_weight, high_weight,
                           low_cutoff, high_cutoff, temporal_scale,
                           spectral_weight
        """
        # Get base params from expert config
        params = self._get_expert_spectral_params()
        
        # Apply timestep range override if applicable
        if current_timestep is not None:
            override = self._get_timestep_range_override(current_timestep)
            if override:
                # Apply spectral band weight overrides
                if override.get('spectral_low_weight') is not None:
                    params['low_weight'] = override['spectral_low_weight']
                if override.get('spectral_mid_weight') is not None:
                    params['mid_weight'] = override['spectral_mid_weight']
                if override.get('spectral_high_weight') is not None:
                    params['high_weight'] = override['spectral_high_weight']
                if override.get('spectral_weight') is not None:
                    params['spectral_weight'] = override['spectral_weight']
                # Apply spectral filter overrides
                if override.get('spectral_low_cutoff') is not None:
                    params['low_cutoff'] = override['spectral_low_cutoff']
                if override.get('spectral_high_cutoff') is not None:
                    params['high_cutoff'] = override['spectral_high_cutoff']
                if override.get('spectral_temporal_scale') is not None:
                    params['temporal_scale'] = override['spectral_temporal_scale']
        
        return params

    def _get_flow_weight_with_override(self, current_timestep=None):
        """
        Get flow loss weight with timestep range override applied.
        
        Args:
            current_timestep: Current timestep for range override lookup
        
        Returns:
            float: Flow weight to use
        """
        expert = self._get_active_expert_label()
        tc = self.train_config
        
        # Get base flow weight
        if expert == "low" and tc.spectral_flow_weight_low is not None:
            base_flow_weight = tc.spectral_flow_weight_low
        elif expert == "high" and tc.spectral_flow_weight_high is not None:
            base_flow_weight = tc.spectral_flow_weight_high
        else:
            base_flow_weight = tc.spectral_flow_weight
        
        # Apply timestep range override if applicable
        if current_timestep is not None:
            override = self._get_timestep_range_override(current_timestep)
            if override and override.get('flow_weight') is not None:
                return override['flow_weight']
        
        return base_flow_weight

    def _get_mse_weight_with_override(self, current_timestep=None):
        """
        Get MSE loss weight with timestep range override applied.
        
        Args:
            current_timestep: Current timestep for range override lookup
        
        Returns:
            float: MSE weight to use
        """
        expert = self._get_active_expert_label()
        tc = self.train_config
        
        # Get base MSE weight
        if expert == "low" and tc.mse_spectral_flow_mse_weight_low is not None:
            base_mse_weight = tc.mse_spectral_flow_mse_weight_low
        elif expert == "high" and tc.mse_spectral_flow_mse_weight_high is not None:
            base_mse_weight = tc.mse_spectral_flow_mse_weight_high
        else:
            base_mse_weight = tc.mse_spectral_flow_mse_weight
        
        # Apply timestep range override if applicable
        if current_timestep is not None:
            override = self._get_timestep_range_override(current_timestep)
            if override and override.get('mse_weight') is not None:
                return override['mse_weight']
        
        return base_mse_weight

    def _get_lcr_weight_with_override(self, current_timestep=None):
        """
        Get LCR (Low-Cut Ratio) weight with timestep range override applied.
        
        Args:
            current_timestep: Current timestep for range override lookup
        
        Returns:
            float: LCR weight to use
        """
        tc = self.train_config
        
        # Get base LCR weight
        base_lcr_weight = tc.spectral_lcr_weight
        
        # Apply timestep range override if applicable
        if current_timestep is not None:
            override = self._get_timestep_range_override(current_timestep)
            if override and override.get('spectral_lcr_weight') is not None:
                return override['spectral_lcr_weight']
        
        return base_lcr_weight

    def _update_flow_gate_log(self, timesteps):
        """Log the per-batch flow gate mean and its running EMA.

        The flow loss is a zero-inflated estimator: items sampled at
        t >= max_timestep have gate=0 and contribute nothing. The loss is
        renormalized by the per-batch gate sum (see FlowConsistencyLoss), so the
        gate mean here is the expected fraction of the batch that actually drives
        the flow objective -- a direct readout of the flow dilution factor for the
        current timestep distribution.
        """
        if timesteps is None:
            return
        if isinstance(timesteps, torch.Tensor):
            t = timesteps[:, 0].float() if timesteps.dim() == 2 else timesteps.float()
        else:
            t = torch.as_tensor(float(timesteps), dtype=torch.float32)
        max_t = float(self.train_config.spectral_flow_max_timestep)
        if self.train_config.spectral_flow_reverse_gate:
            gate = torch.clamp(t / max_t, min=0.0, max=1.0)
        else:
            gate = torch.clamp(1.0 - (t / max_t), min=0.0)
        gmean = float(gate.mean().item())
        self._flow_gate_ema = gmean if self._flow_gate_ema <= 1e-6 else 0.98 * self._flow_gate_ema + 0.02 * gmean
        self.additional_logs['flow/gate_mean'] = gmean
        self.additional_logs['flow/gate_ema'] = self._flow_gate_ema

    def _gradient_projection_backward(self, spectral_loss, flow_loss, mse_loss=None):
        """Compute gradients for losses separately, then project (PCGrad).
        
        This implements gradient projection (PCGrad) to resolve conflicts between
        loss objectives. When gradients conflict, each gradient is projected to remove
        the component that would worsen other losses.
        
        Supports both 2-loss (spectral+flow) and 3-loss (mse+spectral+flow) modes.
        
        Uses torch.autograd.grad() instead of .backward() to avoid in-place modifications
        to shared computation graph intermediates (critical for compiled models).
        
        Handles gradient accumulation correctly by preserving accumulated gradients from
        previous batches while computing per-batch projected gradients.
        
        Args:
            spectral_loss: scalar loss tensor for spectral component
            flow_loss: scalar loss tensor for flow component
            mse_loss: scalar loss tensor for MSE component (optional, for mse_spectral_flow)
        """
        # Collect all parameters that need gradients
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad:
                    params.append(param)
        
        if not params:
            # Fallback to normal backward
            total_loss = spectral_loss + flow_loss + (mse_loss if mse_loss is not None else 0.0)
            if total_loss.requires_grad:
                self.accelerator.backward(total_loss)
            # If no component has a computation graph (e.g. all weights zero /
            # flow gate 0), there is nothing to backprop - skip silently.
            return
        
        # Save currently accumulated gradients (from previous batches in accumulation loop)
        # These must be preserved - we only want to compute and project THIS batch's gradients
        saved_grads = {}
        for p in params:
            if p.grad is not None:
                saved_grads[p] = p.grad.clone()
            else:
                saved_grads[p] = None
        
        # Helper function: compute gradient of a loss w.r.t. params using autograd.grad()
        # This avoids in-place modifications to shared computation graph intermediates
        def compute_grad(loss_tensor):
            """Compute gradient dict for a loss using torch.autograd.grad().

            Returns an empty dict when the loss has no computation graph.
            This happens when a component's computation is skipped entirely,
            e.g. the flow loss when the timestep gate is 0 for the whole batch
            or the flow weight is 0 (the loss functions return a plain
            constant tensor in that case), or the spectral loss when all
            spectral band weights are 0. A constant contributes no gradient;
            calling autograd.grad() on such a tensor would raise
            "element 0 of tensors does not require grad and does not have a grad_fn".
            """
            if loss_tensor is None or loss_tensor.grad_fn is None:
                return {}
            grads = torch.autograd.grad(
                loss_tensor, params, retain_graph=True, allow_unused=True
            )
            grad_dict = {}
            for p, g in zip(params, grads):
                if g is not None:
                    grad_dict[p] = g.clone()
            return grad_dict
        
        is_three_loss = mse_loss is not None
        
        if is_three_loss:
            # === 3-LOSS MODE: MSE + Spectral + Flow ===
            # Use autograd.grad() for each loss - no shared state issues
            grad_mse = compute_grad(mse_loss)
            grad_spectral = compute_grad(spectral_loss)
            grad_flow = compute_grad(flow_loss)
            
            # Project gradients (PCGrad: symmetric projection of all three)
            projected_mse = self._project_gradient_3way(grad_mse, grad_spectral, grad_flow)
            projected_spectral = self._project_gradient_3way(grad_spectral, grad_mse, grad_flow)
            projected_flow = self._project_gradient_3way(grad_flow, grad_mse, grad_spectral)
            
            # Set combined projected gradients into parameters
            for p in params:
                base_grad = saved_grads[p].clone() if saved_grads[p] is not None else torch.zeros_like(p)
                
                if p in projected_mse and p in projected_spectral and p in projected_flow:
                    p.grad = base_grad + projected_mse[p] + projected_spectral[p] + projected_flow[p]
                elif p in projected_mse and p in projected_spectral:
                    p.grad = base_grad + projected_mse[p] + projected_spectral[p]
                elif p in projected_mse and p in projected_flow:
                    p.grad = base_grad + projected_mse[p] + projected_flow[p]
                elif p in projected_spectral and p in projected_flow:
                    p.grad = base_grad + projected_spectral[p] + projected_flow[p]
                elif p in projected_mse:
                    p.grad = base_grad + projected_mse[p]
                elif p in projected_spectral:
                    p.grad = base_grad + projected_spectral[p]
                elif p in projected_flow:
                    p.grad = base_grad + projected_flow[p]
                else:
                    p.grad = base_grad
        else:
            # === 2-LOSS MODE: Spectral + Flow ===
            grad_spectral = compute_grad(spectral_loss)
            grad_flow = compute_grad(flow_loss)
            
            # Project gradients (PCGrad: symmetric projection of both)
            projected_spectral = self._project_gradients(grad_spectral, grad_flow)
            projected_flow = self._project_gradients(grad_flow, grad_spectral)
            
            # Set combined projected gradients into parameters
            for p in params:
                base_grad = saved_grads[p].clone() if saved_grads[p] is not None else torch.zeros_like(p)
                
                if p in projected_spectral and p in projected_flow:
                    p.grad = base_grad + projected_spectral[p] + projected_flow[p]
                elif p in projected_spectral:
                    p.grad = base_grad + projected_spectral[p]
                elif p in projected_flow:
                    p.grad = base_grad + projected_flow[p]
                else:
                    p.grad = base_grad

    def _project_gradients(self, grad_spatial_dict, grad_flow_dict):
        """Project spectral gradient to remove component that conflicts with flow gradient.
        
        Implements PCGrad-style projection: when ∇L_spatial and ∇L_flow conflict
        (one wants to increase what other wants to decrease), project ∇L_spatial
        onto the direction that doesn't worsen flow loss.
        
        Args:
            grad_spatial_dict: dict of {param: gradient} for spectral loss
            grad_flow_dict: dict of {param: gradient} for flow loss
        
        Returns:
            projected_grad_spatial_dict: modified gradients for spectral loss
        """
        def gradient_dot(grad_a, grad_b):
            """Compute dot product of two gradient dictionaries."""
            dot = 0.0
            for param in grad_a:
                if param in grad_b:
                    dot += (grad_a[param] * grad_b[param]).sum().item()
            return dot
        
        def gradient_norm_sq(grad):
            """Compute squared L2 norm of gradient dictionary."""
            norm_sq = 0.0
            for param in grad:
                norm_sq += (grad[param] * grad[param]).sum().item()
            return norm_sq
        
        # Check for conflict: dot product < 0 means gradients oppose each other
        dot_product = gradient_dot(grad_spatial_dict, grad_flow_dict)
        
        # Track conflict statistics (both cumulative and per-step)
        if dot_product < 0:
            self.gradient_projection_stats['total_conflicts'] += 1
            self.gradient_projection_stats['step_conflicts'] += 1
        
        if dot_product >= 0:
            # No conflict - gradients align, return unchanged
            return grad_spatial_dict
        
        # Conflict detected: project grad_spatial to remove component that conflicts with grad_flow
        # g_proj = g_spatial - (g_spatial · g_flow / ||g_flow||²) * g_flow
        flow_norm_sq = gradient_norm_sq(grad_flow_dict)
        
        if flow_norm_sq < 1e-10:
            return grad_spatial_dict
        
        proj_coef = dot_product / flow_norm_sq
        self.gradient_projection_stats['total_projections'] += 1
        self.gradient_projection_stats['step_projections'] += 1
        
        # Apply projection
        projected_grad_spatial = {}
        for param in grad_spatial_dict:
            if param in grad_flow_dict:
                projected_grad_spatial[param] = grad_spatial_dict[param] - proj_coef * grad_flow_dict[param]
            else:
                projected_grad_spatial[param] = grad_spatial_dict[param].clone()
        
        return projected_grad_spatial

    def _project_gradient_3way(self, grad_target, grad_other1, grad_other2):
        """Project target gradient against two other gradients sequentially (3-way PCGrad).
        
        For mse_spectral_flow loss, projects one gradient (e.g., MSE) against the other two
        (e.g., spectral and flow) by sequential projection. This removes components that
        conflict with either of the other gradients.
        
        Algorithm:
        1. Project g_target onto plane orthogonal to g_other1 (if conflicting)
        2. Project result onto plane orthogonal to g_other2 (if conflicting)
        
        Args:
            grad_target: dict of {param: gradient} for the loss to project
            grad_other1: dict of {param: gradient} for first constraint loss
            grad_other2: dict of {param: gradient} for second constraint loss
        
        Returns:
            projected_grad_target: modified gradients for target loss
        """
        def gradient_dot(grad_a, grad_b):
            dot = 0.0
            for param in grad_a:
                if param in grad_b:
                    dot += (grad_a[param] * grad_b[param]).sum().item()
            return dot
        
        def gradient_norm_sq(grad):
            norm_sq = 0.0
            for param in grad:
                norm_sq += (grad[param] * grad[param]).sum().item()
            return norm_sq
        
        def project_onto_normal(grad_to_project, grad_constraint):
            """Project grad_to_project onto normal plane of grad_constraint if conflicting."""
            dot_product = gradient_dot(grad_to_project, grad_constraint)
            
            if dot_product >= 0:
                # No conflict
                return grad_to_project
            
            # Track conflict
            self.gradient_projection_stats['total_conflicts'] += 1
            self.gradient_projection_stats['step_conflicts'] += 1
            
            constraint_norm_sq = gradient_norm_sq(grad_constraint)
            if constraint_norm_sq < 1e-10:
                return grad_to_project
            
            proj_coef = dot_product / constraint_norm_sq
            
            # Track projection
            self.gradient_projection_stats['total_projections'] += 1
            self.gradient_projection_stats['step_projections'] += 1
            
            projected = {}
            for param in grad_to_project:
                if param in grad_constraint:
                    projected[param] = grad_to_project[param] - proj_coef * grad_constraint[param]
                else:
                    projected[param] = grad_to_project[param].clone()
            
            return projected
        
        # Sequential projection: first against other1, then against other2
        result = project_onto_normal(grad_target, grad_other1)
        result = project_onto_normal(result, grad_other2)
        
        return result

    def _should_reject_step(self, expert_label, current_loss, current_spatial_loss, current_flow_loss):
        """Check if optimizer step should be rejected based on loss thresholds.
        
        Supports two modes:
        1. Basic threshold mode: reject steps with excessive loss or spikes
        2. Constraint mode: enforce spectral loss decrease while bounding flow loss increase
        
        Returns:
            tuple: (should_reject: bool, reason: str)
        """
        # Mode 1: Basic threshold checks
        if self.train_config.spectral_flow_loss_rejection_enabled:
            # Get threshold for this expert
            if expert_label == "low":
                max_loss = self.train_config.spectral_flow_loss_rejection_max_low
            elif expert_label == "high":
                max_loss = self.train_config.spectral_flow_loss_rejection_max_high
            else:
                # Single expert: use average of both thresholds
                max_loss = (self.train_config.spectral_flow_loss_rejection_max_low + 
                           self.train_config.spectral_flow_loss_rejection_max_high) / 2
            
            max_increase_pct = self.train_config.spectral_flow_loss_rejection_max_increase_pct
            
            # Check absolute threshold
            if current_loss > max_loss:
                return True, f"loss {current_loss:.4f} > max {max_loss:.1f}"
            
            # Check % increase from previous step
            prev_loss = self.prev_expert_loss.get(expert_label)
            if prev_loss is not None and prev_loss > 0:
                increase_pct = ((current_loss - prev_loss) / prev_loss) * 100
                if increase_pct > max_increase_pct:
                    return True, f"increase {increase_pct:.1f}% > max {max_increase_pct:.0f}%"
        
        # Mode 2: Constraint-based rejection (spectral primary, flow constraint)
        if self.train_config.spectral_flow_constraint_rejection_enabled:
            prev_spatial = self.prev_expert_spatial_loss.get(expert_label)
            prev_flow = self.prev_expert_flow_loss.get(expert_label)
            
            if prev_spatial is not None and prev_flow is not None:
                # Calculate changes
                spatial_change = current_spatial_loss - prev_spatial
                flow_change = current_flow_loss - prev_flow
                flow_increase_pct = (flow_change / prev_flow * 100) if prev_flow > 0 else 0
                
                # Constraint: flow loss should not increase significantly
                max_flow_increase_pct = self.train_config.spectral_flow_constraint_flow_max_increase_pct
                
                # Case: spectral improved but flow got worse → reject (wrong direction)
                if spatial_change < 0 and flow_increase_pct > max_flow_increase_pct:
                    return True, f"constraint: spectral↓ but flow↑ {flow_increase_pct:.1f}% (max {max_flow_increase_pct:.0f}%)"
                
                # Optional: require spectral to actually decrease
                if (self.train_config.spectral_flow_constraint_spectral_must_decrease and
                    spatial_change > 0):
                    return True, f"constraint: spectral didn't decrease (+{spatial_change:.4f})"
        
        return False, ""

    def process_output_for_turbo(self, pred, noisy_latents, timesteps, noise, batch):
        # to process turbo learning, we make one big step from our current timestep to the end
        # we then denoise the prediction on that remaining step and target our loss to our target latents
        # this currently only works on euler_a (that I know of). Would work on others, but needs to be coded to do so.
        # needs to be done on each item in batch as they may all have different timesteps
        batch_size = pred.shape[0]
        pred_chunks = torch.chunk(pred, batch_size, dim=0)
        noisy_latents_chunks = torch.chunk(noisy_latents, batch_size, dim=0)
        timesteps_chunks = torch.chunk(timesteps, batch_size, dim=0)
        latent_chunks = torch.chunk(batch.latents, batch_size, dim=0)
        noise_chunks = torch.chunk(noise, batch_size, dim=0)

        with torch.no_grad():
            # set the timesteps to 1000 so we can capture them to calculate the sigmas
            self.sd.noise_scheduler.set_timesteps(
                self.sd.noise_scheduler.config.num_train_timesteps,
                device=self.device_torch
            )
            train_timesteps = self.sd.noise_scheduler.timesteps.clone().detach()

            train_sigmas = self.sd.noise_scheduler.sigmas.clone().detach()

            # set the scheduler to one timestep, we build the step and sigmas for each item in batch for the partial step
            self.sd.noise_scheduler.set_timesteps(
                1,
                device=self.device_torch
            )

        denoised_pred_chunks = []
        target_pred_chunks = []

        for i in range(batch_size):
            pred_item = pred_chunks[i]
            noisy_latents_item = noisy_latents_chunks[i]
            timesteps_item = timesteps_chunks[i]
            latents_item = latent_chunks[i]
            noise_item = noise_chunks[i]
            with torch.no_grad():
                timestep_idx = [(train_timesteps == t).nonzero().item() for t in timesteps_item][0]
                single_step_timestep_schedule = [timesteps_item.squeeze().item()]
                # extract the sigma idx for our midpoint timestep
                sigmas = train_sigmas[timestep_idx:timestep_idx + 1].to(self.device_torch)

                end_sigma_idx = random.randint(timestep_idx, len(train_sigmas) - 1)
                end_sigma = train_sigmas[end_sigma_idx:end_sigma_idx + 1].to(self.device_torch)

                # add noise to our target

                # build the big sigma step. The to step will now be to 0 giving it a full remaining denoising half step
                # self.sd.noise_scheduler.sigmas = torch.cat([sigmas, torch.zeros_like(sigmas)]).detach()
                self.sd.noise_scheduler.sigmas = torch.cat([sigmas, end_sigma]).detach()
                # set our single timstep
                self.sd.noise_scheduler.timesteps = torch.from_numpy(
                    np.array(single_step_timestep_schedule, dtype=np.float32)
                ).to(device=self.device_torch)

                # set the step index to None so it will be recalculated on first step
                self.sd.noise_scheduler._step_index = None

            denoised_latent = self.sd.noise_scheduler.step(
                pred_item, timesteps_item, noisy_latents_item.detach(), return_dict=False
            )[0]

            residual_noise = (noise_item * end_sigma.flatten()).detach().to(self.device_torch, dtype=get_torch_dtype(
                self.train_config.dtype))
            # remove the residual noise from the denoised latents. Output should be a clean prediction (theoretically)
            denoised_latent = denoised_latent - residual_noise

            denoised_pred_chunks.append(denoised_latent)

        denoised_latents = torch.cat(denoised_pred_chunks, dim=0)
        # set the scheduler back to the original timesteps
        self.sd.noise_scheduler.set_timesteps(
            self.sd.noise_scheduler.config.num_train_timesteps,
            device=self.device_torch
        )

        output = denoised_latents / self.sd.vae.config['scaling_factor']
        output = self.sd.vae.decode(output).sample

        if self.train_config.show_turbo_outputs:
            # since we are completely denoising, we can show them here
            with torch.no_grad():
                show_tensors(output)

        # we return our big partial step denoised latents as our pred and our untouched latents as our target.
        # you can do mse against the two here  or run the denoised through the vae for pixel space loss against the
        # input tensor images.

        return output, batch.tensor.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))

    # you can expand these in a child class to make customization easier
    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        loss_target = self.train_config.loss_target
        is_reg = any(batch.get_is_reg_list())
        additional_loss = 0.0

        prior_mask_multiplier = None
        target_mask_multiplier = None
        dtype = get_torch_dtype(self.train_config.dtype)

        has_mask = batch.mask_tensor is not None

        with torch.no_grad():
            loss_multiplier = torch.tensor(batch.loss_multiplier_list).to(self.device_torch, dtype=torch.float32)

        if self.train_config.match_noise_norm:
            # match the norm of the noise
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)

        if self.train_config.pred_scaler != 1.0:
            noise_pred = noise_pred * self.train_config.pred_scaler

        target = None

        if self.train_config.target_noise_multiplier != 1.0:
            noise = noise * self.train_config.target_noise_multiplier

        if self.train_config.correct_pred_norm or (self.train_config.inverted_mask_prior and prior_pred is not None and has_mask):
            if self.train_config.correct_pred_norm and not is_reg:
                with torch.no_grad():
                    # this only works if doing a prior pred
                    if prior_pred is not None:
                        prior_mean = prior_pred.mean([2,3], keepdim=True)
                        prior_std = prior_pred.std([2,3], keepdim=True)
                        noise_mean = noise_pred.mean([2,3], keepdim=True)
                        noise_std = noise_pred.std([2,3], keepdim=True)

                        mean_adjust = prior_mean - noise_mean
                        std_adjust = prior_std - noise_std

                        mean_adjust = mean_adjust * self.train_config.correct_pred_norm_multiplier
                        std_adjust = std_adjust * self.train_config.correct_pred_norm_multiplier

                        target_mean = noise_mean + mean_adjust
                        target_std = noise_std + std_adjust

                        eps = 1e-5
                        # match the noise to the prior
                        noise = (noise - noise_mean) / (noise_std + eps)
                        noise = noise * (target_std + eps) + target_mean
                        noise = noise.detach()

            if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
                assert not self.train_config.train_turbo
                with torch.no_grad():
                    prior_mask = batch.mask_tensor.to(self.device_torch, dtype=dtype)
                    if len(noise_pred.shape) == 5:
                        # video B,C,T,H,W
                        lat_height = batch.latents.shape[3]
                        lat_width = batch.latents.shape[4]
                    else: 
                        lat_height = batch.latents.shape[2]
                        lat_width = batch.latents.shape[3]
                    # resize to size of noise_pred
                    prior_mask = torch.nn.functional.interpolate(prior_mask, size=(lat_height, lat_width), mode='bicubic')
                    # stack first channel to match channels of noise_pred
                    prior_mask = torch.cat([prior_mask[:1]] * noise_pred.shape[1], dim=1)
                    
                    if len(noise_pred.shape) == 5:
                        prior_mask = prior_mask.unsqueeze(2)  # add time dimension back for video
                        prior_mask = prior_mask.repeat(1, 1, noise_pred.shape[2], 1, 1) 

                    prior_mask_multiplier = 1.0 - prior_mask
                    
                    # scale so it is a mean of 1
                    prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
                if hasattr(self.sd, 'get_loss_target'):
                    target = self.sd.get_loss_target(
                        noise=noise, 
                        batch=batch, 
                        timesteps=timesteps,
                    ).detach()
                elif self.sd.is_flow_matching:
                    target = (noise - batch.latents).detach()
                else:
                    target = noise
        elif prior_pred is not None and not self.train_config.do_prior_divergence:
            assert not self.train_config.train_turbo
            # matching adapter prediction
            target = prior_pred
        elif self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = self.sd.noise_scheduler.get_velocity(batch.tensor, noise, timesteps)
        elif self.train_config.do_signal_amplification:
            if not self.sd.is_flow_matching:
                raise ValueError("Signal amplification is only supported for flow matching models")
            with torch.no_grad():
                nas = 1.0 - (timesteps / 1000).to(noise.device, dtype=noise.dtype)
                nas = nas * self.train_config.signal_amplification_strength
                while len(nas.shape) < len(noise.shape):
                    nas = nas.unsqueeze(-1)
                aug = batch.latents * nas
                target = noise - (batch.latents + aug)
                target = target.detach()
        elif hasattr(self.sd, 'get_loss_target'):
            target = self.sd.get_loss_target(
                noise=noise, 
                batch=batch, 
                timesteps=timesteps,
            ).detach()
            
        elif self.sd.is_flow_matching:
            # forward ODE
            target = (noise - batch.latents).detach()
            # reverse ODE
            # target = (batch.latents - noise).detach()
        else:
            target = noise
            
        if self.dfe is not None:
            if self.dfe.version == 1:
                model = self.sd
                if model is not None and hasattr(model, 'get_stepped_pred'):
                    stepped_latents = model.get_stepped_pred(noise_pred, noise)
                else:
                    # stepped_latents = noise - noise_pred
                    # first we step the scheduler from current timestep to the very end for a full denoise
                    bs = noise_pred.shape[0]
                    noise_pred_chunks = torch.chunk(noise_pred, bs)
                    timestep_chunks = torch.chunk(timesteps, bs)
                    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
                    stepped_chunks = []
                    for idx in range(bs):
                        model_output = noise_pred_chunks[idx]
                        timestep = timestep_chunks[idx]
                        self.sd.noise_scheduler._step_index = None
                        self.sd.noise_scheduler._init_step_index(timestep)
                        sample = noisy_latent_chunks[idx].to(torch.float32)
                        
                        sigma = self.sd.noise_scheduler.sigmas[self.sd.noise_scheduler.step_index]
                        sigma_next = self.sd.noise_scheduler.sigmas[-1] # use last sigma for final step
                        prev_sample = sample + (sigma_next - sigma) * model_output
                        stepped_chunks.append(prev_sample)
                    
                    stepped_latents = torch.cat(stepped_chunks, dim=0)
                    
                stepped_latents = stepped_latents.to(self.sd.vae.device, dtype=self.sd.vae.dtype)
                sl = stepped_latents
                if len(sl.shape) == 5:
                    # video B,C,T,H,W
                    sl = sl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                    b, t, c, h, w = sl.shape
                    sl = sl.reshape(b * t, c, h, w)
                pred_features = self.dfe(sl.float())
                with torch.no_grad():
                    bl = batch.latents
                    bl = bl.to(self.sd.vae.device)
                    if len(bl.shape) == 5:
                        # video B,C,T,H,W
                        bl = bl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                        b, t, c, h, w = bl.shape
                        bl = bl.reshape(b * t, c, h, w)
                    target_features = self.dfe(bl.float())
                    # scale dfe so it is weaker at higher noise levels
                    dfe_scaler = 1 - (timesteps.float() / 1000.0).view(-1, 1, 1, 1).to(self.device_torch)
                
                dfe_loss = torch.nn.functional.mse_loss(pred_features, target_features, reduction="none") * \
                    self.train_config.diffusion_feature_extractor_weight * dfe_scaler
                additional_loss += dfe_loss.mean()
            elif self.dfe.version == 2:
                # version 2
                # do diffusion feature extraction on target
                with torch.no_grad():
                    rectified_flow_target = noise.float() - batch.latents.float()
                    target_feature_list = self.dfe(torch.cat([rectified_flow_target, noise.float()], dim=1))
                
                # do diffusion feature extraction on prediction
                pred_feature_list = self.dfe(torch.cat([noise_pred.float(), noise.float()], dim=1))
                
                dfe_loss = 0.0
                for i in range(len(target_feature_list)):
                    dfe_loss += torch.nn.functional.mse_loss(pred_feature_list[i], target_feature_list[i], reduction="mean")
                
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight * 100.0
            elif self.dfe.version in [3, 4, 5, 6, 7, 8, 9, 10]:
                dfe_loss = self.dfe(
                    noise=noise,
                    noise_pred=noise_pred,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    batch=batch,
                    scheduler=self.sd.noise_scheduler
                )
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight 
            else:
                raise ValueError(f"Unknown diffusion feature extractor version {self.dfe.version}")
        
        if self.train_config.do_guidance_loss:
            with torch.no_grad():
                # we make cached blank prompt embeds that match the batch size
                unconditional_embeds = concat_prompt_embeds(
                    [self.unconditional_embeds] * noisy_latents.shape[0],
                )
                unconditional_target = self.predict_noise(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    conditional_embeds=unconditional_embeds,
                    unconditional_embeds=None,
                    batch=batch,
                )
                is_video = len(target.shape) == 5
                
                if self.train_config.do_guidance_loss_cfg_zero:
                    # zero cfg
                    # ref https://github.com/WeichenFan/CFG-Zero-star/blob/cdac25559e3f16cb95f0016c04c709ea1ab9452b/wan_pipeline.py#L557
                    batch_size = target.shape[0]
                    positive_flat = target.view(batch_size, -1)
                    negative_flat = unconditional_target.view(batch_size, -1)
                    # Calculate dot production
                    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                    # Squared norm of uncondition
                    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
                    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
                    st_star = dot_product / squared_norm

                    alpha = st_star
                    
                    alpha = alpha.view(batch_size, 1, 1, 1) if not is_video else alpha.view(batch_size, 1, 1, 1, 1)
                else:
                    alpha = 1.0

                guidance_scale = self._guidance_loss_target_batch
                if isinstance(guidance_scale, list):
                    guidance_scale = torch.tensor(guidance_scale).to(target.device, dtype=target.dtype)
                    guidance_scale = guidance_scale.view(-1, 1, 1, 1) if not is_video else guidance_scale.view(-1, 1, 1, 1, 1)
                
                unconditional_target = unconditional_target * alpha
                target = unconditional_target + guidance_scale * (target - unconditional_target)

            if self.train_config.do_differential_guidance:
                with torch.no_grad():
                    guidance_scale = self.train_config.differential_guidance_scale
                    target = noise_pred + guidance_scale * (target - noise_pred)
            
        if target is None:
            target = noise

        pred = noise_pred

        if self.train_config.train_turbo:
            pred, target = self.process_output_for_turbo(pred, noisy_latents, timesteps, noise, batch)

        ignore_snr = False

        if loss_target == 'source' or loss_target == 'unaugmented':
            assert not self.train_config.train_turbo
            # ignore_snr = True
            if batch.sigmas is None:
                raise ValueError("Batch sigmas is None. This should not happen")

            # src https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1190
            denoised_latents = noise_pred * (-batch.sigmas) + noisy_latents
            weighing = batch.sigmas ** -2.0
            if loss_target == 'source':
                # denoise the latent and compare to the latent in the batch
                target = batch.latents
            elif loss_target == 'unaugmented':
                # we have to encode images into latents for now
                # we also denoise as the unaugmented tensor is not a noisy diffirental
                with torch.no_grad():
                    unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor).to(self.device_torch, dtype=dtype)
                    unaugmented_latents = unaugmented_latents * self.train_config.latent_multiplier
                    target = unaugmented_latents.detach()

                # Get the target for loss depending on the prediction type
                if self.sd.noise_scheduler.config.prediction_type == "epsilon":
                    target = target  # we are computing loss against denoise latents
                elif self.sd.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.sd.noise_scheduler.get_velocity(target, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.sd.noise_scheduler.config.prediction_type}")

            # mse loss without reduction
            loss_per_element = (weighing.float() * (denoised_latents.float() - target.float()) ** 2)
            loss = loss_per_element
        else:
            local_loss_scale = 1.0
            if self.train_config.t0_loss_target or self.train_config.do_fft_loss:
                # do the loss on a stepped timestep 0 prediction
                # doto handle doing priors, preservations, masking, etc
                with torch.no_grad():
                    tv = timesteps.to(noise_pred.device).to(noise_pred.dtype) / 1000.0
                    # expand shape to match noise_pred
                    while len(tv.shape) < len(noise_pred.shape):
                        tv = tv.unsqueeze(-1)
                        # min 0.001
                        tv = torch.clamp(tv, min=0.001)
                
                # step latent, use here or with do_fft_loss
                if self.sd.x0_pred:
                    t0 = noise_pred
                else:
                    t0 = noisy_latents - tv * noise_pred
                
                if self.train_config.t0_loss_target:
                    # replace the loss targets and pred
                    target = batch.latents.detach()
                    pred = t0
                    # handle velocity equiv loss if set. This scales t0 loss to match velocity of flowmatchhing loss
                    if self.train_config.t0_velocity_equiv_weight:
                        velocity_equiv_weight = (1.0 / torch.clamp(tv, min=0.1) ** 2)
                        local_loss_scale = velocity_equiv_weight
                        
                if self.train_config.do_fft_loss:
                    with torch.no_grad():
                        target_mag = torch.fft.rfft2(batch.latents.to(t0.device).float(), norm="ortho").abs()
                    pred_mag = torch.fft.rfft2(t0.float(), norm="ortho").abs()
                    fft_loss = F.mse_loss(pred_mag, target_mag, reduction="none")
                    if self.train_config.do_fft_velocity_equiv_weight:
                        velocity_equiv_weight = (1.0 / torch.clamp(tv, min=0.1) ** 2)
                        fft_loss = fft_loss * velocity_equiv_weight
                    additional_loss += fft_loss.mean()
            if self.train_config.loss_type == "pseudo_huber":
                diff = pred.float() - target.float()
                c = self.train_config.pseudo_huber_c
                loss = (torch.sqrt(diff.pow(2) + c ** 2) - c)
            elif self.train_config.loss_type == "mae":
                loss = torch.nn.functional.l1_loss(pred.float(), target.float(), reduction="none")
            elif self.train_config.loss_type == "wavelet":
                loss = wavelet_loss(pred, batch.latents, noise)
            elif self.train_config.loss_type == "spectral":
                # Get per-expert spectral parameters with timestep range override
                # Use first timestep (all timesteps in batch are identical in flow-matching)
                current_t = timesteps[0].item() if isinstance(timesteps, torch.Tensor) else float(timesteps)
                spec_params = self._get_expert_spectral_params_with_override(current_t)
                lcr_weight = self._get_lcr_weight_with_override(current_t)
                loss = spectral_loss(
                    pred,
                    batch.latents,
                    noise,
                    low_weight=spec_params['low_weight'],
                    mid_weight=spec_params['mid_weight'],
                    high_weight=spec_params['high_weight'],
                    low_cutoff=spec_params['low_cutoff'],
                    high_cutoff=spec_params['high_cutoff'],
                    use_phase=self.train_config.spectral_use_phase,
                    lcr_weight=lcr_weight,
                    spectral_transform=self.train_config.spectral_transform,
                    prediction_target=self.train_config.prediction_target,
                    temporal_scale=spec_params['temporal_scale'],
                )
            elif self.train_config.loss_type == "spectral_flow":
                # Combined spectral (spatial frequency) + optical flow (temporal motion) loss

                # BUG FIX D: guard against x0_pred mode (e.g. turbo or any x0-predicting model).
                # spectral_flow loss assumes velocity prediction (model_pred = ε - x0), so
                # x0 reconstruction uses: pred_latents = noise - model_pred.
                # If the model directly predicts x0, this gives noise - x0 ≠ x0.
                # Previously the condition was "x0_pred AND train_turbo" which missed
                # cases where x0_pred=True but train_turbo=False.
                if getattr(self.sd, 'x0_pred', False):
                    if self.accelerator.is_main_process:
                        print_acc("[WARN] spectral_flow loss is incompatible with x0_pred mode. "
                                  "Falling back to spectral loss.")
                    # Get per-expert spectral parameters with timestep range override
                    # Use first timestep (all timesteps in batch are identical in flow-matching)
                    current_t = timesteps[0].item() if isinstance(timesteps, torch.Tensor) else float(timesteps)
                    spec_params = self._get_expert_spectral_params_with_override(current_t)
                    lcr_weight = self._get_lcr_weight_with_override(current_t)
                    loss = spectral_loss(
                        pred,
                        batch.latents,
                        noise,
                        low_weight=spec_params['low_weight'],
                        mid_weight=spec_params['mid_weight'],
                        high_weight=spec_params['high_weight'],
                        low_cutoff=spec_params['low_cutoff'],
                        high_cutoff=spec_params['high_cutoff'],
                        use_phase=self.train_config.spectral_use_phase,
                        lcr_weight=lcr_weight,
                        spectral_transform=self.train_config.spectral_transform,
                        prediction_target=self.train_config.prediction_target,
                        temporal_scale=spec_params['temporal_scale'],
                    )
                    # Continue with standard loss handling below (falls through)
                else:
                    vae_ts = self.flow_loss_module.vae_temporal_stride if self.flow_loss_module else 4
                    vae_ss = self.flow_loss_module.vae_spatial_stride if self.flow_loss_module else 8

                    expert = self._get_active_expert_label()

                    # Use timestep-aware weight getters with range override support
                    # Use first timestep (all timesteps in batch are identical in flow-matching)
                    current_t = timesteps[0].item() if isinstance(timesteps, torch.Tensor) else float(timesteps)
                    base_flow_weight = self._get_flow_weight_with_override(current_t)

                    # Issue #1 fix: use per-expert current_flow_weight (adaptive adjustment)
                    expert_flow_weight = self.current_flow_weight.get(expert, base_flow_weight)

                    # Get per-expert spectral parameters with timestep range override
                    spec_params = self._get_expert_spectral_params_with_override(current_t)
                    lcr_weight = self._get_lcr_weight_with_override(current_t)

                    # Log the flow gate mean (effective flow dilution factor)
                    self._update_flow_gate_log(timesteps)

                    (total_loss, flow_dev, spatial_val, flow_val,
                     spectral_component, flow_component) = spectral_flow_loss(
                        model_pred=pred,
                        latents=batch.latents,
                        noise=noise,
                        batch_flow=getattr(batch, 'flow', None),
                        timesteps=timesteps,
                        flow_loss_module=self.flow_loss_module,
                        vae_temporal_stride=vae_ts,
                        vae_spatial_stride=vae_ss,
                        low_weight=spec_params['low_weight'],
                        mid_weight=spec_params['mid_weight'],
                        high_weight=spec_params['high_weight'],
                        low_cutoff=spec_params['low_cutoff'],
                        high_cutoff=spec_params['high_cutoff'],
                        use_phase=self.train_config.spectral_use_phase,
                        lcr_weight=lcr_weight,
                        spectral_transform=self.train_config.spectral_transform,
                        prediction_target=self.train_config.prediction_target,
                        temporal_scale=spec_params['temporal_scale'],
                        spectral_weight=spec_params['spectral_weight'],
                        flow_weight=base_flow_weight,
                        flow_max_timestep=self.train_config.spectral_flow_max_timestep,
                        motion_weighted=self.train_config.spectral_flow_motion_weighted,
                        reverse_gate=self.train_config.spectral_flow_reverse_gate,
                        adaptive=self.train_config.spectral_flow_adaptive,
                        current_flow_weight=expert_flow_weight,
                    )

                    # Issue #1 fix: per-expert flow deviation tracking
                    if expert not in self.flow_deviation_history:
                        self.flow_deviation_history[expert] = []
                    # Only record when flow was actually evaluated. When the timestep
                    # gate is 0 for the whole batch (or there is no flow data / only
                    # one latent frame), the loss functions return a plain constant 0
                    # (requires_grad=False). Appending those zeros would dilute the
                    # moving average and drive the adaptive weight down as if motion
                    # were already consistent.
                    if flow_component.requires_grad:
                        self.flow_deviation_history[expert].append(flow_dev)

                    # Issue #1 fix: per-expert adaptive weight adjustment
                    if self.train_config.spectral_flow_adaptive:
                        expert_history = self.flow_deviation_history[expert]
                        if len(expert_history) > 50:
                            import numpy as np
                            recent_avg = np.mean(expert_history[-50:])
                            threshold = self.train_config.spectral_flow_rejection_threshold

                            if expert not in self.current_flow_weight:
                                self.current_flow_weight[expert] = base_flow_weight

                            if recent_avg > threshold:
                                self.current_flow_weight[expert] = min(
                                    self.current_flow_weight[expert] * 1.2,
                                    base_flow_weight * 5.0
                                )
                            elif recent_avg < threshold * 0.3:
                                self.current_flow_weight[expert] = max(
                                    self.current_flow_weight[expert] * 0.95,
                                    base_flow_weight * 0.1
                                )

                    # Issue #1 fix: per-expert rejection budget
                    # NOTE: If gradient projection is enabled, DON'T detach flow_component
                    # because we need its gradient for projection.
                    if expert not in self.flow_rejection_count:
                        self.flow_rejection_count[expert] = 0
                    max_rejections = self.train_config.spectral_flow_max_rejections
                    if (flow_dev > self.train_config.spectral_flow_rejection_threshold
                            and self.flow_rejection_count[expert] < max_rejections):
                        self.flow_rejection_count[expert] += 1
                        if self.accelerator.is_main_process:
                            print_acc(f"[FLOW REJECT] Expert={expert} Deviation={flow_dev:.4f} > "
                                      f"{self.train_config.spectral_flow_rejection_threshold}. "
                                      f"Rejecting step {self.flow_rejection_count[expert]}/{max_rejections}")
                        # Only detach flow gradients if gradient projection is NOT enabled
                        # Previously this did total_loss.detach() which killed ALL learning.
                        if not self.train_config.spectral_flow_gradient_projection_enabled:
                            flow_component = flow_component.detach()

                    # Bug 2.1 fix: apply mask BEFORE mean reduction, with proper time dim
                    # Build per-video expansion of mask_multiplier (B,1,H,W) -> (B,C,T,H,W) for video
                    loss_multiplier_batch = mask_multiplier
                    if len(noise_pred.shape) == 5:
                        # video B,C,T,H,W — expand mask to match
                        loss_multiplier_batch = loss_multiplier_batch.unsqueeze(2)
                        loss_multiplier_batch = loss_multiplier_batch.repeat(
                            1, 1, noise_pred.shape[2], 1, 1
                        )

                    # BUG FIX (issue #2): Apply per-pixel mask AND I2V conditioning mask ONLY to
                    # spectral_component (spatially structured tensor). The flow_component is already
                    # a fully-reduced scalar (MSE across all frames/pixels). Broadcasting it to
                    # (B,C,T,H,W) and then applying scale_loss would zero its contribution in first-frame
                    # masked regions and renormalize by masked mean — incorrectly attenuating its weight.
                    # So: mask+scale spectral only, reduce it to (B,), THEN add flow_component.

                    # Apply mask to spectral component
                    loss = spectral_component * loss_multiplier_batch

                    # Apply model-specific loss scaling (e.g., I2V conditioning mask via _i2v_loss_mask).
                    # Must be per-element for Wan22 I2V: only generated frames contribute.
                    loss = self.sd.scale_loss(loss)

                    # Reduce spectral loss to (B,)
                    if len(noise_pred.shape) == 5:
                        loss = loss.mean([1, 2, 3, 4])  # (B,)
                    else:
                        loss = loss.mean([1, 2, 3])     # (B,)

                    # Apply per-batch loss_multiplier (reg weight)
                    loss = loss * loss_multiplier

                    # Per-expert loss logging
                    self.additional_logs[f'loss_{expert}/spatial'] = spatial_val
                    self.additional_logs[f'loss_{expert}/flow'] = flow_val
                    self.additional_logs[f'loss_{expert}/flow_weight'] = expert_flow_weight
                    self.additional_logs[f'loss_{expert}/flow_rejections'] = self.flow_rejection_count.get(expert, 0)
                    self.additional_logs[f'loss_{expert}/step_rejections'] = self.step_rejection_count.get(expert, 0)
                    
                    # Gradient projection stats logging (per-step, not cumulative)
                    if self.train_config.spectral_flow_gradient_projection_enabled:
                        self.additional_logs['grad_proj/step_conflicts'] = self.gradient_projection_stats['step_conflicts']
                        self.additional_logs['grad_proj/step_projections'] = self.gradient_projection_stats['step_projections']
                        self.additional_logs['grad_proj/total_conflicts'] = self.gradient_projection_stats['total_conflicts']
                        self.additional_logs['grad_proj/total_projections'] = self.gradient_projection_stats['total_projections']

                    # Track per-expert losses separately for step rejection
                    expert_spatial_val = loss.mean().item()
                    expert_flow_val = flow_component.item()
                    expert_total_val = expert_spatial_val + expert_flow_val
                    
                    if expert not in self.current_step_expert_loss:
                        self.current_step_expert_loss[expert] = 0.0
                        self.current_step_expert_spatial[expert] = 0.0
                        self.current_step_expert_flow[expert] = 0.0
                    
                    self.current_step_expert_loss[expert] += expert_total_val
                    self.current_step_expert_spatial[expert] += expert_spatial_val
                    self.current_step_expert_flow[expert] += expert_flow_val

                    # SNR weighting on (B,) loss, THEN final mean
                    if not self.train_config.train_turbo:
                        if self.train_config.learnable_snr_gos:
                            loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
                        elif (self.train_config.snr_gamma is not None and
                              self.train_config.snr_gamma > 0.000001 and not ignore_snr):
                            loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler,
                                                    self.train_config.snr_gamma, fixed=True)
                        elif (self.train_config.min_snr_gamma is not None and
                              self.train_config.min_snr_gamma > 0.000001 and not ignore_snr):
                            loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler,
                                                    self.train_config.min_snr_gamma)

                    loss = loss.mean()  # scalar spectral loss after SNR

                    # Store separate loss tensors for gradient projection (if enabled)
                    # These are the final scalar losses for each component with computation graph intact
                    if self.train_config.spectral_flow_gradient_projection_enabled:
                        self._spectral_loss_tensor = loss
                        self._flow_loss_tensor = flow_component

                    # Combine with flow component (scalar, already fully reduced).
                    # Added here after all per-pixel masking/scaling so its weight is not
                    # attenuated by I2V conditioning masks or renormalization.
                    loss = loss + flow_component

                    # Check for audio loss
                    if batch.audio_pred is not None and batch.audio_target is not None:
                        audio_loss = torch.nn.functional.mse_loss(
                            batch.audio_pred.float(), batch.audio_target.float(), reduction="mean"
                        )
                        audio_loss = audio_loss * self.train_config.audio_loss_multiplier
                        loss = loss + audio_loss

                    # Check for additional losses from adapter
                    if (self.adapter is not None and hasattr(self.adapter, "additional_loss")
                            and self.adapter.additional_loss is not None):
                        loss = loss + self.adapter.additional_loss.mean()
                        self.adapter.additional_loss = None

                    if self.train_config.target_norm_std:
                        pred_std = noise_pred.std([2, 3], keepdim=True)
                        norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
                        loss = loss + norm_std_loss

                    loss = loss + additional_loss

                    if hasattr(self.sd, "get_additional_loss"):
                        additional_model_loss = self.sd.get_additional_loss(pred, target)
                        if additional_model_loss is not None:
                            loss = loss + additional_model_loss
                            self.additional_logs["additional_model_loss"] = additional_model_loss.item()

                    if self.train_config.max_loss_debug and self.train_config.max_loss is not None:
                        if loss.item() > self.train_config.max_loss:
                            print_acc(f"Loss {loss.item()} is greater than max loss {self.train_config.max_loss}. "
                                      f"Clipping to max loss.")
                            print_acc(f"timesteps: {timesteps}")

                    if self.train_config.max_loss is not None:
                        loss = torch.clamp(loss, max=self.train_config.max_loss)

                    # Skip the rest of the function for spectral_flow
                    return loss
            elif self.train_config.loss_type == "mse_spectral_flow":
                # Combined MSE + spectral (spatial frequency) + optical flow (temporal motion) loss

                # BUG FIX: guard against x0_pred mode (e.g. turbo or any x0-predicting model).
                # mse_spectral_flow loss assumes velocity prediction (model_pred = ε - x0), so
                # x0 reconstruction uses: pred_latents = noise - model_pred.
                if getattr(self.sd, 'x0_pred', False):
                    if self.accelerator.is_main_process:
                        print_acc("[WARN] mse_spectral_flow loss is incompatible with x0_pred mode. "
                                  "Falling back to spectral loss.")
                    # Get per-expert spectral parameters with timestep range override
                    # Use first timestep (all timesteps in batch are identical in flow-matching)
                    current_t = timesteps[0].item() if isinstance(timesteps, torch.Tensor) else float(timesteps)
                    spec_params = self._get_expert_spectral_params_with_override(current_t)
                    lcr_weight = self._get_lcr_weight_with_override(current_t)
                    loss = spectral_loss(
                        pred=pred,
                        latents=batch.latents,
                        noise=noise,
                        low_weight=spec_params['low_weight'],
                        mid_weight=spec_params['mid_weight'],
                        high_weight=spec_params['high_weight'],
                        low_cutoff=spec_params['low_cutoff'],
                        high_cutoff=spec_params['high_cutoff'],
                        use_phase=self.train_config.spectral_use_phase,
                        lcr_weight=lcr_weight,
                        spectral_transform=self.train_config.spectral_transform,
                        prediction_target=self.train_config.prediction_target,
                        temporal_scale=spec_params['temporal_scale'],
                    )
                else:
                    vae_ts = self.flow_loss_module.vae_temporal_stride if self.flow_loss_module else 4
                    vae_ss = self.flow_loss_module.vae_spatial_stride if self.flow_loss_module else 8

                    expert = self._get_active_expert_label()

                    # Use timestep-aware weight getters with range override support
                    # Use first timestep (all timesteps in batch are identical in flow-matching)
                    current_t = timesteps[0].item() if isinstance(timesteps, torch.Tensor) else float(timesteps)
                    base_mse_weight = self._get_mse_weight_with_override(current_t)
                    base_flow_weight = self._get_flow_weight_with_override(current_t)

                    # Issue #1 fix: use per-expert current_flow_weight (adaptive adjustment)
                    expert_flow_weight = self.current_flow_weight.get(expert, base_flow_weight)

                    # Get per-expert spectral parameters with timestep range override
                    spec_params = self._get_expert_spectral_params_with_override(current_t)
                    lcr_weight = self._get_lcr_weight_with_override(current_t)

                    # Log the flow gate mean (effective flow dilution factor)
                    self._update_flow_gate_log(timesteps)

                    (total_loss, flow_dev, mse_val, spatial_val, flow_val,
                     mse_component, spectral_component, flow_component) = mse_spectral_flow_loss(
                        model_pred=pred,
                        latents=batch.latents,
                        noise=noise,
                        batch_flow=getattr(batch, 'flow', None),
                        timesteps=timesteps,
                        flow_loss_module=self.flow_loss_module,
                        vae_temporal_stride=vae_ts,
                        vae_spatial_stride=vae_ss,
                        mse_weight=base_mse_weight,
                        low_weight=spec_params['low_weight'],
                        mid_weight=spec_params['mid_weight'],
                        high_weight=spec_params['high_weight'],
                        low_cutoff=spec_params['low_cutoff'],
                        high_cutoff=spec_params['high_cutoff'],
                        use_phase=self.train_config.spectral_use_phase,
                        lcr_weight=lcr_weight,
                        spectral_transform=self.train_config.spectral_transform,
                        prediction_target=self.train_config.prediction_target,
                        temporal_scale=spec_params['temporal_scale'],
                        spectral_weight=spec_params['spectral_weight'],
                        flow_weight=base_flow_weight,
                        flow_max_timestep=self.train_config.spectral_flow_max_timestep,
                        motion_weighted=self.train_config.spectral_flow_motion_weighted,
                        reverse_gate=self.train_config.spectral_flow_reverse_gate,
                        adaptive=self.train_config.spectral_flow_adaptive,
                        current_flow_weight=expert_flow_weight,
                    )

                    # Issue #1 fix: per-expert flow deviation tracking
                    if expert not in self.flow_deviation_history:
                        self.flow_deviation_history[expert] = []
                    # Only record when flow was actually evaluated (see the
                    # spectral_flow branch above for the rationale).
                    if flow_component.requires_grad:
                        self.flow_deviation_history[expert].append(flow_dev)

                    # Issue #1 fix: per-expert adaptive weight adjustment
                    if self.train_config.spectral_flow_adaptive:
                        if expert not in self.current_flow_weight:
                            self.current_flow_weight[expert] = base_flow_weight

                        # Use last N steps to compute moving average
                        window = min(len(self.flow_deviation_history[expert]), 20)
                        recent = self.flow_deviation_history[expert][-window:]
                        recent_avg = sum(recent) / len(recent) if recent else 0

                        # Increase weight if flow deviation is too high (motion not consistent)
                        # Decrease weight if flow deviation is very low (already consistent)
                        threshold = self.train_config.spectral_flow_rejection_threshold

                        if expert not in self.current_flow_weight:
                            self.current_flow_weight[expert] = base_flow_weight

                        if recent_avg > threshold:
                            self.current_flow_weight[expert] = min(
                                self.current_flow_weight[expert] * 1.2,
                                base_flow_weight * 5.0
                            )
                        elif recent_avg < threshold * 0.3:
                            self.current_flow_weight[expert] = max(
                                self.current_flow_weight[expert] * 0.95,
                                base_flow_weight * 0.1
                            )

                    # Issue #1 fix: per-expert rejection budget
                    # NOTE: If gradient projection is enabled, DON'T detach flow_component
                    # because we need its gradient for projection.
                    if expert not in self.flow_rejection_count:
                        self.flow_rejection_count[expert] = 0
                    max_rejections = self.train_config.spectral_flow_max_rejections
                    if (flow_dev > self.train_config.spectral_flow_rejection_threshold
                            and self.flow_rejection_count[expert] < max_rejections):
                        self.flow_rejection_count[expert] += 1
                        if self.accelerator.is_main_process:
                            print_acc(f"[FLOW REJECT] Expert={expert} Deviation={flow_dev:.4f} > "
                                      f"{self.train_config.spectral_flow_rejection_threshold}. "
                                      f"Rejecting step {self.flow_rejection_count[expert]}/{max_rejections}")
                        # Only detach flow gradients if gradient projection is NOT enabled
                        if not self.train_config.mse_spectral_flow_gradient_projection_enabled:
                            flow_component = flow_component.detach()

                    # Bug 2.1 fix: apply mask BEFORE mean reduction, with proper time dim
                    # Build per-video expansion of mask_multiplier (B,1,H,W) -> (B,C,T,H,W) for video
                    loss_multiplier_batch = mask_multiplier
                    if len(noise_pred.shape) == 5:
                        # video B,C,T,H,W — expand mask to match
                        loss_multiplier_batch = loss_multiplier_batch.unsqueeze(2)
                        loss_multiplier_batch = loss_multiplier_batch.repeat(
                            1, 1, noise_pred.shape[2], 1, 1
                        )

                    # Apply mask to all three components
                    mse_loss = mse_component * loss_multiplier_batch
                    spectral_loss = spectral_component * loss_multiplier_batch
                    flow_loss = flow_component * loss_multiplier_batch

                    # Apply model-specific loss scaling (e.g., I2V conditioning mask via _i2v_loss_mask).
                    mse_loss = self.sd.scale_loss(mse_loss)
                    spectral_loss = self.sd.scale_loss(spectral_loss)
                    flow_loss = self.sd.scale_loss(flow_loss)

                    # Reduce each component to (B,)
                    if len(noise_pred.shape) == 5:
                        mse_loss = mse_loss.mean([1, 2, 3, 4])  # (B,)
                        spectral_loss = spectral_loss.mean([1, 2, 3, 4])  # (B,)
                        flow_loss = flow_loss.mean([1, 2, 3, 4])  # (B,)
                    else:
                        mse_loss = mse_loss.mean([1, 2, 3])     # (B,)
                        spectral_loss = spectral_loss.mean([1, 2, 3])     # (B,)
                        flow_loss = flow_loss.mean([1, 2, 3])     # (B,)

                    # Apply per-batch loss_multiplier (reg weight)
                    mse_loss = mse_loss * loss_multiplier
                    spectral_loss = spectral_loss * loss_multiplier
                    flow_loss = flow_loss * loss_multiplier

                    # Per-expert loss logging
                    self.additional_logs[f'loss_{expert}/mse'] = mse_val
                    self.additional_logs[f'loss_{expert}/spatial'] = spatial_val
                    self.additional_logs[f'loss_{expert}/flow'] = flow_val
                    self.additional_logs[f'loss_{expert}/flow_weight'] = expert_flow_weight
                    self.additional_logs[f'loss_{expert}/mse_weight'] = base_mse_weight
                    self.additional_logs[f'loss_{expert}/flow_rejections'] = self.flow_rejection_count.get(expert, 0)
                    self.additional_logs[f'loss_{expert}/step_rejections'] = self.step_rejection_count.get(expert, 0)
                    
                    # Gradient projection stats logging (per-step, not cumulative)
                    if self.train_config.mse_spectral_flow_gradient_projection_enabled:
                        self.additional_logs['grad_proj/step_conflicts'] = self.gradient_projection_stats['step_conflicts']
                        self.additional_logs['grad_proj/step_projections'] = self.gradient_projection_stats['step_projections']
                        self.additional_logs['grad_proj/total_conflicts'] = self.gradient_projection_stats['total_conflicts']
                        self.additional_logs['grad_proj/total_projections'] = self.gradient_projection_stats['total_projections']

                    # Track per-expert losses separately for step rejection
                    expert_mse_val = mse_loss.mean().item()
                    expert_spatial_val = spectral_loss.mean().item()
                    expert_flow_val = flow_loss.mean().item()
                    expert_total_val = expert_mse_val + expert_spatial_val + expert_flow_val
                    
                    if expert not in self.current_step_expert_loss:
                        self.current_step_expert_loss[expert] = 0.0
                        self.current_step_expert_spatial[expert] = 0.0
                        self.current_step_expert_flow[expert] = 0.0
                        self.current_step_expert_mse[expert] = 0.0
                    
                    self.current_step_expert_loss[expert] += expert_total_val
                    self.current_step_expert_spatial[expert] += expert_spatial_val
                    self.current_step_expert_flow[expert] += expert_flow_val
                    self.current_step_expert_mse[expert] += expert_mse_val

                    # SNR weighting on (B,) loss, THEN final mean
                    if not self.train_config.train_turbo:
                        if self.train_config.learnable_snr_gos:
                            mse_loss = apply_learnable_snr_gos(mse_loss, timesteps, self.snr_gos)
                            spectral_loss = apply_learnable_snr_gos(spectral_loss, timesteps, self.snr_gos)
                        elif (self.train_config.snr_gamma is not None and
                              self.train_config.snr_gamma > 0.000001 and not ignore_snr):
                            mse_loss = apply_snr_weight(mse_loss, timesteps, self.sd.noise_scheduler,
                                                        self.train_config.snr_gamma, fixed=True)
                            spectral_loss = apply_snr_weight(spectral_loss, timesteps, self.sd.noise_scheduler,
                                                             self.train_config.snr_gamma, fixed=True)
                        elif (self.train_config.min_snr_gamma is not None and
                              self.train_config.min_snr_gamma > 0.000001 and not ignore_snr):
                            mse_loss = apply_snr_weight(mse_loss, timesteps, self.sd.noise_scheduler,
                                                        self.train_config.min_snr_gamma)
                            spectral_loss = apply_snr_weight(spectral_loss, timesteps, self.sd.noise_scheduler,
                                                             self.train_config.min_snr_gamma)

                    loss_mse = mse_loss.mean()  # scalar MSE loss after SNR
                    loss_spatial = spectral_loss.mean()  # scalar spectral loss after SNR

                    # Store separate loss tensors for gradient projection (if enabled)
                    # These are the final scalar losses for each component with computation graph intact
                    if self.train_config.mse_spectral_flow_gradient_projection_enabled:
                        self._mse_loss_tensor = loss_mse
                        self._spectral_loss_tensor = loss_spatial
                        self._flow_loss_tensor = flow_loss.mean()

                    # Combine all three components (scalar, already fully reduced).
                    # Added here after all per-pixel masking/scaling so weights are not
                    # attenuated by I2V conditioning masks or renormalization.
                    loss = loss_mse + loss_spatial + flow_loss.mean()

                    # Check for audio loss
                    if batch.audio_pred is not None and batch.audio_target is not None:
                        audio_loss = torch.nn.functional.mse_loss(
                            batch.audio_pred.float(), batch.audio_target.float(), reduction="mean"
                        )
                        audio_loss = audio_loss * self.train_config.audio_loss_multiplier
                        loss = loss + audio_loss

                    # Check for additional losses from adapter
                    if (self.adapter is not None and hasattr(self.adapter, "additional_loss")
                            and self.adapter.additional_loss is not None):
                        loss = loss + self.adapter.additional_loss.mean()
                        self.adapter.additional_loss = None

                    if self.train_config.target_norm_std:
                        pred_std = noise_pred.std([2, 3], keepdim=True)
                        norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
                        loss = loss + norm_std_loss

                    loss = loss + additional_loss

                    if hasattr(self.sd, "get_additional_loss"):
                        additional_model_loss = self.sd.get_additional_loss(pred, target)
                        if additional_model_loss is not None:
                            loss = loss + additional_model_loss
                            self.additional_logs["additional_model_loss"] = additional_model_loss.item()

                    if self.train_config.max_loss_debug and self.train_config.max_loss is not None:
                        if loss.item() > self.train_config.max_loss:
                            print_acc(f"Loss {loss.item()} is greater than max loss {self.train_config.max_loss}. "
                                      f"Clipping to max loss.")
                            print_acc(f"timesteps: {timesteps}")

                    if self.train_config.max_loss is not None:
                        loss = torch.clamp(loss, max=self.train_config.max_loss)

                    # Skip the rest of the function for mse_spectral_flow
                    return loss
            elif self.train_config.loss_type == "stepped":
                loss = stepped_loss(pred, batch.latents, noise, noisy_latents, timesteps, self.sd.noise_scheduler)
                # the way this loss works, it is low, increase it to match predictable LR effects
                loss = loss * 10.0
            else:
                loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
            
            loss = loss * local_loss_scale
            
            # apply model specific loss scaling
            loss = self.sd.scale_loss(loss)
                
            do_weighted_timesteps = False
            if self.sd.is_flow_matching:
                if self.train_config.linear_timesteps or self.train_config.linear_timesteps2:
                    do_weighted_timesteps = True
                if self.train_config.timestep_type == "weighted":
                    # use the noise scheduler to get the weights for the timesteps
                    do_weighted_timesteps = True

            # handle linear timesteps and only adjust the weight of the timesteps
            if do_weighted_timesteps:
                # calculate the weights for the timesteps
                timestep_weight = self.sd.noise_scheduler.get_weights_for_timesteps(
                    timesteps,
                    v2=self.train_config.linear_timesteps2,
                    timestep_type=self.train_config.timestep_type
                ).to(loss.device, dtype=loss.dtype)
                if len(loss.shape) == 4:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1).detach()
                elif len(loss.shape) == 5:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1, 1).detach()
                loss = loss * timestep_weight

        if self.train_config.do_prior_divergence and prior_pred is not None:
            loss = loss + (torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none") * -1.0)

        if self.train_config.train_turbo:
            mask_multiplier = mask_multiplier[:, 3:, :, :]
            # resize to the size of the loss
            mask_multiplier = torch.nn.functional.interpolate(mask_multiplier, size=(pred.shape[2], pred.shape[3]), mode='nearest')

        # multiply by our mask
        try:
            if len(noise_pred.shape) == 5:
                # video B,C,T,H,W
                mask_multiplier = mask_multiplier.unsqueeze(2)  # add time dimension back for video
                mask_multiplier = mask_multiplier.repeat(1, 1, noise_pred.shape[2], 1, 1)
            loss = loss * mask_multiplier
        except Exception as e:
            # todo handle mask with video models
            print("Could not apply mask multiplier to loss")
            print(e)
            pass

        prior_loss = None
        if self.train_config.inverted_mask_prior and prior_pred is not None and prior_mask_multiplier is not None:
            assert not self.train_config.train_turbo
            if self.train_config.loss_type == "mae":
                prior_loss = torch.nn.functional.l1_loss(pred.float(), prior_pred.float(), reduction="none")
            else:
                prior_loss = torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none")

            prior_loss = prior_loss * prior_mask_multiplier * self.train_config.inverted_mask_prior_multiplier
            if not torch.isfinite(prior_loss).all():
                print_acc("Prior loss is nan")
                prior_loss = None
            else:
                if len(noise_pred.shape) == 5:
                    # video B,C,T,H,W
                    prior_loss = prior_loss.mean([1, 2, 3, 4])
                else:
                    prior_loss = prior_loss.mean([1, 2, 3])
                # loss = loss + prior_loss
                # loss = loss + prior_loss
            # loss = loss + prior_loss
        if len(noise_pred.shape) == 5:
            loss = loss.mean([1, 2, 3, 4])
        else:
            loss = loss.mean([1, 2, 3])
        # apply loss multiplier before prior loss
        # multiply by our mask
        try:
            loss = loss * loss_multiplier
        except:
            # todo handle mask with video models
            pass
        if prior_loss is not None:
            loss = loss + prior_loss

        if not self.train_config.train_turbo:
            if self.train_config.learnable_snr_gos:
                # add snr_gamma
                loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
            elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001 and not ignore_snr:
                # add snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma,
                                        fixed=True)
            elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001 and not ignore_snr:
                # add min_snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        loss = loss.mean()
        
        # check for audio loss
        if batch.audio_pred is not None and batch.audio_target is not None:
            audio_loss = torch.nn.functional.mse_loss(batch.audio_pred.float(), batch.audio_target.float(), reduction="mean")
            audio_loss = audio_loss * self.train_config.audio_loss_multiplier
            loss = loss + audio_loss

        # check for additional losses
        if self.adapter is not None and hasattr(self.adapter, "additional_loss") and self.adapter.additional_loss is not None:

            loss = loss + self.adapter.additional_loss.mean()
            self.adapter.additional_loss = None

        if self.train_config.target_norm_std:
            # seperate out the batch and channels
            pred_std = noise_pred.std([2, 3], keepdim=True)
            norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
            loss = loss + norm_std_loss


        loss = loss + additional_loss
        
        if hasattr(self.sd, "get_additional_loss"):
            additional_model_loss = self.sd.get_additional_loss(pred, target)
            if additional_model_loss is not None:
                loss = loss + additional_model_loss
                self.additional_logs["additional_model_loss"] = additional_model_loss.item()

        if self.train_config.max_loss_debug and self.train_config.max_loss is not None:
            if loss.item() > self.train_config.max_loss:
                print_acc(f"Loss {loss.item()} is greater than max loss {self.train_config.max_loss}. Clipping to max loss.")
                print_acc(f"timesteps: {timesteps}")

        if self.train_config.max_loss is not None:
            loss = torch.clamp(loss, max=self.train_config.max_loss)
        
        return loss

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        loss = get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            unconditional_embeds=unconditional_embeds,
            train_config=self.train_config,
            **kwargs
        )

        return loss
    
    
    # ------------------------------------------------------------------
    #  Mean-Flow loss (Geng et al., “Mean Flows for One-step Generative
    #  Modelling”, 2025 – see Alg. 1 + Eq. (6) of the paper)
    # This version avoids jvp / double-back-prop issues with Flash-Attention
    # adapted from the work of lodestonerock
    # ------------------------------------------------------------------
    def get_mean_flow_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        total_steps = float(self.sd.noise_scheduler.config.num_train_timesteps)  # e.g. 1000
        base_eps = 1e-3
        min_time_gap = 1e-2
        
        with torch.no_grad():
            num_train_timesteps = self.sd.noise_scheduler.config.num_train_timesteps
            batch_size = batch.latents.shape[0]
            timestep_t_list = []
            timestep_r_list = []

            for i in range(batch_size):
                t1 = random.randint(0, num_train_timesteps - 1)
                t2 = random.randint(0, num_train_timesteps - 1)
                t_t = self.sd.noise_scheduler.timesteps[min(t1, t2)]
                t_r = self.sd.noise_scheduler.timesteps[max(t1, t2)]
                if (t_t - t_r).item() < min_time_gap * 1000:
                    scaled_time_gap = min_time_gap * 1000
                    if t_t.item() + scaled_time_gap > 1000:
                        t_r = t_r - scaled_time_gap
                    else:
                        t_t = t_t + scaled_time_gap
                timestep_t_list.append(t_t)
                timestep_r_list.append(t_r)

            timesteps_t = torch.stack(timestep_t_list, dim=0).float()
            timesteps_r = torch.stack(timestep_r_list, dim=0).float()

            t_frac = timesteps_t / total_steps  # [0,1]
            r_frac = timesteps_r / total_steps  # [0,1]

            latents_clean = batch.latents.to(dtype)
            noise_sample = noise.to(dtype)

            lerp_vector = latents_clean * (1.0 - t_frac[:, None, None, None]) + noise_sample * t_frac[:, None, None, None]

            eps = base_eps

            # concatenate timesteps as input for u(z, r, t)
            timesteps_cat = torch.cat([t_frac, r_frac], dim=0) * total_steps

        # model predicts u(z, r, t)
        u_pred = self.predict_noise(
            noisy_latents=lerp_vector.to(dtype),
            timesteps=timesteps_cat.to(dtype),
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            batch=batch,
            **pred_kwargs
        )

        with torch.no_grad():
            t_frac_plus_eps = (t_frac + eps).clamp(0.0, 1.0)
            lerp_perturbed = latents_clean * (1.0 - t_frac_plus_eps[:, None, None, None]) + noise_sample * t_frac_plus_eps[:, None, None, None]
            timesteps_cat_perturbed = torch.cat([t_frac_plus_eps, r_frac], dim=0) * total_steps

            u_perturbed = self.predict_noise(
                noisy_latents=lerp_perturbed.to(dtype),
                timesteps=timesteps_cat_perturbed.to(dtype),
                conditional_embeds=conditional_embeds,
                unconditional_embeds=unconditional_embeds,
                batch=batch,
                **pred_kwargs
            )

        # compute du/dt via finite difference (detached)
        du_dt = (u_perturbed - u_pred).detach() / eps
        # du_dt = (u_perturbed - u_pred).detach()
        du_dt = du_dt.to(dtype)
        
        
        time_gap = (t_frac - r_frac)[:, None, None, None].to(dtype)
        time_gap.clamp(min=1e-4)
        u_shifted = u_pred + time_gap * du_dt
        # u_shifted = u_pred + du_dt / time_gap
        # u_shifted = u_pred

        # a step is done like this:
        # stepped_latent = model_input + (timestep_next - timestep) * model_output
        
        # flow target velocity
        # v_target = (noise_sample - latents_clean) / time_gap
        # flux predicts opposite of velocity, so we need to invert it
        v_target = (latents_clean - noise_sample) / time_gap

        # compute loss
        loss = torch.nn.functional.mse_loss(
            u_shifted.float(),
            v_target.float(),
            reduction='none'
        )

        with torch.no_grad():
            pure_loss = loss.mean().detach()
            pure_loss.requires_grad_(True)

        loss = loss.mean()
        if loss.item() > 1e3:
            pass
        self.accelerator.backward(loss)
        return pure_loss



    def get_prior_prediction(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            conditioned_prompts=None,
            **kwargs
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        can_disable_adapter = False
        was_adapter_active = False
        if self.adapter is not None and (isinstance(self.adapter, IPAdapter) or
                                         isinstance(self.adapter, ReferenceAdapter) or
                                         (isinstance(self.adapter, CustomAdapter))
        ):
            can_disable_adapter = True
            was_adapter_active = self.adapter.is_active
            self.adapter.is_active = False

        if self.train_config.unload_text_encoder and self.adapter is not None and not isinstance(self.adapter, CustomAdapter):
            raise ValueError("Prior predictions currently do not support unloading text encoder with adapter")
        # do a prediction here so we can match its output with network multiplier set to 0.0
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)

            embeds_to_use = conditional_embeds.clone().detach()
            # handle clip vision adapter by removing triggers from prompt and replacing with the class name
            if (self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter)) or self.embedding is not None:
                prompt_list = batch.get_caption_list()
                class_name = ''

                triggers = ['[trigger]', '[name]']
                remove_tokens = []

                if self.embed_config is not None:
                    triggers.append(self.embed_config.trigger)
                    for i in range(1, self.embed_config.tokens):
                        remove_tokens.append(f"{self.embed_config.trigger}_{i}")
                    if self.embed_config.trigger_class_name is not None:
                        class_name = self.embed_config.trigger_class_name

                if self.adapter is not None:
                    triggers.append(self.adapter_config.trigger)
                    for i in range(1, self.adapter_config.num_tokens):
                        remove_tokens.append(f"{self.adapter_config.trigger}_{i}")
                    if self.adapter_config.trigger_class_name is not None:
                        class_name = self.adapter_config.trigger_class_name

                for idx, prompt in enumerate(prompt_list):
                    for remove_token in remove_tokens:
                        prompt = prompt.replace(remove_token, '')
                    for trigger in triggers:
                        prompt = prompt.replace(trigger, class_name)
                    prompt_list[idx] = prompt

                if batch.prompt_embeds is not None:
                    embeds_to_use = batch.prompt_embeds.clone().to(self.device_torch, dtype=dtype)
                else:
                    prompt_kwargs = {}
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    embeds_to_use = self.sd.encode_prompt(
                        prompt_list,
                        long_prompts=self.do_long_prompts).to(
                        self.device_torch,
                        dtype=dtype,
                        **prompt_kwargs
                    ).detach()

            # dont use network on this
            # self.network.multiplier = 0.0
            self.sd.unet.eval()

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not self.sd.is_flux and not self.sd.is_lumina2:
                # we need to remove the image embeds from the prompt except for flux
                embeds_to_use: PromptEmbeds = embeds_to_use.clone().detach()
                end_pos = embeds_to_use.text_embeds.shape[1] - self.adapter_config.num_tokens
                embeds_to_use.text_embeds = embeds_to_use.text_embeds[:, :end_pos, :]
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.clone().detach()
                    unconditional_embeds.text_embeds = unconditional_embeds.text_embeds[:, :end_pos]

            if unconditional_embeds is not None:
                unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
            
            guidance_embedding_scale = self.train_config.cfg_scale
            if self.train_config.do_guidance_loss:
                guidance_embedding_scale = self._guidance_loss_target_batch

            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                unconditional_embeddings=unconditional_embeds,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                guidance_embedding_scale=guidance_embedding_scale,
                rescale_cfg=self.train_config.cfg_rescale,
                batch=batch,
                **pred_kwargs  # adapter residuals in here
            )
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']
            if match_adapter_assist and 'down_block_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_block_additional_residuals']
            if match_adapter_assist and 'mid_block_additional_residual' in pred_kwargs:
                del pred_kwargs['mid_block_additional_residual']

            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def save(self, step=None):
        """
        Override save to finalize rank gates before saving.
        
        Gates are finalized (hardened to binary {0,1}) ONLY right before the final save
        (step=None), and only if final_hardening is enabled in config.
        This ensures both experts are finalized regardless of which was last active.
        """
        super().save(step)
        
        # Finalize gates ONLY on the final save (step=None) and only if configured.
        # Run AFTER super().save() so that under DDP, only main_process finalizes.
        # (In practice gates are identical across ranks, but this is cleaner.)
        if (not hasattr(self, 'accelerator') or self.accelerator.is_main_process):
            if (step is None and
                self.rank_gates_scheduler is not None and
                self.network is not None and
                self.network_config is not None and
                self.network_config.rank_gates is not None and
                self.network_config.rank_gates.final_hardening):
                gated_loras = getattr(self.network, 'gated_loras', None)
                if gated_loras:
                    remaining = [gl for gl in gated_loras if not gl.is_hardened()]
                    if remaining:
                        finalize_gates(remaining)
                        print(f"\n[RankGates] Finalized {len(remaining)} gate sets before final save")

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: Union[int, torch.Tensor] = 1,
        conditional_embeds: Union[PromptEmbeds, None] = None,
        unconditional_embeds: Union[PromptEmbeds, None] = None,
        batch: Optional['DataLoaderBatchDTO'] = None,
        is_primary_pred: bool = False,
        **kwargs,
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        guidance_embedding_scale = self.train_config.cfg_scale
        if self.train_config.do_guidance_loss:
            guidance_embedding_scale = self._guidance_loss_target_batch
        # Pass per-item image-conditioning masks (from the conditioning-dropout config)
        # to models that support them (e.g. Wan I2V). The masks encode which items keep /
        # drop the first-frame image on the positive and (when CFG is active) negative
        # branch. Models without support ignore the extra kwargs.
        if getattr(self, '_cur_image_cond_mask_pos', None) is not None:
            kwargs = dict(kwargs)
            kwargs['image_cond_mask_pos'] = self._cur_image_cond_mask_pos
            if getattr(self, '_cur_image_cond_mask_neg', None) is not None:
                kwargs['image_cond_mask_neg'] = self._cur_image_cond_mask_neg
        return self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=dtype),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
            unconditional_embeddings=unconditional_embeds,
            timestep=timesteps,
            guidance_scale=self.train_config.cfg_scale,
            guidance_embedding_scale=guidance_embedding_scale,
            detach_unconditional=False,
            rescale_cfg=self.train_config.cfg_rescale,
            bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
            batch=batch,
            **kwargs
        )

    def _apply_text_dropout_mask(
            self,
            embeds: PromptEmbeds,
            drop_mask,
    ) -> PromptEmbeds:
        """Replace the prompt embeddings of dropped samples with blank embeddings.

        Args:
            embeds: the PromptEmbeds to modify in place (and return).
            drop_mask: per-sample boolean mask (list/tuple of bool, or a torch bool
                tensor) with length == batch size. True => drop (replace with blank).
        """
        if embeds is None:
            return embeds
        batch_size = embeds.text_embeds.shape[0]
        if isinstance(drop_mask, (list, tuple)):
            drop_mask_t = torch.tensor(list(drop_mask), device=self.device_torch)
        else:
            drop_mask_t = drop_mask.to(self.device_torch).bool()
        if drop_mask_t.shape[0] != batch_size:
            # mask does not line up with the embeds (exotic batching) -> leave as-is
            return embeds
        if not drop_mask_t.any():
            return embeds

        blank = self.cached_blank_embeds.to(
            self.device_torch, dtype=embeds.text_embeds.dtype
        )
        target_seq_len = embeds.text_embeds.shape[1]

        # Pad or truncate blank text embeds to match batch sequence length
        blank_text = blank.text_embeds[0]  # (src_len, dim)
        src_len = blank_text.shape[0]
        if src_len < target_seq_len:
            blank_text = F.pad(blank_text, (0, 0, 0, target_seq_len - src_len))
        elif src_len > target_seq_len:
            blank_text = blank_text[:target_seq_len, :]

        # Replace dropped samples with blank embeddings
        drop_indices = drop_mask_t.nonzero(as_tuple=True)[0]
        num_dropped = len(drop_indices)
        embeds.text_embeds[drop_indices] = blank_text.unsqueeze(0).expand(num_dropped, -1, -1)

        if embeds.pooled_embeds is not None and blank.pooled_embeds is not None:
            embeds.pooled_embeds[drop_indices] = blank.pooled_embeds[0].unsqueeze(0).expand(num_dropped, -1)

        if embeds.attention_mask is not None:
            if blank.attention_mask is not None:
                blank_mask = blank.attention_mask[0]
                mask_len = len(blank_mask)
                if mask_len < target_seq_len:
                    blank_mask = F.pad(blank_mask, (0, target_seq_len - mask_len))
                elif mask_len > target_seq_len:
                    blank_mask = blank_mask[:target_seq_len]
                embeds.attention_mask[drop_indices] = blank_mask.unsqueeze(0).expand(num_dropped, -1)
            else:
                embeds.attention_mask[drop_indices] = 0

        return embeds

    def _apply_caption_dropout(
            self,
            embeds: PromptEmbeds,
            dropout_rate: float,
            is_i2v_modes: List[bool] = None,
            dropout_rate_t2v: float = 0.0,
    ) -> PromptEmbeds:
        """Randomly replace some samples' cached embeddings with blank embeddings.

        Legacy helper kept for compatibility. New code uses
        ``_apply_conditioning_dropout`` which unifies text + image, positive +
        negative, sync/invert, global + per-dataset rates.
        """
        batch_size = embeds.text_embeds.shape[0]
        if is_i2v_modes is not None and len(is_i2v_modes) == batch_size:
            rates = [dropout_rate if m else dropout_rate_t2v for m in is_i2v_modes]
            drop_mask = torch.rand(batch_size, device=self.device_torch) < torch.tensor(rates, device=self.device_torch)
        else:
            drop_mask = torch.rand(batch_size, device=self.device_torch) < dropout_rate
        return self._apply_text_dropout_mask(embeds, drop_mask)

    def _resolve_conditioning_dropout_rates(self, dataset_configs: List, is_i2v_modes: List[bool]):
        """Resolve the effective per-item conditioning dropout rates.

        Precedence per item (using that item's dataset config):
            per-dataset value (if not None)  >  global value (if not None)  >  fallback

        Returns (pos_text, neg_text, pos_img, neg_img) as lists of floats (len == items).
        """
        tc = self.train_config
        n = len(dataset_configs)
        pos_text, neg_text, pos_img, neg_img = [], [], [], []
        for i in range(n):
            ds = dataset_configs[i] if i < len(dataset_configs) and dataset_configs[i] is not None else None

            # --- positive text ---
            rate = getattr(ds, 'text_dropout_rate', None) if ds is not None else None
            if rate is None:
                rate = tc.text_dropout_rate
            if rate is None:
                ds_legacy = getattr(ds, 'caption_dropout_rate', 0.0) if ds is not None else 0.0
                # honor either legacy source so existing configs keep working
                rate = max(float(ds_legacy), float(tc.prompt_dropout_prob))
            pos_text.append(float(rate))

            # --- negative text (only meaningful when CFG is active) ---
            rate = getattr(ds, 'text_dropout_rate_negative', None) if ds is not None else None
            if rate is None:
                rate = tc.text_dropout_rate_negative
            neg_text.append(0.0 if rate is None else float(rate))

            # --- positive image (I2V first frame) ---
            rate = getattr(ds, 'image_dropout_rate', None) if ds is not None else None
            if rate is None:
                rate = tc.image_dropout_rate
            pos_img.append(0.0 if rate is None else float(rate))

            # --- negative image (only meaningful when CFG is active) ---
            rate = getattr(ds, 'image_dropout_rate_negative', None) if ds is not None else None
            if rate is None:
                rate = tc.image_dropout_rate_negative
            if rate is None:
                # cfg_same_prompt historically drops the image on the uncond side
                rate = 1.0 if tc.cfg_same_prompt else 0.0
            neg_img.append(float(rate))

        return pos_text, neg_text, pos_img, neg_img

    def _resolve_branch_drop(self, pos_drop: List[bool], neg_rates: List[float], sync: bool, invert: bool):
        """Resolve the negative-branch drop state from the positive state.

        sync=True  => negative mirrors the positive state (optionally inverted).
        sync=False => negative uses its own independent per-item rate.
        """
        if sync:
            state = [bool(p) for p in pos_drop]
            if invert:
                state = [not s for s in state]
            return state
        return [random.random() < r for r in neg_rates]

    def _apply_conditioning_dropout(
            self,
            batch,
            conditional_embeds: PromptEmbeds,
            unconditional_embeds: PromptEmbeds,
            item_index: int = 0,
    ):
        """Apply the configured text + image conditioning dropout for one step.

        Computes per-item drop states for the positive and (when CFG is active) the
        negative branch, applies the text dropout to the prompt embeddings, and
        returns the image-conditioning masks to hand to the model.

        Returns:
            (conditional_embeds, unconditional_embeds, pos_image_mask, neg_image_mask)
            where the image masks are lists of bool (True => apply image conditioning).
        """
        tc = self.train_config
        do_cfg = bool(tc.do_cfg)
        bs = conditional_embeds.text_embeds.shape[0] if conditional_embeds is not None else 0

        # Per-item dataset configs + i2v flags for the current (possibly chunked) batch.
        file_items = batch.file_items
        if tc.single_item_batching and len(file_items) > bs:
            ds_configs = [file_items[item_index].dataset_config]
            is_i2v = [file_items[item_index].is_i2v_mode]
        else:
            ds_configs = [fi.dataset_config for fi in file_items]
            is_i2v = [fi.is_i2v_mode for fi in file_items]
        if len(ds_configs) != bs:
            # cannot align per-item configs -> fall back to no dropout (keep image on i2v)
            fallback = [bool(m) for m in is_i2v] if len(is_i2v) == bs else [True] * bs
            return conditional_embeds, unconditional_embeds, fallback, fallback

        pos_text_rates, neg_text_rates, pos_img_rates, neg_img_rates = \
            self._resolve_conditioning_dropout_rates(ds_configs, is_i2v)

        # --- positive drop states ---
        pos_text_drop = [random.random() < r for r in pos_text_rates]
        pos_img_drop = [random.random() < r for r in pos_img_rates]

        # --- negative drop states (only when CFG is active) ---
        neg_img_drop = None
        if do_cfg:
            neg_img_drop = self._resolve_branch_drop(
                pos_img_drop, neg_img_rates, tc.sync_image_dropout, tc.invert_image_dropout)

        # The cfg_same_prompt negative branch must share the EXACT same prompt (and
        # dropout state) as the positive. The encode block's negative clone predates
        # the positive text dropout, so we rebuild it afterwards (see below). When the
        # invert toggle is on we need the non-dropped base, so keep a copy of it.
        base_embeds = None
        if do_cfg and tc.cfg_same_prompt and tc.invert_text_dropout:
            base_embeds = conditional_embeds.clone()

        # --- apply positive text dropout ---
        conditional_embeds = self._apply_text_dropout_mask(conditional_embeds, pos_text_drop)

        # --- apply negative branch text dropout ---
        if do_cfg:
            if tc.cfg_same_prompt:
                if tc.invert_text_dropout:
                    # inverted: negative uses the OPPOSITE dropout state (positive
                    # dropped => negative keeps its prompt, and vice versa). Start
                    # from the non-dropped base and drop the inverted set.
                    unconditional_embeds = base_embeds.clone()
                    unconditional_embeds = self._apply_text_dropout_mask(
                        unconditional_embeds, [not d for d in pos_text_drop])
                elif unconditional_embeds is None or any(pos_text_drop):
                    # same state as positive: refresh the negative clone so dropped
                    # prompts are dropped on BOTH branches (keeps "same prompt" intact)
                    unconditional_embeds = conditional_embeds.clone()
                # else: the encode block's clone is already identical to positive
            elif unconditional_embeds is not None:
                neg_text_drop = self._resolve_branch_drop(
                    pos_text_drop, neg_text_rates, tc.sync_text_dropout, tc.invert_text_dropout)
                unconditional_embeds = self._apply_text_dropout_mask(unconditional_embeds, neg_text_drop)

        # --- image conditioning masks (True => keep the image) ---
        pos_image_mask = [bool(i2v) and (not d) for i2v, d in zip(is_i2v, pos_img_drop)]
        neg_image_mask = None
        if do_cfg:
            neg_image_mask = [bool(i2v) and (not d) for i2v, d in zip(is_i2v, neg_img_drop)]

        return conditional_embeds, unconditional_embeds, pos_image_mask, neg_image_mask

    def train_single_accumulation(self, batch: DataLoaderBatchDTO):
        # Update softcap step counter BEFORE forward pass so logging checks
        # use the correct step number. Called here instead of at end of loop
        # because attention ops happen during forward, not after.
        if hasattr(self, 'step_num'):
            try:
                from toolkit.models.wan21.wan_attn import update_softcap_step
                expert = self._get_active_expert_label()
                update_softcap_step(self.step_num, expert=expert)
            except ImportError:
                pass  # Non-critical
        
        with torch.no_grad():
            self.timer.start('preprocess_batch')
            if isinstance(self.adapter, CustomAdapter):
                batch = self.adapter.edit_batch_raw(batch)
            batch = self.preprocess_batch(batch)
            if isinstance(self.adapter, CustomAdapter):
                batch = self.adapter.edit_batch_processed(batch)
            dtype = get_torch_dtype(self.train_config.dtype)
            # When the model's front-end runs in fp32 (TREAD fp32_front), keep the text
            # embeddings at full precision so they are not rounded to the training dtype
            # before reaching the fp32 condition embedder. (Latents are handled the same way
            # in ``process_general_training_batch``.)
            embed_dtype = torch.promote_types(dtype, self.sd.get_cache_dtype())
            # sanity check
            if self.sd.vae.dtype != self.sd.vae_torch_dtype:
                self.sd.vae = self.sd.vae.to(self.sd.vae_torch_dtype)
            if isinstance(self.sd.text_encoder, list):
                for encoder in self.sd.text_encoder:
                    if encoder.dtype != self.sd.te_torch_dtype:
                        encoder.to(self.sd.te_torch_dtype)
            else:
                if self.sd.text_encoder.dtype != self.sd.te_torch_dtype:
                    self.sd.text_encoder.to(self.sd.te_torch_dtype)

            noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
            if (self.train_config.do_cfg or self.train_config.do_random_cfg) and not self.train_config.cfg_same_prompt:
                # pick random negative prompts
                if self.negative_prompt_pool is not None:
                    negative_prompts = []
                    for i in range(noisy_latents.shape[0]):
                        num_neg = random.randint(1, self.train_config.max_negative_prompts)
                        this_neg_prompts = [random.choice(self.negative_prompt_pool) for _ in range(num_neg)]
                        this_neg_prompt = ', '.join(this_neg_prompts)
                        negative_prompts.append(this_neg_prompt)
                    self.batch_negative_prompt = negative_prompts
                else:
                    self.batch_negative_prompt = ['' for _ in range(batch.latents.shape[0])]

            if self.adapter and isinstance(self.adapter, CustomAdapter):
                # condition the prompt
                # todo handle more than one adapter image
                conditioned_prompts = self.adapter.condition_prompt(conditioned_prompts)

            network_weight_list = batch.get_network_weight_list()
            if self.train_config.single_item_batching:
                network_weight_list = network_weight_list + network_weight_list

            has_adapter_img = batch.control_tensor is not None
            has_clip_image = batch.clip_image_tensor is not None
            has_clip_image_embeds = batch.clip_image_embeds is not None
            # force it to be true if doing regs as we handle those differently
            if any([batch.file_items[idx].is_reg for idx in range(len(batch.file_items))]):
                has_clip_image = True
                if self._clip_image_embeds_unconditional is not None:
                    has_clip_image_embeds = True  # we are caching embeds, handle that differently
                    has_clip_image = False

            # do prior pred if prior regularization batch
            do_reg_prior = False
            if any([batch.file_items[idx].prior_reg for idx in range(len(batch.file_items))]):
                do_reg_prior = True

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not has_clip_image and has_adapter_img:
                raise ValueError(
                    "IPAdapter control image is now 'clip_image_path' instead of 'control_path'. Please update your dataset config ")

            match_adapter_assist = False

            # check if we are matching the adapter assistant
            if self.assistant_adapter:
                if self.train_config.match_adapter_chance == 1.0:
                    match_adapter_assist = True
                elif self.train_config.match_adapter_chance > 0.0:
                    match_adapter_assist = torch.rand(
                        (1,), device=self.device_torch, dtype=dtype
                    ) < self.train_config.match_adapter_chance

            self.timer.stop('preprocess_batch')

            is_reg = False
            loss_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            for idx, file_item in enumerate(batch.file_items):
                if file_item.is_reg:
                    loss_multiplier[idx] = loss_multiplier[idx] * self.train_config.reg_weight
                    is_reg = True

            adapter_images = None
            sigmas = None
            if has_adapter_img and (self.adapter or self.assistant_adapter):
                with self.timer('get_adapter_images'):
                    # todo move this to data loader
                    if batch.control_tensor is not None:
                        adapter_images = batch.control_tensor.to(self.device_torch, dtype=dtype).detach()
                        # match in channels
                        if self.assistant_adapter is not None:
                            in_channels = self.assistant_adapter.config.in_channels
                            if adapter_images.shape[1] != in_channels:
                                # we need to match the channels
                                adapter_images = adapter_images[:, :in_channels, :, :]
                    else:
                        raise NotImplementedError("Adapter images now must be loaded with dataloader")

            clip_images = None
            if has_clip_image:
                with self.timer('get_clip_images'):
                    # todo move this to data loader
                    if batch.clip_image_tensor is not None:
                        clip_images = batch.clip_image_tensor.to(self.device_torch, dtype=dtype).detach()

            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            if batch.mask_tensor is not None and self.sd.do_masked_loss:
                with self.timer('get_mask_multiplier'):
                    # FIXED: BF16 interpolation is fully supported in modern PyTorch (2.0+)
                    # Previous FP16 hardcoding caused precision loss and gradient instability
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=dtype).detach()
                    # scale down to the size of the latents, mask multiplier shape(bs, 1, width, height), noisy_latents shape(bs, channels, width, height)
                    if len(noisy_latents.shape) == 5:
                        # video B,C,T,H,W
                        h = noisy_latents.shape[3]
                        w = noisy_latents.shape[4]
                    else:
                        h = noisy_latents.shape[2]
                        w = noisy_latents.shape[3]
                    mask_multiplier = torch.nn.functional.interpolate(
                        mask_multiplier, size=(h, w)
                    )
                    # expand to match latents
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    # make avg 1.0
                    mask_multiplier = mask_multiplier / mask_multiplier.mean()

        def get_adapter_multiplier():
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                # training a t2i adapter, not using as assistant.
                return 1.0
            elif match_adapter_assist:
                # training a texture. We want it high
                adapter_strength_min = 0.9
                adapter_strength_max = 1.0
            else:
                # training with assistance, we want it low
                # adapter_strength_min = 0.4
                # adapter_strength_max = 0.7
                adapter_strength_min = 0.5
                adapter_strength_max = 1.1

            adapter_conditioning_scale = torch.rand(
                (1,), device=self.device_torch, dtype=dtype
            )

            adapter_conditioning_scale = value_map(
                adapter_conditioning_scale,
                0.0,
                1.0,
                adapter_strength_min,
                adapter_strength_max
            )
            return adapter_conditioning_scale

        # flush()
        with self.timer('grad_setup'):

            # text encoding
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True

            if self.embedding is not None:
                grad_on_text_encoder = True

            if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                grad_on_text_encoder = True

            if self.adapter_config and self.adapter_config.type == 'te_augmenter':
                grad_on_text_encoder = True

            # have a blank network so we can wrap it in a context and set multipliers without checking every time
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()

            # set the weights
            network.multiplier = network_weight_list

        # activate network if it exits

        prompts_1 = conditioned_prompts
        prompts_2 = None
        if self.train_config.short_and_long_captions_encoder_split and self.sd.is_xl:
            prompts_1 = batch.get_caption_short_list()
            prompts_2 = conditioned_prompts

            # make the batch splits
        if self.train_config.single_item_batching:
            if self.model_config.refiner_name_or_path is not None:
                raise ValueError("Single item batching is not supported when training the refiner")
            batch_size = noisy_latents.shape[0]
            # chunk/split everything
            noisy_latents_list = torch.chunk(noisy_latents, batch_size, dim=0)
            noise_list = torch.chunk(noise, batch_size, dim=0)
            timesteps_list = torch.chunk(timesteps, batch_size, dim=0)
            conditioned_prompts_list = [[prompt] for prompt in prompts_1]
            if imgs is not None:
                imgs_list = torch.chunk(imgs, batch_size, dim=0)
            else:
                imgs_list = [None for _ in range(batch_size)]
            if adapter_images is not None:
                adapter_images_list = torch.chunk(adapter_images, batch_size, dim=0)
            else:
                adapter_images_list = [None for _ in range(batch_size)]
            if clip_images is not None:
                clip_images_list = torch.chunk(clip_images, batch_size, dim=0)
            else:
                clip_images_list = [None for _ in range(batch_size)]
            mask_multiplier_list = torch.chunk(mask_multiplier, batch_size, dim=0)
            if prompts_2 is None:
                prompt_2_list = [None for _ in range(batch_size)]
            else:
                prompt_2_list = [[prompt] for prompt in prompts_2]

        else:
            noisy_latents_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditioned_prompts_list = [prompts_1]
            imgs_list = [imgs]
            adapter_images_list = [adapter_images]
            clip_images_list = [clip_images]
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for batch_idx, (noisy_latents, noise, timesteps, conditioned_prompts, imgs, adapter_images, clip_images, mask_multiplier, prompt_2) in enumerate(zip(
                noisy_latents_list,
                noise_list,
                timesteps_list,
                conditioned_prompts_list,
                imgs_list,
                adapter_images_list,
                clip_images_list,
                mask_multiplier_list,
                prompt_2_list
        )):

            # if self.train_config.negative_prompt is not None:
            #     # add negative prompt
            #     conditioned_prompts = conditioned_prompts + [self.train_config.negative_prompt for x in
            #                                                  range(len(conditioned_prompts))]
            #     if prompt_2 is not None:
            #         prompt_2 = prompt_2 + [self.train_config.negative_prompt for x in range(len(prompt_2))]

            with (network):
                # encode clip adapter here so embeds are active for tokenizer
                if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                    with self.timer('encode_clip_vision_embeds'):
                        if has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True
                            )
                        else:
                            # just do a blank one
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                torch.zeros(
                                    (noisy_latents.shape[0], 3, 512, 512),
                                    device=self.device_torch, dtype=dtype
                                ),
                                is_training=True,
                                has_been_preprocessed=True,
                                drop=True
                            )
                        # it will be injected into the tokenizer when called
                        self.adapter(conditional_clip_embeds)

                # do the custom adapter after the prior prediction
                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or is_reg):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    self.adapter.trigger_pre_te(
                        tensors_preprocessed=clip_images if not is_reg else None,  # on regs we send none to get random noise
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count,
                        batch_tensor=batch.tensor if not is_reg else None,
                        batch_size=noisy_latents.shape[0]
                    )

                with self.timer('encode_prompt'):
                    unconditional_embeds = None
                    prompt_kwargs = {}
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
                        with torch.set_grad_enabled(False):
                            if batch.prompt_embeds is not None:
                                # use the cached embeds (full precision when fp32_front)
                                # (text conditioning dropout is applied centrally after the
                                #  encode block via _apply_conditioning_dropout)
                                conditional_embeds = batch.prompt_embeds.clone().detach().to(
                                    self.device_torch, dtype=embed_dtype
                                )
                            else:
                                embeds_to_use = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=embed_dtype
                                )
                                if self.cached_trigger_embeds is not None and not is_reg:
                                    embeds_to_use = self.cached_trigger_embeds.clone().detach().to(
                                        self.device_torch, dtype=embed_dtype
                                    )
                                conditional_embeds = concat_prompt_embeds(
                                    [embeds_to_use] * noisy_latents.shape[0]
                                )
                            if self.train_config.do_cfg:
                                if self.train_config.cfg_same_prompt:
                                    # use the same (conditional) prompt embeds for the unconditional side
                                    unconditional_embeds = conditional_embeds.clone().detach().to(
                                        self.device_torch, dtype=embed_dtype
                                    )
                                else:
                                    unconditional_embeds = self.cached_blank_embeds.clone().detach().to(
                                        self.device_torch, dtype=embed_dtype
                                    )
                                    unconditional_embeds = concat_prompt_embeds(
                                        [unconditional_embeds] * noisy_latents.shape[0]
                                    )

                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False

                    elif grad_on_text_encoder:
                        with torch.set_grad_enabled(True):
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=0.0,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(
                                self.device_torch,
                                dtype=dtype)

                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                if self.train_config.cfg_same_prompt:
                                    # Reuse the conditional embeds so both CFG branches share the exact
                                    # same text (and the same caption-dropout decision). Re-encoding would
                                    # re-roll dropout independently per branch and could make the two prompts
                                    # differ, breaking the "same prompt" invariant.
                                    unconditional_embeds = conditional_embeds.clone()
                                else:
                                    # todo only do one and repeat it
                                    uncond_prompts = uncond_prompts_2 = self.batch_negative_prompt
                                    unconditional_embeds = self.sd.encode_prompt(
                                        uncond_prompts,
                                        uncond_prompts_2,
                                        dropout_prob=0.0,
                                        long_prompts=self.do_long_prompts,
                                        **prompt_kwargs
                                    ).to(
                                        self.device_torch,
                                        dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                    else:
                        with torch.set_grad_enabled(False):
                            # make sure it is in eval mode
                            if isinstance(self.sd.text_encoder, list):
                                for te in self.sd.text_encoder:
                                    te.eval()
                            else:
                                self.sd.text_encoder.eval()
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            if self.sd.encode_control_in_text_embeddings and batch.control_tensor_list is not None:
                                prompt_kwargs['control_images'] = batch.control_tensor_list
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=0.0,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(
                                self.device_torch,
                                dtype=dtype)
                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                if self.train_config.cfg_same_prompt:
                                    # Reuse the conditional embeds so both CFG branches share the exact
                                    # same text (and the same caption-dropout decision).
                                    unconditional_embeds = conditional_embeds.clone()
                                else:
                                    uncond_prompts = self.batch_negative_prompt
                                    unconditional_embeds = self.sd.encode_prompt(
                                        uncond_prompts,
                                        dropout_prob=0.0,
                                        long_prompts=self.do_long_prompts,
                                        **prompt_kwargs
                                    ).to(
                                        self.device_torch,
                                        dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                            
                            if self.train_config.diff_output_preservation:
                                dop_prompts = [p.replace(self.trigger_word, self.train_config.diff_output_preservation_class) for p in conditioned_prompts]
                                dop_prompts_2 = None
                                if prompt_2 is not None:
                                    dop_prompts_2 = [p.replace(self.trigger_word, self.train_config.diff_output_preservation_class) for p in prompt_2]
                                self.diff_output_preservation_embeds = self.sd.encode_prompt(
                                    dop_prompts, dop_prompts_2,
                                    dropout_prob=0.0,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
                        # detach the embeddings
                        conditional_embeds = conditional_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_embeds = unconditional_embeds.detach()
                    
                    if self.decorator:
                        conditional_embeds.text_embeds = self.decorator(
                            conditional_embeds.text_embeds
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds.text_embeds = self.decorator(
                                unconditional_embeds.text_embeds, 
                                is_unconditional=True
                            )

                # Apply configured conditioning dropout (text + image, positive +
                # negative, sync/invert, global + per-dataset) and compute the
                # per-item image-conditioning masks for the model.
                conditional_embeds, unconditional_embeds, pos_image_mask, neg_image_mask = \
                    self._apply_conditioning_dropout(
                        batch, conditional_embeds, unconditional_embeds, item_index=batch_idx
                    )
                self._cur_image_cond_mask_pos = pos_image_mask
                self._cur_image_cond_mask_neg = neg_image_mask

                # flush()
                pred_kwargs = {}

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, T2IAdapter)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, T2IAdapter)):
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                down_block_additional_residuals = adapter(adapter_images)
                                if self.assistant_adapter:
                                    # not training. detach
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype).detach() * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]
                                else:
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype) * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]

                                pred_kwargs['down_intrablock_additional_residuals'] = down_block_additional_residuals

                if self.adapter and isinstance(self.adapter, IPAdapter):
                    with self.timer('encode_adapter_embeds'):
                        # number of images to do if doing a quad image
                        quad_count = random.randint(1, 4)
                        image_size = self.adapter.input_size
                        if has_clip_image_embeds:
                            # todo handle reg images better than this
                            if is_reg:
                                # get unconditional image embeds from cache
                                embeds = [
                                    load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in
                                    range(noisy_latents.shape[0])
                                ]
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                    embeds,
                                    quad_count=quad_count
                                )

                                if self.train_config.do_cfg:
                                    embeds = [
                                        load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in
                                        range(noisy_latents.shape[0])
                                    ]
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                        embeds,
                                        quad_count=quad_count
                                    )

                            else:
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                    batch.clip_image_embeds,
                                    quad_count=quad_count
                                )
                                if self.train_config.do_cfg:
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                        batch.clip_image_embeds_unconditional,
                                        quad_count=quad_count
                                    )
                        elif is_reg:
                            # we will zero it out in the img embedder
                            clip_images = torch.zeros(
                                (noisy_latents.shape[0], 3, image_size, image_size),
                                device=self.device_torch, dtype=dtype
                            ).detach()
                            # drop will zero it out
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images,
                                drop=True,
                                is_training=True,
                                has_been_preprocessed=False,
                                quad_count=quad_count
                            )
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                    torch.zeros(
                                        (noisy_latents.shape[0], 3, image_size, image_size),
                                        device=self.device_torch, dtype=dtype
                                    ).detach(),
                                    is_training=True,
                                    drop=True,
                                    has_been_preprocessed=False,
                                    quad_count=quad_count
                                )
                        elif has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True,
                                quad_count=quad_count,
                                # do cfg on clip embeds to normalize the embeddings for when doing cfg
                                # cfg_embed_strength=3.0 if not self.train_config.do_cfg else None
                                # cfg_embed_strength=3.0 if not self.train_config.do_cfg else None
                            )
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                    clip_images.detach().to(self.device_torch, dtype=dtype),
                                    is_training=True,
                                    drop=True,
                                    has_been_preprocessed=True,
                                    quad_count=quad_count
                                )
                        else:
                            print_acc("No Clip Image")
                            print_acc([file_item.path for file_item in batch.file_items])
                            raise ValueError("Could not find clip image")

                    if not self.adapter_config.train_image_encoder:
                        # we are not training the image encoder, so we need to detach the embeds
                        conditional_clip_embeds = conditional_clip_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_clip_embeds = unconditional_clip_embeds.detach()

                    with self.timer('encode_adapter'):
                        self.adapter.train()
                        conditional_embeds = self.adapter(
                            conditional_embeds.detach(),
                            conditional_clip_embeds,
                            is_unconditional=False
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds = self.adapter(
                                unconditional_embeds.detach(),
                                unconditional_clip_embeds,
                                is_unconditional=True
                            )
                        else:
                            # wipe out unconsitional
                            self.adapter.last_unconditional = None

                if self.adapter and isinstance(self.adapter, ReferenceAdapter):
                    # pass in our noise scheduler
                    self.adapter.noise_scheduler = self.sd.noise_scheduler
                    if has_clip_image or has_adapter_img:
                        img_to_use = clip_images if has_clip_image else adapter_images
                        # currently 0-1 needs to be -1 to 1
                        reference_images = ((img_to_use - 0.5) * 2).detach().to(self.device_torch, dtype=dtype)
                        self.adapter.set_reference_images(reference_images)
                        self.adapter.noise_scheduler = self.sd.noise_scheduler
                    elif is_reg:
                        self.adapter.set_blank_reference_images(noisy_latents.shape[0])
                    else:
                        self.adapter.set_reference_images(None)

                prior_pred = None

                do_inverted_masked_prior = False
                if self.train_config.inverted_mask_prior and batch.mask_tensor is not None:
                    do_inverted_masked_prior = True

                do_correct_pred_norm_prior = self.train_config.correct_pred_norm

                do_guidance_prior = False

                if batch.unconditional_latents is not None:
                    # for this not that, we need a prior pred to normalize
                    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type
                    if guidance_type == 'tnt':
                        do_guidance_prior = True

                if ((
                        has_adapter_img and self.assistant_adapter and match_adapter_assist) or self.do_prior_prediction or do_guidance_prior or do_reg_prior or do_inverted_masked_prior or self.train_config.correct_pred_norm):
                    with self.timer('prior predict'):
                        prior_embeds_to_use = conditional_embeds
                        # use diff_output_preservation embeds if doing dfe
                        if self.train_config.diff_output_preservation:
                            prior_embeds_to_use = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                        
                        if self.train_config.blank_prompt_preservation:
                            blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                self.device_torch, dtype=dtype
                            )
                            prior_embeds_to_use = concat_prompt_embeds(
                                [blank_embeds] * noisy_latents.shape[0]
                            )
                        
                        prior_pred = self.get_prior_prediction(
                            noisy_latents=noisy_latents,
                            conditional_embeds=prior_embeds_to_use,
                            match_adapter_assist=match_adapter_assist,
                            network_weight_list=network_weight_list,
                            timesteps=timesteps,
                            pred_kwargs=pred_kwargs,
                            noise=noise,
                            batch=batch,
                            unconditional_embeds=unconditional_embeds,
                            conditioned_prompts=conditioned_prompts
                        )
                        if prior_pred is not None:
                            prior_pred = prior_pred.detach()

                # do the custom adapter after the prior prediction
                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or self.adapter_config.type in ['llm_adapter', 'text_encoder']):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    conditional_embeds = self.adapter.condition_encoded_embeds(
                        tensors_0_1=clip_images,
                        prompt_embeds=conditional_embeds,
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count
                    )
                    if self.train_config.do_cfg and unconditional_embeds is not None:
                        unconditional_embeds = self.adapter.condition_encoded_embeds(
                            tensors_0_1=clip_images,
                            prompt_embeds=unconditional_embeds,
                            is_training=True,
                            has_been_preprocessed=True,
                            is_unconditional=True,
                            quad_count=quad_count
                        )

                if self.adapter and isinstance(self.adapter, CustomAdapter) and batch.extra_values is not None:
                    self.adapter.add_extra_values(batch.extra_values.detach())

                    if self.train_config.do_cfg:
                        self.adapter.add_extra_values(torch.zeros_like(batch.extra_values.detach()),
                                                      is_unconditional=True)

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, ControlNetModel)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, ControlNetModel)):
                        if self.train_config.do_cfg:
                            raise ValueError("ControlNetModel is not supported with CFG")
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter: ControlNetModel = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                # add_text_embeds is pooled_prompt_embeds for sdxl
                                added_cond_kwargs = {}
                                if self.sd.is_xl:
                                    added_cond_kwargs["text_embeds"] = conditional_embeds.pooled_embeds
                                    added_cond_kwargs['time_ids'] = self.sd.get_time_ids_from_latents(noisy_latents)
                                down_block_res_samples, mid_block_res_sample = adapter(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=conditional_embeds.text_embeds,
                                    controlnet_cond=adapter_images,
                                    conditioning_scale=1.0,
                                    guess_mode=False,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )
                                pred_kwargs['down_block_additional_residuals'] = down_block_res_samples
                                pred_kwargs['mid_block_additional_residual'] = mid_block_res_sample
                
                if self.train_config.do_guidance_loss and isinstance(self.train_config.guidance_loss_target, list):
                    batch_size = noisy_latents.shape[0]
                    # update the guidance value, random float between guidance_loss_target[0] and guidance_loss_target[1]
                    self._guidance_loss_target_batch = [
                        random.uniform(
                            self.train_config.guidance_loss_target[0],
                            self.train_config.guidance_loss_target[1]
                        ) for _ in range(batch_size)
                    ]

                self.before_unet_predict()
                
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
                with self.timer('condition_noisy_latents'):
                    # do it for the model
                    noisy_latents = self.sd.condition_noisy_latents(noisy_latents, batch)
                    if self.adapter and isinstance(self.adapter, CustomAdapter):
                        noisy_latents = self.adapter.condition_noisy_latents(noisy_latents, batch)
                
                if self.train_config.timestep_type == 'next_sample':
                    with self.timer('next_sample_step'):
                        with torch.no_grad():
                            
                            stepped_timestep_indicies = [self.sd.noise_scheduler.index_for_timestep(t) + 1 for t in timesteps]
                            stepped_timesteps = [self.sd.noise_scheduler.timesteps[x] for x in stepped_timestep_indicies]
                            stepped_timesteps = torch.stack(stepped_timesteps, dim=0)
                            
                            # do a sample at the current timestep and step it, then determine new noise
                            next_sample_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                                unconditional_embeds=unconditional_embeds,
                                batch=batch,
                                **pred_kwargs
                            )
                            stepped_latents = self.sd.step_scheduler(
                                next_sample_pred,
                                noisy_latents,
                                timesteps,
                                self.sd.noise_scheduler
                            )
                            # stepped latents is our new noisy latents. Now we need to determine noise in the current sample
                            noisy_latents = stepped_latents
                            original_samples = batch.latents.to(self.device_torch, dtype=dtype)
                            # todo calc next timestep, for now this may work as it
                            t_01 = (stepped_timesteps / 1000).to(original_samples.device)
                            if len(stepped_latents.shape) == 4:
                                t_01 = t_01.view(-1, 1, 1, 1)
                            elif len(stepped_latents.shape) == 5:
                                t_01 = t_01.view(-1, 1, 1, 1, 1)
                            else:
                                raise ValueError("Unknown stepped latents shape", stepped_latents.shape)
                            next_sample_noise = (stepped_latents - (1.0 - t_01) * original_samples) / t_01
                            noise = next_sample_noise
                            timesteps = stepped_timesteps
                # do a prior pred if we have an unconditional image, we will swap out the giadance later
                if batch.unconditional_latents is not None or self.do_guided_loss:
                    # do guided loss
                    loss = self.get_guided_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        mask_multiplier=mask_multiplier,
                        prior_pred=prior_pred,
                    )
                    
                elif self.train_config.loss_type == 'mean_flow':
                    loss = self.get_mean_flow_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        prior_pred=prior_pred,
                    )
                else:
                    # If the text embedding is not successfully obtained in the previous encoding stage, perform a safe fallback to avoid a None error.
                    # At the same time, extract the control graph from the batch again here to avoid missing data due to the upstream prompt_kwargs not being set.
                    if self.sd.encode_control_in_text_embeddings:
                        if getattr(self.sd, 'has_multiple_control_images', False) and batch.control_tensor_list is not None:
                            # Multiple images: Grouped by control channel and stacked into a batch tensor list
                            num_controls = len(batch.control_tensor_list[0])
                            batched_controls = []
                            for c_idx in range(num_controls):
                                per_items = [item_controls[c_idx] for item_controls in batch.control_tensor_list]
                                batched = torch.stack(per_items, dim=0).to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                                batched_controls.append(batched)
                            prompt_kwargs['control_images'] = batched_controls
                        elif batch.control_tensor is not None:
                            # Single image: Directly use batch tensors
                            prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if conditional_embeds is None:
                        # Encode a default embedding using the prompt word of the current batch; if no prompt word is specified, encode an empty string.
                        fallback_prompts = conditioned_prompts if conditioned_prompts is not None else ['']
                        conditional_embeds = self.sd.encode_prompt(
                            fallback_prompts,
                            prompt_2,
                            dropout_prob=0.0,
                            long_prompts=self.do_long_prompts,
                            **prompt_kwargs
                        ).to(self.device_torch, dtype=dtype)
                    if self.train_config.do_cfg and unconditional_embeds is None:
                        # When CFG is enabled but no unconditional embedding is generated, fall back to generating an unconditional embedding with an empty hint.
                        if self.train_config.cfg_same_prompt and conditional_embeds is not None:
                            # Reuse the conditional embeds so both branches share the same text.
                            unconditional_embeds = conditional_embeds.clone()
                        else:
                            uncond_prompts = uncond_prompts_2 = self.batch_negative_prompt if hasattr(self, 'batch_negative_prompt') else ['']
                            unconditional_embeds = self.sd.encode_prompt(
                                uncond_prompts,
                                uncond_prompts_2,
                                dropout_prob=0.0,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(self.device_torch, dtype=dtype)
                    with self.timer('predict_unet'):
                        noise_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            is_primary_pred=True,
                            **pred_kwargs
                        )
                    self.after_unet_predict()

                    with self.timer('calculate_loss'):
                        noise = noise.to(self.device_torch, dtype=dtype).detach()
                        prior_to_calculate_loss = prior_pred
                        # if we are doing diff_output_preservation and not noing inverted masked prior
                        # then we need to send none here so it will not target the prior
                        doing_preservation = self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation
                        if doing_preservation and not do_inverted_masked_prior:
                            prior_to_calculate_loss = None
                        
                        loss = self.calculate_loss(
                            noise_pred=noise_pred,
                            noise=noise,
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=mask_multiplier,
                            prior_pred=prior_to_calculate_loss,
                        )
                    
                    if self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation:
                        with torch.no_grad():
                            if self.train_config.diff_output_preservation:
                                preservation_embeds = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                            elif self.train_config.blank_prompt_preservation:
                                blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                preservation_embeds = concat_prompt_embeds(
                                    [blank_embeds] * noisy_latents.shape[0]
                                )
                        preservation_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            **pred_kwargs
                        )
                        multiplier = self.train_config.diff_output_preservation_multiplier if self.train_config.diff_output_preservation else self.train_config.blank_prompt_preservation_multiplier
                        preservation_loss = torch.nn.functional.mse_loss(preservation_pred, prior_pred) * multiplier
                        self.additional_logs['loss/normal'] = loss.item()
                        self.additional_logs['loss/preservation'] = preservation_loss.item()
                        loss = loss + preservation_loss
                
                # NOTE: L_mid is NOT added to the training loss here.
                # Gates are deliberately excluded from the optimizer and updated
                # by a dedicated SparseForge rule in hook_train_loop (update_rank_gates).
                # Adding L_mid to the loss would compute gradients that are never
                # applied (and periodically zeroed), so it has zero training effect.
                # The mid-preference pressure is instead applied via eq.(4) nudge
                # inside update_rank_gates, which is the actual SparseForge mechanism.
                # We DO log L_mid for monitoring, but only from the post-optimizer block.

                # check if nan
                if torch.isnan(loss):
                    print_acc("loss is nan")
                    loss = torch.zeros_like(loss).requires_grad_(True)

                with self.timer('backward'):
                    # todo we have multiplier seperated. works for now as res are not in same batch, but need to change
                    loss = loss * loss_multiplier.mean()
                    # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                    # it will destroy the gradients. This is because the network is a context manager
                    # and will change the multipliers back to 0.0 when exiting. They will be
                    # 0.0 for the backward pass and the gradients will be 0.0
                    # I spent weeks on fighting this. DON'T DO IT
                    # with fsdp_overlap_step_with_backward():
                    # if self.is_bfloat:
                    # loss.backward()
                    # else:
                    
                    # Gradient projection for spectral_flow/mse_spectral_flow loss: compute gradients
                    # separately and project to resolve conflicts between loss objectives
                    if (self.train_config.mse_spectral_flow_gradient_projection_enabled and
                        self._mse_loss_tensor is not None and
                        self._spectral_loss_tensor is not None and
                        self._flow_loss_tensor is not None):
                        # 3-loss mode: MSE + Spectral + Flow
                        self._gradient_projection_backward(
                            self._spectral_loss_tensor, self._flow_loss_tensor,
                            self._mse_loss_tensor
                        )
                        # Clear stored tensors
                        self._mse_loss_tensor = None
                        self._spectral_loss_tensor = None
                        self._flow_loss_tensor = None
                    elif (self.train_config.spectral_flow_gradient_projection_enabled and
                          self._spectral_loss_tensor is not None and
                          self._flow_loss_tensor is not None):
                        # 2-loss mode: Spectral + Flow
                        self._gradient_projection_backward(
                            self._spectral_loss_tensor, self._flow_loss_tensor
                        )
                        # Clear stored tensors
                        self._spectral_loss_tensor = None
                        self._flow_loss_tensor = None
                    else:
                        self.accelerator.backward(loss)

        return loss.detach()
        # flush()

    def hook_train_loop(self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO]]):
        if isinstance(batch, list):
            batch_list = batch
        else:
            batch_list = [batch]
        total_loss = None
        self.optimizer.zero_grad()
        for batch in batch_list:
            if self.sd.is_multistage:
                # Boundary switching is handled by BaseSDTrainProcess.switch_boundary_if_needed()
                # in the step loop (based on switch_boundary_every).
                # Here we only ensure we are on a trainable boundary.
                if self.current_boundary_index not in self.sd.trainable_multistage_boundaries:
                    # iterate to make sure we only train trainable_multistage_boundaries
                    while True:
                        self.current_boundary_index += 1
                        if self.current_boundary_index >= len(self.sd.multistage_boundaries):
                            self.current_boundary_index = 0
                        if self.current_boundary_index in self.sd.trainable_multistage_boundaries:
                            # if this boundary is trainable, we can stop looking
                            break
            loss = self.train_single_accumulation(batch)
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            if len(batch_list) > 1 and self.model_config.low_vram:
                torch.cuda.empty_cache()


        grad_norm_value = None
        if not self.is_grad_accumulation_step:
            # grads of memory-managed (offloaded) params are async D2H copies into
            # pinned tensors; join them before anything on the CPU reads .grad
            sync_grad_transfers()
            
            # Update Fisher EMA for rank gate annealing (after sync_grad_transfers
            # to ensure grads are available even with low_vram offloading)
            if self.fisher_tracker is not None:
                self.fisher_tracker.update(self.params)
            
            grad_norm_tensor = self._calculate_grad_norm(self.params)
            if grad_norm_tensor is not None:
                grad_norm_value = grad_norm_tensor.item()
            # fix this for multi params
            if self.train_config.optimizer != 'adafactor':
                if isinstance(self.params[0], dict):
                    for i in range(len(self.params)):
                        self.accelerator.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                else:
                    self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm)

            # Determine which expert is active for this batch (Wan 2.2 multistage).
            # We'll freeze the inactive expert's LoRA params during optimizer.step()
            # so they are completely ignored: no weight decay, no updates.
            active_experts = None
            if hasattr(self.sd, 'model') and hasattr(self.sd.model, '_active_transformer_name'):
                active_t = self.sd.model._active_transformer_name  # "transformer_1" or "transformer_2"
                active_experts = {active_t}

            # Freeze inactive expert LoRAs before optimizer.step()
            frozen_inactive_loras = self._freeze_inactive_expert_loras(active_experts)

            # Step loss rejection: check if any expert's loss exceeds thresholds
            step_rejected = False
            rejection_reasons = []
            if not self.is_grad_accumulation_step:
                # Normalize per-expert loss by number of batches in this step
                num_batches = len(batch_list) if batch_list else 1
                for expert_label, accum_loss in self.current_step_expert_loss.items():
                    avg_expert_loss = accum_loss / num_batches
                    avg_spatial_loss = self.current_step_expert_spatial.get(expert_label, 0.0) / num_batches
                    avg_flow_loss = self.current_step_expert_flow.get(expert_label, 0.0) / num_batches
                    should_reject, reason = self._should_reject_step(
                        expert_label, avg_expert_loss, avg_spatial_loss, avg_flow_loss
                    )
                    if should_reject:
                        step_rejected = True
                        rejection_reasons.append(f"{expert_label}: {reason}")
            
            if step_rejected:
                # Reject this step: zero gradients, skip optimizer step
                if self.accelerator.is_main_process:
                    print_acc(f"[STEP REJECTED] {'; '.join(rejection_reasons)}")
                
                # Track rejection count
                for expert_label in self.current_step_expert_loss.keys():
                    if expert_label not in self.step_rejection_count:
                        self.step_rejection_count[expert_label] = 0
                    self.step_rejection_count[expert_label] += 1
                
                # Zero gradients without stepping
                self.optimizer.zero_grad(set_to_none=True)
                self._unfreeze_inactive_expert_loras(frozen_inactive_loras)
                
                # Reset current step tracking
                self.current_step_expert_loss = {}
            else:
                # only step if we are not accumulating
                with self.timer('optimizer_step'):
                    self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                    # Update previous loss tracking for next step's rejection check
                    num_batches = len(batch_list) if batch_list else 1
                    for expert_label, accum_loss in self.current_step_expert_loss.items():
                        self.prev_expert_loss[expert_label] = accum_loss / num_batches
                        self.prev_expert_spatial_loss[expert_label] = self.current_step_expert_spatial.get(expert_label, 0.0) / num_batches
                        self.prev_expert_flow_loss[expert_label] = self.current_step_expert_flow.get(expert_label, 0.0) / num_batches
                        self.prev_expert_mse_loss[expert_label] = self.current_step_expert_mse.get(expert_label, 0.0) / num_batches
                    self.current_step_expert_loss = {}  # Reset for next step
                    self.current_step_expert_spatial = {}
                    self.current_step_expert_flow = {}
                    self.current_step_expert_mse = {}
                    
                    # Reset per-step gradient projection stats for next step
                    self.gradient_projection_stats['step_conflicts'] = 0
                    self.gradient_projection_stats['step_projections'] = 0

                    # Unfreeze inactive expert LoRAs after optimizer.step()
                    self._unfreeze_inactive_expert_loras(frozen_inactive_loras)

                    if self.adapter and isinstance(self.adapter, CustomAdapter):
                        self.adapter.post_weight_update()
                if self.ema is not None:
                    with self.timer('ema_update'):
                        # Determine active experts for per-expert EMA updates.
                        # For Wan 2.2 multistage models, only the expert that processed
                        # this batch should have its EMA updated. The other expert's EMA
                        # must remain completely frozen.
                        self.ema.update(active_experts=active_experts)

                # Step LR scheduler only when optimizer steps (not during gradient accumulation)
                # Scheduler total_iters is adjusted for gradient accumulation in BaseSDTrainProcess
                with self.timer('scheduler_step'):
                    if self.use_per_expert_schedulers:
                        # Dual-expert mode: step only the active expert's scheduler
                        # and track per-expert step counts.
                        if active_experts is not None:
                            for expert in active_experts:
                                expert_scheduler = self.expert_lr_schedulers.get(expert)
                                if expert_scheduler is not None:
                                    expert_scheduler.step()
                                    self.expert_step_counts[expert] += 1
                    else:
                        # Single-expert or non-multistage mode: use global scheduler
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
            
            # Rank gate annealing updates (after optimizer step)
            # Uses active_experts from above (same as used for EMA and LoRA freezing).
            # Schedule operates on GLOBAL steps (user-provided start_step/end_step
            # are global steps). Per-expert filtering only controls WHICH gates update.
            if self.rank_gates_scheduler is not None and self.network is not None:
                gated_loras = self.network.gated_loras
                if gated_loras:
                    global_step = self.step_num
                    
                    # During hardening window: apply soft→hard interpolation to ALL experts.
                    # The interpolation is idempotent (snapshot-based, no gradients), so
                    # applying it to frozen experts is safe and necessary: with
                    # switch_boundary_every > hardening_window, an expert could otherwise
                    # be inactive for the entire window and its gates would jump abruptly
                    # from soft to binary at finalize_gates.
                    if self.rank_gates_scheduler.is_hardening(global_step):
                        apply_hardening_interpolation(
                            gated_loras, self.rank_gates_scheduler, global_step
                        )
                    
                    # During annealing: update gates every N steps.
                    # Only active expert's gates are updated (frozen expert untouched).
                    elif self.rank_gates_scheduler.should_update(global_step):
                        # Increment per-expert update count for correct temperature decay.
                        if active_experts is not None:
                            for expert in active_experts:
                                self.rank_gates_expert_update_counts[expert] = \
                                    self.rank_gates_expert_update_counts.get(expert, 0) + 1
                        else:
                            # Single-expert or shared mode: track under "shared" key
                            self.rank_gates_expert_update_counts["shared"] = \
                                self.rank_gates_expert_update_counts.get("shared", 0) + 1
                        
                        L_mid = update_rank_gates(
                            gated_loras, self.fisher_tracker, 
                            self.rank_gates_scheduler, global_step,
                            active_experts=active_experts,
                            expert_update_counts=self.rank_gates_expert_update_counts
                        )
                        # Log L_mid for monitoring
                        self.additional_logs['loss/L_mid'] = L_mid.item()
                    
                    # Log stats periodically
                    if global_step % 500 == 0:
                        stats = log_gate_stats(gated_loras, global_step)
                        writer = getattr(self, 'writer', None)
                        if stats and writer is not None:
                            for key, value in stats.items():
                                writer.add_scalar(key, value, global_step)
        else:
            # gradient accumulation. Just a place for breakpoint
            pass

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()
        if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
            with self.timer('restore_adapter'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.adapter.restore_embeddings()

        avg_loss = (total_loss / len(batch_list)).item()

        # For MoE models training both experts, log only expert-tagged loss (loss_low/loss_high).
        # For single-expert training, log the standard 'loss'.
        expert_label = self._get_active_expert_label()
        if expert_label != "single":
            # MoE mode: log per-expert loss only
            loss_dict = OrderedDict()
            loss_dict[f'loss_{expert_label}'] = avg_loss
        else:
            # Single expert mode
            loss_dict = OrderedDict({'loss': avg_loss})

        if grad_norm_value is not None:
            loss_dict['grad_norm'] = grad_norm_value

        # Log attention softcapping stats periodically
        # Note: update_softcap_step() is called at start of train_single_accumulation()
        # so step counter is correct during forward pass (when attention ops happen).
        if hasattr(self, 'step_num') and self.step_num % 10 == 0:
            self._log_attention_stats()

        self.end_of_training_loop()

        return loss_dict
