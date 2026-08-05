import torch
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
import torch
from toolkit.config_modules import GenerateImageConfig
from .wan22_pipeline import Wan22Pipeline

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from diffusers import WanImageToVideoPipeline
from torchvision.transforms import functional as TF

from .wan22_14b_model import Wan2214bModel

class Wan2214bI2VModel(Wan2214bModel):
    arch = "wan22_14b_i2v"
    
    
    def generate_single_image(
        self,
        pipeline: Wan22Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        
        # todo 
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)

        num_frames = (
            (gen_config.num_frames - 1) // 4
        ) * 4 + 1  # make sure it is divisible by 4 + 1
        gen_config.num_frames = num_frames

        height = gen_config.height
        width = gen_config.width
        first_frame_n1p1 = None
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img).convert("RGB")

            d = self.get_bucket_divisibility()

            # make sure they are divisible by d
            height = height // d * d
            width = width // d * d

            # resize the control image
            control_img = control_img.resize((width, height), Image.LANCZOS)

            # 5. Prepare latent variables
            # num_channels_latents = self.transformer.config.in_channels
            num_channels_latents = 16
            latents = pipeline.prepare_latents(
                1,
                num_channels_latents,
                height,
                width,
                gen_config.num_frames,
                torch.float32,
                self.device_torch,
                generator,
                None,
            ).to(self.torch_dtype)

            first_frame_n1p1 = (
                TF.to_tensor(control_img)
                .unsqueeze(0)
                .to(self.device_torch, dtype=self.torch_dtype)
                * 2.0
                - 1.0
            )  # normalize to [-1, 1]
            
            # Add conditioning using the standalone function
            gen_config.latents = add_first_frame_conditioning(
                latent_model_input=latents,
                first_frame=first_frame_n1p1,
                vae=self.vae
            )

        output = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            height=height,
            width=width,
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
            **extra,
        )[0]

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img
    
    def _apply_first_frame_conditioning_cached(
        self, latent_model_input: torch.Tensor, first_frame_latents: torch.Tensor
    ) -> torch.Tensor:
        """Apply first frame conditioning using cached first frame latents."""
        device = latent_model_input.device
        dtype = latent_model_input.dtype
        
        batch_size = latent_model_input.shape[0]
        num_latent_frames = latent_model_input.shape[2]
        num_frames = (num_latent_frames - 1) * 4 + 1
        
        vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        latent_height = first_frame_latents.shape[3]
        latent_width = first_frame_latents.shape[4]
        
        # Initialize mask for all frames
        mask_lat_size = torch.ones(
            batch_size, 1, num_frames, latent_height, latent_width, device=device, dtype=dtype)
        
        # Set all non-first frames to 0
        mask_lat_size[:, :, 1:] = 0
        
        # Special handling for first frame
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
        
        # Combine first frame mask with rest
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        
        # Reshape and transpose for model input
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        
        # Pad first frame latents to match temporal dimension
        latent_condition = torch.zeros(
            batch_size, first_frame_latents.shape[1], num_latent_frames, latent_height, latent_width,
            device=device, dtype=dtype
        )
        latent_condition[:, :, 0:1] = first_frame_latents
        
        # Combine conditioning with latent input
        first_frame_condition = torch.concat(
            [mask_lat_size, latent_condition], dim=1)
        return torch.cat([latent_model_input, first_frame_condition], dim=1)

    def _pad_for_no_conditioning(
        self, latent_model_input: torch.Tensor
    ) -> torch.Tensor:
        """Pad latents with zero conditioning to match the shape of conditioned inputs.
        
        For mixed I2V/T2V batches, T2V items need to be padded so they have the same
        channel dimension as I2V items. The zero conditioning means the model will
        effectively ignore conditioning for these items.
        
        Structure matches _apply_first_frame_conditioning_cached:
        - mask_lat_size: vae_scale_factor_temporal channels (temporal reshape)
        - latent_condition: z_dim channels (from first frame latent)
        - Total added: vae_scale_factor_temporal + z_dim
        
        Optimized: since all padding is zeros, we create it directly without intermediate
        reshape/concat operations that were needed for non-zero masks.
        """
        device = latent_model_input.device
        dtype = latent_model_input.dtype
        num_latent_frames = latent_model_input.shape[2]
        
        vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        z_dim = latent_model_input.shape[1]
        
        # Directly create the zero padding tensor - avoids intermediate (bs, 1, num_frames, H, W)
        # that required repeat_interleave + concat + view + transpose
        padding_shape = (
            latent_model_input.shape[0],
            vae_scale_factor_temporal + z_dim,
            num_latent_frames,
            latent_model_input.shape[3],
            latent_model_input.shape[4],
        )
        first_frame_condition = torch.zeros(padding_shape, device=device, dtype=dtype)
        
        return torch.cat([latent_model_input, first_frame_condition], dim=1)
        # Final: (batch_size, z_dim + vae_scale_factor_temporal + z_dim, T, H, W)
        # For Wan22 14B: 16 + 4 + 16 = 36 channels

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs
    ):
        # videos come in (bs, num_frames, channels, height, width)
        # images come in (bs, channels, height, width)
        
        # Get per-item I2V mode indicators
        is_i2v_modes = batch.get_is_i2v_mode_list()
        i2v_indices = [i for i, mode in enumerate(is_i2v_modes) if mode]
        t2v_indices = [i for i, mode in enumerate(is_i2v_modes) if not mode]
        
        # Determine batch type
        is_mixed_batch = len(i2v_indices) > 0 and len(t2v_indices) > 0
        pure_i2v = len(i2v_indices) == latent_model_input.shape[0]
        
        with torch.no_grad():
            # Handle pure I2V case (original behavior - no overhead)
            if pure_i2v and batch.dataset_config.do_i2v:
                if batch.first_frame_latents is not None:
                    first_frame_latents = batch.first_frame_latents.to(
                        self.device_torch, self.torch_dtype
                    )
                    conditioned_latent = self._apply_first_frame_conditioning_cached(
                        latent_model_input, first_frame_latents
                    )
                else:
                    frames = batch.tensor
                    if frames is None:
                        raise ValueError("batch.tensor is None and batch.first_frame_latents is None. Cannot compute I2V conditioning. Ensure cache_latents_to_disk is working or do_i2v is correctly set.")
                    if len(frames.shape) == 4:
                        first_frames = frames
                    elif len(frames.shape) == 5:
                        first_frames = frames[:, 0]
                    else:
                        raise ValueError(f"Unknown frame shape {frames.shape}")
                    conditioned_latent = add_first_frame_conditioning(
                        latent_model_input=latent_model_input,
                        first_frame=first_frames,
                        vae=self.vae
                    )
            else:
                # Mixed or pure T2V: build conditioned_latent by concatenating processed items
                # This avoids cloning and handles channel dimension change (16 -> 36 for I2V)
                i2v_processed = []
                t2v_processed = []
                
                # Process I2V items
                if i2v_indices and batch.dataset_config.do_i2v:
                    i2v_latents = latent_model_input[i2v_indices]
                    if batch.first_frame_latents is not None:
                        # batch.first_frame_latents contains only I2V items' first frame latents
                        # (filtered at batch creation level, T2V items are excluded)
                        i2v_first_frame_latents = batch.first_frame_latents.to(
                            self.device_torch, self.torch_dtype
                        )
                        i2v_processed = [self._apply_first_frame_conditioning_cached(
                            i2v_latents, i2v_first_frame_latents
                        )]
                    else:
                        frames = batch.tensor
                        if frames is not None:
                            first_frames = frames[i2v_indices, 0] if len(frames.shape) == 5 else frames
                            i2v_processed = [add_first_frame_conditioning(
                                latent_model_input=i2v_latents,
                                first_frame=first_frames,
                                vae=self.vae
                            )]
                
                # Process T2V items
                if t2v_indices:
                    if is_mixed_batch:
                        # Pad with zero conditioning to match I2V shape
                        t2v_latents = latent_model_input[t2v_indices]
                        t2v_processed = [self._pad_for_no_conditioning(t2v_latents)]
                    else:
                        # Pure T2V: still pad with zero conditioning so the channel
                        # dimension matches what the I2V model's patch_embedding
                        # expects (16 latent + 4 mask + 16 latent = 36 channels)
                        t2v_processed = [self._pad_for_no_conditioning(latent_model_input)]
                
                # Combine in original order
                if i2v_processed and t2v_processed:
                    # Mixed: interleave based on original indices
                    i2v_result = i2v_processed[0]
                    t2v_result = t2v_processed[0]
                    conditioned_latent = latent_model_input.new_empty(
                        (len(is_i2v_modes), i2v_result.shape[1]) + i2v_result.shape[2:]
                    )
                    for idx, orig_idx in enumerate(i2v_indices):
                        conditioned_latent[orig_idx] = i2v_result[idx]
                    for idx, orig_idx in enumerate(t2v_indices):
                        conditioned_latent[orig_idx] = t2v_result[idx]
                elif i2v_processed:
                    conditioned_latent = i2v_processed[0]
                else:
                    conditioned_latent = t2v_processed[0]
        
        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred