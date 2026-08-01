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
        with torch.no_grad():
            conditioned_latent = latent_model_input
            if batch.dataset_config.do_i2v:
                if batch.first_frame_latents is not None:
                    # Use cached first frame latents
                    first_frame_latents = batch.first_frame_latents.to(
                        self.device_torch, self.torch_dtype
                    )
                    
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
                    conditioned_latent = torch.cat(
                        [latent_model_input, first_frame_condition], dim=1)
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
                    
                    # Add conditioning using the standalone function
                    conditioned_latent = add_first_frame_conditioning(
                        latent_model_input=latent_model_input,
                        first_frame=first_frames,
                        vae=self.vae
                    )
        
        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred