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
		timestep: torch.Tensor,
		text_embeddings: PromptEmbeds,
		batch: DataLoaderBatchDTO,
		**kwargs
	):
		with torch.no_grad():
			frames = batch.tensor
			
			if frames is not None:
				# Standard path: Pixels are available in RAM
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
			else:
				# Cached path: Pixels are None, construct 36-channel input from cached latents
				cached_latents = batch.latents  # Shape: (B, 16, T, H, W)
				B, C, T, H, W = cached_latents.shape
				
				# Extract the first frame's latent representation
				first_frame_latent = cached_latents[:, :, 0:1, :, :]
				
				# Broadcast the first frame latent across all T temporal frames
				first_frame_latent_rep = first_frame_latent.repeat(1, 1, T, 1, 1)
				
				# Create the 4-channel temporal mask (1.0 for the first frame, 0.0 for the rest)
				mask = torch.zeros((B, 4, T, H, W), device=cached_latents.device, dtype=cached_latents.dtype)
				mask[:, :, 0:1, :, :] = 1.0
				
				# Concatenate to form the 20-channel conditioning tensor
				conditioning = torch.cat([first_frame_latent_rep, mask], dim=1)
				
				# Concatenate with the 16-channel noise latent to form the 36-channel input
				conditioned_latent = torch.cat([latent_model_input, conditioning], dim=1)

		noise_pred = self.model(
			hidden_states=conditioned_latent,
			timestep=timestep,
			encoder_hidden_states=text_embeddings.text_embeds,
			return_dict=False,
			**kwargs
		)[0]

		return noise_pred