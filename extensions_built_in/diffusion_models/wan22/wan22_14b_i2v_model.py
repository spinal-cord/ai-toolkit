import torch
from typing import List
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
            # FIX: add_first_frame_conditioning returns a tuple (conditioned_latent, loss_mask).
            # We only need the conditioned_latent for inference.
            gen_config.latents, _ = add_first_frame_conditioning(
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
    ) -> tuple:
        """Apply first frame conditioning using cached first frame latents.
        
        Returns:
            (conditioned_latent, loss_mask) where loss_mask is (B, 1, T_lat, H_lat, W_lat)
            with 0 for conditioned tokens (first frame) and 1 for generated tokens.
            For single-frame (image) input, loss_mask is None since masking the only
            latent frame would zero all gradients.
        """
        device = latent_model_input.device
        dtype = latent_model_input.dtype
        
        batch_size = latent_model_input.shape[0]
        num_latent_frames = latent_model_input.shape[2]
        num_frames = (num_latent_frames - 1) * 4 + 1
        
        vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        latent_height = first_frame_latents.shape[3]
        latent_width = first_frame_latents.shape[4]
        
        # Build the conditioning mask used in the model input
        mask_lat_size = torch.ones(
            batch_size, 1, num_frames, latent_height, latent_width, device=device, dtype=dtype)
        mask_lat_size[:, :, 1:] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        
        # Pad first frame latents to match temporal dimension
        latent_condition = torch.zeros(
            batch_size, first_frame_latents.shape[1], num_latent_frames, latent_height, latent_width,
            device=device, dtype=dtype
        )
        latent_condition[:, :, 0:1] = first_frame_latents
        
        first_frame_condition = torch.concat(
            [mask_lat_size, latent_condition], dim=1)
        conditioned_latent = torch.cat([latent_model_input, first_frame_condition], dim=1)
        
        # Build loss mask: 0 for first latent frame (conditioned), 1 for all others
        # EXCEPTION: single-frame (image) datasets — don't mask loss or training breaks
        # entirely since there is only the first frame. The model learns to output
        # the conditioning image itself (T2V semantics, with I2V channel layout).
        if num_latent_frames == 1:
            return conditioned_latent, None

        loss_mask = torch.ones(
            batch_size, 1, num_latent_frames, latent_height, latent_width,
            device=device, dtype=dtype
        )
        loss_mask[:, :, 0:1] = 0  # zero out first frame
        
        return conditioned_latent, loss_mask

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

    def _condition_first_frame(
        self,
        latents: torch.Tensor,
        batch: DataLoaderBatchDTO,
        condition_mask: List[bool],
    ):
        """Apply first-frame I2V conditioning to a (single, undoubled) latent batch.

        For every item, image conditioning is applied iff ``condition_mask[i]`` is True
        (the mask already encodes "is an I2V item AND the image was not dropped").
        Items without conditioning get zero (no) conditioning so the I2V channel layout
        (16 latent + 4 mask + 16 latent = 36 for Wan22 14B) is preserved.

        Args:
            latents: (bs, C, T, H, W) noisy latents for one (undoubled) batch.
            batch: the training batch (source of first frames / cached latents).
            condition_mask: per-item boolean, length == bs. True => apply first-frame
                conditioning to that item.

        Returns:
            (conditioned_latent (bs, 36, T, H, W), loss_mask (bs, 1, T, H, W) or None)
            The loss_mask is 0 on the first latent frame of conditioned items and 1
            elsewhere (None for single-frame input where masking would zero all loss).
        """
        bs = latents.shape[0]
        # Only condition items the mask asks for (they are guaranteed to be I2V items)
        active = [i for i in range(bs) if condition_mask[i]]

        if not active:
            # Nothing to condition -> pure T2V layout for the whole batch
            return self._pad_for_no_conditioning(latents), None

        with torch.no_grad():
            active_latents = latents[active]
            if batch.first_frame_latents is not None:
                # first_frame_latents is aligned with the I2V items in original order;
                # map each active original index to its row in that tensor. Active items
                # are guaranteed to be I2V items, so this is a 1:1 in-order mapping.
                is_i2v = batch.get_is_i2v_mode_list()
                i2v_orig = [i for i in range(len(is_i2v)) if is_i2v[i]]
                idx_to_i2v = {i: k for k, i in enumerate(i2v_orig)}
                active_rows = [idx_to_i2v[i] for i in active]
                ff = batch.first_frame_latents.to(self.device_torch, self.torch_dtype)[active_rows]
                conditioned, _ = self._apply_first_frame_conditioning_cached(active_latents, ff)
            else:
                frames = batch.tensor
                if frames is None:
                    raise ValueError(
                        "batch.tensor is None and batch.first_frame_latents is None. "
                        "Cannot compute I2V conditioning. Ensure cache_latents_to_disk is "
                        "working or do_i2v is correctly set."
                    )
                first_frames = frames[active, 0] if len(frames.shape) == 5 else frames[active]
                conditioned, _ = add_first_frame_conditioning(
                    latent_model_input=active_latents, first_frame=first_frames, vae=self.vae
                )

            # Build the full output: base noisy latents + zero conditioning, then
            # overwrite the conditioned items with their (latents + first-frame) result.
            ch = conditioned.shape[1]
            out = latents.new_empty((bs, ch) + latents.shape[2:])
            out.zero_()
            out[:, :latents.shape[1]] = latents  # base noisy latents for every item
            if conditioned is not None:
                out[active] = conditioned

            # loss mask: train everything, except the first (conditioned) frame of
            # conditioned items. Single-frame input can't be masked (would zero all).
            num_latent_frames = latents.shape[2]
            if num_latent_frames <= 1:
                return out, None
            loss_mask = torch.ones(
                bs, 1, num_latent_frames, latents.shape[3], latents.shape[4],
                device=latents.device, dtype=latents.dtype,
            )
            for i in active:
                loss_mask[i, :, 0:1] = 0
            return out, loss_mask

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

        # Per-item image-conditioning masks for the positive (conditional) and negative
        # (unconditional) branches, length == undoubled batch size. The trainer computes
        # these from the conditioning-dropout config (image dropout, cfg_same_prompt, ...).
        image_cond_mask_pos = kwargs.pop('image_cond_mask_pos', None)
        image_cond_mask_neg = kwargs.pop('image_cond_mask_neg', None)

        # Per-item I2V mode indicators (always the ORIGINAL, undoubled batch size)
        is_i2v_modes = batch.get_is_i2v_mode_list()
        orig_bs = len(is_i2v_modes)
        full_bs = latent_model_input.shape[0]
        # Under training-time CFG the latents are doubled: [uncond_copy, cond_copy]
        cfg_active = (full_bs == 2 * orig_bs) and orig_bs > 0

        # Use explicit masks only when they align with the (undoubled) batch size; otherwise
        # fall back to the plain I2V/T2V split (no image dropout) so exotic batching modes
        # keep working.
        use_mask = (
            image_cond_mask_pos is not None
            and len(image_cond_mask_pos) == orig_bs
        )
        if use_mask:
            pos_mask = [bool(m) for m in image_cond_mask_pos]
            neg_mask = (
                [bool(m) for m in image_cond_mask_neg]
                if image_cond_mask_neg is not None and len(image_cond_mask_neg) == orig_bs
                else pos_mask
            )
        else:
            pos_mask = [bool(m) for m in is_i2v_modes]
            neg_mask = pos_mask

        # Initialize loss mask (used by scale_loss to zero out conditioned tokens)
        self._i2v_loss_mask = None

        if cfg_active:
            uncond_half = latent_model_input[:orig_bs]
            cond_half = latent_model_input[orig_bs:]
            uncond_conditioned, _ = self._condition_first_frame(uncond_half, batch, neg_mask)
            cond_conditioned, cond_mask = self._condition_first_frame(cond_half, batch, pos_mask)
            conditioned_latent = torch.cat([uncond_conditioned, cond_conditioned], dim=0)
            # the loss is computed on the combined (orig_bs) prediction -> cond mask
            self._i2v_loss_mask = cond_mask
        else:
            conditioned_latent, self._i2v_loss_mask = self._condition_first_frame(
                latent_model_input, batch, pos_mask,
            )

        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred

    def scale_loss(self, loss):
        """Zero out the loss on I2V conditioned tokens, renormalized so the loss
        magnitude matches unconditioned batches (masked mean).
        
        For Wan22 14B I2V, the first frame is conditioned (not generated), so
        we should not compute loss on those tokens. This ensures spectral_flow
        loss doesn't analyze conditioned-then-zeroed regions.
        """
        if self._i2v_loss_mask is not None:
            loss_mask = self._i2v_loss_mask.to(loss.device, dtype=loss.dtype)
            loss = loss * loss_mask / loss_mask.mean().clamp(min=1e-8)
            self._i2v_loss_mask = None
        return loss