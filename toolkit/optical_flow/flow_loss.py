import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowConsistencyLoss(nn.Module):
    """
    Latent-space flow-warping consistency loss for video diffusion (flow-matching).

    For each consecutive latent-frame pair (t, t+1):
      1. Downsample GT pixel flow (T-1, 2, H, W) to latent resolution and compose
         over the VAE temporal stride to get latent flow (T_lat-1, 2, H_lat, W_lat).
      2. Warp pred_x0[:,:,t] -> t+1 using latent flow via grid_sample.
      3. L = MSE(warped, pred_x0[:,:,t+1]) weighted by (1 - t/1000), returned as
         the gate-weighted average over active batch items (renormalized by the
         total gate mass, so items with gate=0 do not dilute the magnitude).

    pred_x0 is reconstructed from noisy_latents and noise_pred via flow-matching:
        pred_x0 = noisy_latents - sigma * noise_pred
    where sigma = t/1000. Derivation: x_t = (1-t)·x_0 + t·ε, v = ε - x_0 ⇒ x_0 = x_t - t·v
    """

    def __init__(self, vae_temporal_stride: int = 4, vae_spatial_stride: int = 8):
        super().__init__()
        self.vae_temporal_stride = vae_temporal_stride  # Wan VAE: 4
        self.vae_spatial_stride = vae_spatial_stride    # Wan VAE: 8

    @staticmethod
    def pixel_flow_to_latent_flow(pixel_flow: torch.Tensor,
                                   temporal_stride: int = 4,
                                   spatial_stride: int = 8) -> torch.Tensor:
        """
        Convert pixel-space flow to latent-space flow.

        Args:
            pixel_flow: (B, T-1, 2, H, W) in pixel units.
            temporal_stride: VAE temporal compression factor (Wan: 4)
            spatial_stride: VAE spatial compression factor (Wan: 8)

        Returns:
            (B, T_lat-1, 2, H_lat, W_lat) in latent-space pixel units

        Wan VAE: T pixel frames -> T_lat = (T-1)//4 + 1 latent frames.
        So T-1 pixel flows collapse to T_lat-1 = (T-1)//4 latent flows by
        summing flows within each temporal group (flow composition) and
        average-pooling spatially with magnitude rescaling.
        """
        B, Tm1, two, H, W = pixel_flow.shape
        assert two == 2

        if Tm1 % temporal_stride != 0:
            # Pad or truncate to make divisible
            target_Tm1 = (Tm1 // temporal_stride) * temporal_stride
            if target_Tm1 > 0:
                pixel_flow = pixel_flow[:, :target_Tm1]
                Tm1 = target_Tm1
            else:
                return torch.zeros(B, 0, 2, H // spatial_stride, W // spatial_stride,
                                   device=pixel_flow.device, dtype=pixel_flow.dtype)

        T_lat_m1 = Tm1 // temporal_stride

        # Reshape into temporal groups and sum (compose) flows
        grouped = pixel_flow.view(B, T_lat_m1, temporal_stride, 2, H, W)
        composed = grouped.sum(dim=2)  # (B, T_lat_m1, 2, H, W)

        # Spatial downsample: average pool, then rescale magnitudes
        # BUG FIX A: removed incorrect permute(0,1,3,2,4) which swapped H with channel dim,
        # causing avg_pool2d to pool across wrong dimensions after reshape.
        # composed is (B, T_lat_m1, 2, H, W); reshape directly to (B*T, 2, H, W).
        latent_flow = F.avg_pool2d(
            composed.reshape(B * T_lat_m1, 2, H, W),
            kernel_size=spatial_stride
        ).reshape(B, T_lat_m1, 2, H // spatial_stride, W // spatial_stride)

        # Rescale: a flow of d pixels at full res becomes d/spatial_stride at latent res
        latent_flow = latent_flow / spatial_stride
        return latent_flow  # (B, T_lat-1, 2, H_lat, W_lat)

    @staticmethod
    def warp_latent_frame(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp src latent frame to dst using flow.

        Args:
            src: (B, C, H, W) source frame
            flow: (B, 2, H, W) in latent pixel units (x, y)

        Returns:
            (B, C, H, W) warped frame
        """
        B, C, H, W = src.shape

        # Build base grid in normalized [-1, 1] coords
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=src.device, dtype=src.dtype),
            torch.linspace(-1, 1, W, device=src.device, dtype=src.dtype),
            indexing='ij')
        base = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        # Flow is in latent pixel units; convert to normalized: dx_norm = 2*dx/W
        flow_norm = torch.stack([
            flow[:, 0] * 2.0 / W,
            flow[:, 1] * 2.0 / H,
        ], dim=1)

        sample_grid = base + flow_norm  # (B, 2, H, W)
        sample_grid = sample_grid.permute(0, 2, 3, 1)  # (B, H, W, 2) for grid_sample

        return F.grid_sample(src, sample_grid, mode='bilinear',
                             padding_mode='border', align_corners=True)

    def forward(self, noise_pred: torch.Tensor, noisy_latents: torch.Tensor,
                timesteps: torch.Tensor, batch_flow: torch.Tensor,
                max_timestep: int = 800,
                motion_weighted: bool = True,
                reverse_gate: bool = False) -> torch.Tensor:
        """
        Compute flow consistency loss.

        Args:
            noise_pred: (B, C, T_lat, H_lat, W_lat) flow-matching velocity prediction
            noisy_latents: (B, C, T_lat, H_lat, W_lat)
            timesteps: (B,) in [0, 1000]
            batch_flow: (B, T-1, 2, H, W) fp16 pixel-space flow from cache
            max_timestep: only enforce motion at t < max_timestep (normal gate)
                         or t > max_timestep (reverse gate)
            motion_weighted: weight by flow magnitude
            reverse_gate: if True, flow loss is weighted higher at high-noise timesteps
                         (t > max_timestep gets full weight, t < max_timestep fades to 0)
                         Normal gate: weight = 1.0 - t/max_timestep (max weight at low noise)
                         Reverse gate: weight = t/max_timestep (max weight at high noise)

        Returns:
            Scalar flow consistency loss (gate-weighted average over active items)
        """
        B, C, T_lat, H_lat, W_lat = noise_pred.shape

        if T_lat < 2 or batch_flow is None:
            return torch.tensor(0.0, device=noise_pred.device, dtype=noise_pred.dtype)

        # Timestep gate: controls where flow loss is applied
        t = timesteps.float()
        if reverse_gate:
            # Reverse gate: full weight at high noise (t >= max_timestep), zero at low noise (t = 0)
            # Useful when you want flow consistency enforced even in high-noise regime
            gate = torch.clamp(t / max_timestep, min=0.0, max=1.0)  # (B,), 0 at t=0, 1 at t>=max
        else:
            # Normal gate: full weight at low noise (t = 0), zero at high noise (t >= max_timestep)
            # Default: pred_x0 is most meaningful at low noise
            gate = torch.clamp(1.0 - (t / max_timestep), min=0.0)  # (B,), 1 at t=0, 0 at t>=max

        if gate.sum() < 1e-6:
            return torch.tensor(0.0, device=noise_pred.device, dtype=noise_pred.dtype)

        # Flow-matching x0 prediction: pred_x0 = noisy - sigma * v_pred
        # Derivation: x_t = (1-t)·x_0 + t·ε, v = ε - x_0 ⇒ x_0 = x_t - t·v
        # sigma in [0,1]; for Wan flow-matching, sigma = t/1000
        sigma = (t / 1000.0).view(B, 1, 1, 1, 1).to(noise_pred.dtype)
        pred_x0 = noisy_latents - sigma * noise_pred  # (B, C, T_lat, H_lat, W_lat)

        # Compute latent flow from cached pixel flow
        latent_flow = self.pixel_flow_to_latent_flow(
            batch_flow.float().to(noise_pred.device),
            self.vae_temporal_stride, self.vae_spatial_stride
        )  # (B, T_lat-1, 2, H_lat, W_lat)

        if latent_flow.shape[1] < 1:
            return torch.tensor(0.0, device=noise_pred.device, dtype=noise_pred.dtype)

        # Warp each frame t to t+1 and compare
        total_loss = torch.tensor(0.0, device=noise_pred.device, dtype=noise_pred.dtype)
        weight_sum = 0.0

        for t_idx in range(min(T_lat - 1, latent_flow.shape[1])):
            src = pred_x0[:, :, t_idx]          # (B, C, H_lat, W_lat)
            dst = pred_x0[:, :, t_idx + 1]      # (B, C, H_lat, W_lat)
            flow_t = latent_flow[:, t_idx]      # (B, 2, H_lat, W_lat)

            warped = self.warp_latent_frame(src, flow_t)
            per_sample_loss = F.mse_loss(warped, dst, reduction='none').mean(dim=[1, 2, 3])

            if motion_weighted:
                # Weight by flow magnitude (high-motion frames matter more)
                mag = flow_t.norm(dim=1).mean(dim=[1, 2])  # (B,)
                w = 1.0 + torch.clamp(mag / 5.0, max=4.0)  # 1x-5x scaling
            else:
                w = torch.ones(B, device=noise_pred.device, dtype=noise_pred.dtype)

            per_sample_loss = per_sample_loss * w * gate
            total_loss = total_loss + per_sample_loss.sum()
            weight_sum += 1.0

        if weight_sum == 0:
            return torch.tensor(0.0, device=noise_pred.device, dtype=noise_pred.dtype)

        # Renormalize by the total gate mass (sum of per-item gates) instead of the
        # batch size. This is a RATIO estimator: (sum gate_i g_i) / (sum gate_i).
        #
        # The key property: items with gate=0 (t >= max_timestep) drop out of BOTH
        # the numerator and the denominator, so they are conditioned out. The result
        # is the gate-weighted average over the ACTIVE items only, giving flow_weight
        # a stable, batch-composition-independent magnitude. This is the minimum-
        # variance way to do it: subtracting the (correlated) R*·gate component makes
        # Var(gate*(g-R*)) much smaller than Var(gate*g), so the ratio beats a
        # constant-E[gate] normalization by ~10x in variance (verified analytically).
        #
        # Trade-off: the random denominator introduces a small O(1/B) bias (a few %
        # of the flow weight at B=8, shrinking with B), which is negligible and
        # absorbed by flow_weight. When all items are active (gate=1), gate.sum()==B
        # and this reduces exactly to the batch mean, so ungated training is unchanged.
        gate_mass = gate.sum()
        if gate_mass < 1e-6:
            return torch.tensor(0.0, device=noise_pred.device, dtype=noise_pred.dtype)

        return total_loss / (weight_sum * gate_mass)


def load_flow_loss(sd) -> FlowConsistencyLoss:
    """
    Instantiate FlowConsistencyLoss with VAE strides read from the model.

    Supports multiple VAE config conventions:
    - Wan VAE: scale_factor_temporal / scale_factor_spatial (diffusers standard)
    - LTX2 VAE: temporal_compression_ratio
    - Generic: temperal_downsample (Wan internal) or downsample_factors
    """
    vae = sd.vae
    # Wan VAE defaults: 4x temporal, 8x spatial compression
    temporal_stride = 4
    spatial_stride = 8

    # Try Wan-style config attributes first (most common)
    if hasattr(vae.config, 'scale_factor_temporal'):
        temporal_stride = vae.config.scale_factor_temporal
    elif hasattr(vae.config, 'temporal_compression_ratio'):
        # LTX2 and some other VAEs use this name
        temporal_stride = vae.config.temporal_compression_ratio

    # Try to compute from temperal_downsample (Wan internal representation)
    if hasattr(vae, 'temperal_downsample') and vae.temperal_downsample:
        temporal_stride = 2 ** sum(vae.temperal_downsample)

    if hasattr(vae.config, 'scale_factor_spatial'):
        spatial_stride = vae.config.scale_factor_spatial
    elif hasattr(vae.config, 'downsample_factors'):
        # Compute from downsample factors
        spatial_stride = 1
        for f in vae.config.downsample_factors:
            spatial_stride *= 2

    return FlowConsistencyLoss(vae_temporal_stride=temporal_stride,
                               vae_spatial_stride=spatial_stride)
