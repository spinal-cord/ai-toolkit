import torch
import torch.nn.functional as F


_dwt = None


def _get_wavelet_loss(device, dtype):
    global _dwt
    if _dwt is not None:
        return _dwt

    # init wavelets
    from pytorch_wavelets import DWTForward

    # wave='db1'  wave='haar'
    dwt = DWTForward(J=1, mode="zero", wave="haar").to(device=device, dtype=dtype)
    _dwt = dwt
    return dwt


def wavelet_loss(model_pred, latents, noise):
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()
    dwt = _get_wavelet_loss(model_pred.device, model_pred.dtype)
    with torch.no_grad():
        model_input_xll, model_input_xh = dwt(latents)
        model_input_xlh, model_input_xhl, model_input_xhh = torch.unbind(
            model_input_xh[0], dim=2
        )
        model_input = torch.cat(
            [model_input_xll, model_input_xlh, model_input_xhl, model_input_xhh], dim=1
        )

    # reverse the noise to get the model prediction of the pure latents
    model_pred = noise - model_pred

    model_pred_xll, model_pred_xh = dwt(model_pred)
    model_pred_xlh, model_pred_xhl, model_pred_xhh = torch.unbind(
        model_pred_xh[0], dim=2
    )
    model_pred = torch.cat(
        [model_pred_xll, model_pred_xlh, model_pred_xhl, model_pred_xhh], dim=1
    )

    return torch.nn.functional.mse_loss(model_pred, model_input, reduction="none")


def stepped_loss(model_pred, latents, noise, noisy_latents, timesteps, scheduler):
    # this steps the on a 20 step timescale from the current step (50 idx steps ahead)
    # and then reconstructs the original image at that timestep. This should lessen the error
    # possible in high noise timesteps and make the flow smoother.
    bs = model_pred.shape[0]

    noise_pred_chunks = torch.chunk(model_pred, bs)
    timestep_chunks = torch.chunk(timesteps, bs)
    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
    noise_chunks = torch.chunk(noise, bs)

    x0_pred_chunks = []

    for idx in range(bs):
        model_output = noise_pred_chunks[idx]  # predicted noise (same shape as latent)
        timestep = timestep_chunks[idx]  # scalar tensor per sample (e.g., [t])
        sample = noisy_latent_chunks[idx].to(torch.float32)
        noise_i = noise_chunks[idx].to(sample.dtype).to(sample.device)

        # Initialize scheduler step index for this sample
        scheduler._step_index = None
        scheduler._init_step_index(timestep)

        # ---- Step +50 indices (or to the end) in sigma-space ----
        sigma = scheduler.sigmas[scheduler.step_index]
        target_idx = min(scheduler.step_index + 50, len(scheduler.sigmas) - 1)
        sigma_next = scheduler.sigmas[target_idx]

        # One-step update along the model-predicted direction
        stepped = sample + (sigma_next - sigma) * model_output

        # ---- Inverse-Gaussian recovery at the target timestep ----
        t_01 = (
            (scheduler.sigmas[target_idx]).to(stepped.device).to(stepped.dtype)
        )
        original_samples = (stepped - t_01 * noise_i) / (1.0 - t_01)
        x0_pred_chunks.append(original_samples)

    predicted_images = torch.cat(x0_pred_chunks, dim=0)

    return torch.nn.functional.mse_loss(
        predicted_images.float(),
        latents.float().to(device=predicted_images.device),
        reduction="none",
    )


def _create_radial_frequency_masks(shape_h, shape_w, low_cutoff=0.15, high_cutoff=0.5):
    """
    Create radial frequency masks for low/mid/high frequency bands.
    
    Args:
        shape_h: Height dimension of FFT output
        shape_w: Width dimension of FFT output
        low_cutoff: Normalized radius (0-1) separating low and mid frequencies
        high_cutoff: Normalized radius (0-1) separating mid and high frequencies
    
    Returns:
        Tuple of (low_mask, mid_mask, high_mask) - smooth radial masks summing to 1
    """
    # Create coordinate grids normalized to [0, 1]
    y = torch.linspace(0, 1, shape_h, dtype=torch.float32)
    x = torch.linspace(0, 1, shape_w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Center the coordinates
    yy = (yy - 0.5).abs()
    xx = (xx - 0.5).abs()
    
    # Compute radial distance (normalized to max possible radius)
    radius = torch.sqrt(xx**2 + yy**2)
    max_radius = torch.sqrt(torch.tensor(0.5**2 + 0.5**2, dtype=radius.dtype, device=radius.device))  # corner distance
    radius = radius / max_radius
    
    # Create smooth masks using sigmoid transitions
    sigma = 5.0  # controls transition smoothness
    
    # Low frequency mask (center)
    low_mask = torch.sigmoid(sigma * (low_cutoff - radius))
    
    # High frequency mask (edges)
    high_mask = torch.sigmoid(sigma * (radius - high_cutoff))
    
    # Mid frequency mask (between low and high)
    mid_mask = 1.0 - low_mask - high_mask
    mid_mask = torch.clamp(mid_mask, min=0.0)
    
    # Normalize so masks sum to 1
    total = low_mask + mid_mask + high_mask + 1e-8
    low_mask = low_mask / total
    mid_mask = mid_mask / total
    high_mask = high_mask / total
    
    return low_mask, mid_mask, high_mask


def _calculate_lcr_loss_ssvae_style(latents, patch_size=2, alpha=0.75):
    """
    SSVAE Local Correlation Regularization (LCR) - Paper-accurate implementation.
    
    From "Delving into Latent Spectral Biasing of Video VAEs for Superior Diffusability"
    Section 4, Eq. 3-4.
    
    Measures average Pearson correlation within local spatio-temporal patches.
    Higher correlation = more low-frequency energy = better diffusability.
    
    Key design choices from paper:
    - Patch size 2×2×2 for spatio-temporal patches
    - First frame: spatial patches only (no temporal)
    - Remaining frames: full spatio-temporal patches
    - Applied at BATCH level (not per-sample)
    - Uses cosine similarity (= Pearson correlation on normalized data)
    
    Args:
        latents: (B, C, T, H, W) for video or (B, C, H, W) for image
        patch_size: Size of local patches (default 2 per SSVAE)
        alpha: Threshold for hinge loss (default 0.75 per SSVAE)
    
    Returns:
        Scalar LCR loss (ReLU(α - avg_local_corr))
    """
    is_video = len(latents.shape) == 5
    
    # Per-channel standardization (match diffusion preprocessing)
    # Paper: "mean and variance of each channel computed over (B, T, H, W)"
    if is_video:
        mean = latents.mean(dim=(0, 2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        std = latents.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
    else:
        mean = latents.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
        std = latents.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    
    z = (latents - mean) / std  # Normalized latents
    
    if is_video:
        B, C, T, H, W = z.shape
        
        if H < patch_size or W < patch_size:
            return torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
        
        all_corrs = []
        
        # First frame: spatial patches only (no temporal dimension)
        z_first = z[:, :, 0:1, :, :]  # (B, C, 1, H, W)
        z_first = z_first.view(B, C, H, W)
        
        # Unfold spatial patches
        z_first_patches = z_first.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # (B, C, num_h_patches, num_w_patches, patch_size, patch_size)
        z_first_patches = z_first_patches.permute(0, 2, 3, 4, 5, 1)
        # (B, num_h, num_w, patch_size, patch_size, C)
        z_first_flat = z_first_patches.reshape(-1, patch_size*patch_size, C)
        # (all_patches, patch_pixels, C)
        
        # Calculate pairwise Pearson correlation within each patch
        # Using cosine similarity on normalized data
        corrs_first = _patchwise_pearson_correlation(z_first_flat)
        all_corrs.append(corrs_first)
        
        # Remaining frames: spatio-temporal patches
        if T > patch_size:  # Need more than patch_size frames total
            z_rest = z[:, :, 1:, :, :]  # (B, C, T-1, H, W)
            T_rest = z_rest.shape[2]
            
            # Need at least patch_size frames for spatio-temporal patches
            if T_rest >= patch_size:
                # Unfold temporal dimension
                z_rest_t = z_rest.unfold(2, patch_size, patch_size)  # (B, C, num_t, H, W)
                # Then unfold spatial
                z_rest_patches = z_rest_t.unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
                # (B, C, num_t, num_h, num_w, patch_size, patch_size, patch_size)
                z_rest_patches = z_rest_patches.permute(0, 2, 3, 4, 5, 6, 7, 1)
                # (B, num_t, num_h, num_w, patch_size, patch_size, patch_size, C)
                z_rest_flat = z_rest_patches.reshape(-1, patch_size**3, C)
                
                corrs_rest = _patchwise_pearson_correlation(z_rest_flat)
                all_corrs.append(corrs_rest)
        
        if len(all_corrs) == 0:
            return torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
        
        # Handle case where correlations are scalars (0-dim tensors)
        # _patchwise_pearson_correlation returns a scalar mean
        scalar_corrs = [c.item() if c.numel() == 1 else c for c in all_corrs]
        avg_corr = torch.tensor(sum(scalar_corrs) / len(scalar_corrs),
                               device=latents.device, dtype=latents.dtype)
    
    else:
        # Image case: (B, C, H, W)
        B, C, H, W = z.shape
        
        if H < patch_size or W < patch_size:
            return torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
        
        z_patches = z.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # (B, C, num_h, num_w, patch_size, patch_size)
        z_patches = z_patches.permute(0, 2, 3, 4, 5, 1)
        # (B, num_h, num_w, patch_size, patch_size, C)
        z_flat = z_patches.reshape(-1, patch_size*patch_size, C)
        
        avg_corr = _patchwise_pearson_correlation(z_flat)
    
    # LCR loss: ReLU(α - avg_local_corr)
    # Paper uses hinge loss to prevent over-smoothing
    lcr_loss = torch.relu(alpha - avg_corr)
    return lcr_loss


def _patchwise_pearson_correlation(patch_vectors):
    """
    Calculate average pairwise Pearson correlation within patches.
    
    Args:
        patch_vectors: (num_patches, num_elements, channels)
    
    Returns:
        Scalar average correlation
    """
    # patch_vectors shape: (P, N, C) where N = patch_size^2 or patch_size^3
    P, N, C = patch_vectors.shape
    
    if N < 2:
        return torch.tensor(0.0, device=patch_vectors.device, dtype=patch_vectors.dtype)
    
    # Already normalized (zero mean, unit var per channel from batch)
    # Cosine similarity = Pearson correlation for normalized vectors
    # Calculate pairwise similarities within each patch
    
    # Efficient: use batch matrix multiplication
    # sim[i,j] = cosine_sim(v_i, v_j) for all pairs in patch
    sim = torch.matmul(patch_vectors, patch_vectors.transpose(1, 2))  # (P, N, N)
    
    # Take upper triangle (exclude diagonal and lower triangle)
    mask = torch.triu(torch.ones(N, N, device=patch_vectors.device), diagonal=1)
    mask = mask.unsqueeze(0).expand(P, -1, -1)  # (P, N, N)
    
    # Average over all pairs
    corr_sum = (sim * mask).sum(dim=(1, 2))  # (P,)
    num_pairs = mask.sum(dim=(1, 2))  # (P,) = N*(N-1)/2 per patch
    
    avg_corr = (corr_sum / (num_pairs + 1e-8)).mean()
    return avg_corr


def spectral_loss(
    model_pred,
    latents,
    noise,
    low_weight=1.0,
    mid_weight=1.0,
    high_weight=2.0,
    low_cutoff=0.15,
    high_cutoff=0.5,
    use_phase=True,
    lcr_weight=0.0,
):
    """
    Spectral Training Loss with frequency dissociation and balancing.
    
    Implements advanced spectral balancing via FFT analysis to dissociate
    low frequencies (structure/motion) from high frequencies (texture/details).
    Inspired by SSVAE research on latent spectral biasing for superior diffusability.
    
    This loss:
    1. Decomposes prediction and target into frequency bands via FFT
    2. Applies independent MSE loss per frequency band
    3. Weighted combination allows emphasizing texture while preserving structure
    4. Optional LCR (Local Correlation Regularization) for low-frequency bias
    5. Works with both 4D (image: B,C,H,W) and 5D (video: B,C,T,H,W) tensors
    6. Returns loss in original tensor shape for proper masking
    
    Args:
        model_pred: Model's noise prediction (B, C, H, W) or (B, C, T, H, W)
        latents: Target clean latents
        noise: Added noise (used to reconstruct predicted latents)
        low_weight: Loss weight for low frequencies (structure/motion)
        mid_weight: Loss weight for mid frequencies
        high_weight: Loss weight for high frequencies (texture/details)
        low_cutoff: Radius (0-1) separating low from mid frequencies
        high_cutoff: Radius (0-1) separating mid from high frequencies
        use_phase: If True, also penalize phase differences (more accurate but slower)
        lcr_weight: SSVAE-inspired Local Correlation Regularization weight (0.0 = disabled)
                   Encourages low-frequency bias in latents for better diffusability
    
    Returns:
        Loss tensor with same shape as input latents
    """
    # Convert to float32 for FFT stability
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()
    
    # Reverse noise prediction to get predicted clean latents: x0_pred = noise - noise_pred
    pred_latents = noise - model_pred
    
    # Determine if video (5D) or image (4D)
    is_video = len(latents.shape) == 5
    
    if is_video:
        # Video: (B, C, T, H, W)
        # Apply FFT on spatial dimensions (H, W) for each frame
        batch_size, channels, num_frames, height, width = latents.shape
        
        # Reshape to treat all frames as batch for FFT
        pred_reshaped = pred_latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        target_reshaped = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        
        # Apply 2D FFT on spatial dimensions
        pred_fft = torch.fft.rfft2(pred_reshaped, norm='ortho')
        target_fft = torch.fft.rfft2(target_reshaped, norm='ortho')
        
        fft_height, fft_width = pred_fft.shape[-2], pred_fft.shape[-1]
        
        # Create frequency masks
        low_mask, mid_mask, high_mask = _create_radial_frequency_masks(
            fft_height, fft_width, low_cutoff, high_cutoff
        )
        # Expand masks to match FFT shape: (1, 1, H_fft, W_fft)
        masks = (low_mask, mid_mask, high_mask)
        
        # Compute per-band losses
        band_losses = []
        weights = [low_weight, mid_weight, high_weight]
        
        for mask, weight in zip(masks, weights):
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).to(pred_fft.device)
            
            # Apply mask to isolate frequency band
            pred_band = pred_fft * mask_expanded
            target_band = target_fft * mask_expanded
            
            if use_phase:
                # Full complex loss (magnitude + phase)
                band_loss = F.mse_loss(pred_band.real, target_band.real, reduction='none') + \
                           F.mse_loss(pred_band.imag, target_band.imag, reduction='none')
            else:
                # Magnitude-only loss (faster, focuses on frequency content)
                pred_mag = pred_band.abs()
                target_mag = target_band.abs()
                band_loss = F.mse_loss(pred_mag, target_mag, reduction='none')
            
            band_losses.append(band_loss * weight)
        
        # Sum band losses
        total_loss_fft = torch.stack(band_losses, dim=1).sum(dim=1)
        
        # Inverse FFT to get loss in spatial domain
        # Create a complex tensor with loss magnitude
        loss_fft = torch.view_as_complex(
            torch.stack([total_loss_fft, torch.zeros_like(total_loss_fft)], dim=-1)
        )
        
        # Apply inverse FFT
        loss_spatial = torch.fft.irfft2(loss_fft, s=(height, width), norm='ortho').abs()
        
        # Reshape back to video shape
        loss_spatial = loss_spatial.reshape(batch_size, num_frames, channels, height, width)
        loss_spatial = loss_spatial.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
    else:
        # Image: (B, C, H, W)
        batch_size, channels, height, width = latents.shape
        
        # Apply 2D FFT
        pred_fft = torch.fft.rfft2(pred_latents, norm='ortho')
        target_fft = torch.fft.rfft2(latents, norm='ortho')
        
        fft_height, fft_width = pred_fft.shape[-2], pred_fft.shape[-1]
        
        # Create frequency masks
        low_mask, mid_mask, high_mask = _create_radial_frequency_masks(
            fft_height, fft_width, low_cutoff, high_cutoff
        )
        masks = (low_mask, mid_mask, high_mask)
        
        # Compute per-band losses
        band_losses = []
        weights = [low_weight, mid_weight, high_weight]
        
        for mask, weight in zip(masks, weights):
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).to(pred_fft.device)
            
            # Apply mask to isolate frequency band
            pred_band = pred_fft * mask_expanded
            target_band = target_fft * mask_expanded
            
            if use_phase:
                # Full complex loss (magnitude + phase)
                band_loss = F.mse_loss(pred_band.real, target_band.real, reduction='none') + \
                           F.mse_loss(pred_band.imag, target_band.imag, reduction='none')
            else:
                # Magnitude-only loss
                pred_mag = pred_band.abs()
                target_mag = target_band.abs()
                band_loss = F.mse_loss(pred_mag, target_mag, reduction='none')
            
            band_losses.append(band_loss * weight)
        
        # Sum band losses
        total_loss_fft = torch.stack(band_losses, dim=1).sum(dim=1)
        
        # Inverse FFT to get loss in spatial domain
        loss_fft = torch.view_as_complex(
            torch.stack([total_loss_fft, torch.zeros_like(total_loss_fft)], dim=-1)
        )
        
        # Apply inverse FFT
        loss_spatial = torch.fft.irfft2(loss_fft, s=(height, width), norm='ortho').abs()
    
    # SSVAE Local Correlation Regularization (LCR)
    # Paper-accurate implementation from Section 4
    # Encourages low-frequency bias in predicted latents for better diffusability
    if lcr_weight > 0.0:
        # Calculate LCR loss for predicted latents
        # Paper uses hinge loss: ReLU(α - avg_local_corr) with α=0.75
        lcr_loss_scalar = _calculate_lcr_loss_ssvae_style(
            pred_latents,
            patch_size=2,  # Paper default
            alpha=0.75     # Paper default
        )
        
        # Scale LCR loss to match spectral loss magnitude
        # Paper uses weight 0.02 with dynamic gradient-based weighting
        spectral_mean = loss_spatial.mean()
        lcr_loss_scaled = lcr_loss_scalar * lcr_weight * spectral_mean
        
        # Add as uniform penalty (LCR is a regularization, not spatially-varying)
        lcr_loss_uniform = lcr_loss_scaled * torch.ones_like(loss_spatial)
        loss_spatial = loss_spatial + lcr_loss_uniform
    
    return loss_spatial
