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
        
        # Bug 2.4 fix: keep gradients end-to-end — stack tensors, don't use .item()
        avg_corr = torch.stack(all_corrs).mean()
    
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
    spectral_transform='dct',  # 'dct' (default, SSVAE-compliant) or 'fft'
    prediction_target='velocity',  # 'velocity' or 'x0'
    temporal_scale=0.3,  # Scale temporal frequency for video (0.0-1.0)
):
    """
    Spectral Training Loss with frequency dissociation and balancing.
    
    Implements advanced spectral balancing via FFT or DCT analysis to dissociate
    low frequencies (structure/motion) from high frequencies (texture/details).
    Inspired by SSVAE research on latent spectral biasing for superior diffusability.
    
    This loss:
    1. Decomposes prediction and target into frequency bands via FFT or DCT
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
        spectral_transform: 'dct' for DCT-based (SSVAE-compliant, default), 'fft' for FFT-based
        prediction_target: 'velocity' (model predicts ε - x₀) or 'x0' (model predicts x₀ directly)
        temporal_scale: For video, weight for temporal frequency in 3D mask (0.0-1.0).
                       1.0 = all dims equal (can cause motion artifacts)
                       0.3 = recommended for video (temporal down-weighted)
                       0.0 = spatial-only (no temporal frequency penalty)
    
    Returns:
        Loss tensor with same shape as input latents
    """
    # Early exit: if all weights are zero, skip computation entirely
    if low_weight == 0 and mid_weight == 0 and high_weight == 0:
        return torch.zeros_like(latents)

    # Convert to float32 for FFT stability
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()

    # Reconstruct predicted clean latents based on prediction target:
    # - velocity: model_pred = ε - x₀ ⇒ x₀ = ε - model_pred = noise - model_pred
    # - x0: model_pred = x₀ directly
    if prediction_target == 'x0':
        pred_latents = model_pred
    else:
        pred_latents = noise - model_pred
    
    # Determine if video (5D) or image (4D)
    is_video = len(latents.shape) == 5

    if is_video:
        # Video: (B, C, T, H, W)
        # Use 3D FFT over (T, H, W) for spatio-temporal frequency analysis.
        # Per SSVAE paper: "we adopt a 3D DCT to analyze the spatio-temporal
        # frequency spectrum... [2D-only methods] do not adequately address
        # the temporal dimension in video latents."
        if spectral_transform == 'dct':
            loss_spatial = _spectral_loss_3d_video_dct(
                pred_latents, latents,
                low_weight, mid_weight, high_weight,
                low_cutoff, high_cutoff,
                use_phase
            )
        else:
            # Default: FFT
            loss_spatial = _spectral_loss_3d_video(
                pred_latents, latents,
                low_weight, mid_weight, high_weight,
                low_cutoff, high_cutoff,
                use_phase
            )

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


def _dct_1d(x: torch.Tensor, n: int = None, dim: int = -1, norm: str = 'ortho') -> torch.Tensor:
    """
    1D DCT-II using FFT (fallback when torch.dct is unavailable).

    Implements the standard FFT-based DCT algorithm:
      1. Create even-symmetric extension: y[k] = x[k] for k<N, y[2N-1-k] = x[k]
      2. Apply FFT of length 2N
      3. Extract first N coefficients with phase correction
      4. Scale by 0.5 to compensate for the mirror-concatenation doubling the signal.

    NOTE [SCALING BUG FIX]: The original version missed step 4. Concatenating x with
    flip(x) creates length 2N where every sample is counted twice, making the FFT result
    exactly 2× larger than the mathematical DCT-II. For spectral loss this only changed
    absolute magnitude uniformly across bands, but paired with _idct_1d it broke the
    roundtrip. The fix is a single result *= 0.5 (see implementation).

    Args:
        x: Input tensor
        n: Transform length (defaults to input size along dim)
        dim: Dimension along which to compute DCT
        norm: Normalization mode ('ortho' for orthonormal, or None)

    Returns:
        DCT-II transform along specified dimension
    """
    if n is None:
        n = x.shape[dim]

    ndim = x.dim()
    dim = dim % ndim  # Normalize negative dim

    # Create even-symmetric extension of length 2n
    # y = [x[0], x[1], ..., x[n-1], x[n-1], x[n-2], ..., x[0]]
    x_flipped = x.flip(dim)
    y = torch.cat([x, x_flipped], dim=dim)

    # Apply FFT of length 2n
    Y = torch.fft.fft(y, n=2 * n, dim=dim)

    # Extract first n coefficients and apply phase correction
    # Build broadcastable shape for k: 1 everywhere except dim
    k_shape = [1] * ndim
    k_shape[dim] = n
    k = torch.arange(n, device=x.device, dtype=torch.float32).view(k_shape)
    phase = torch.exp(-1j * torch.pi * k / (2 * n))

    # Take first n FFT coefficients
    slices = [slice(None)] * ndim
    slices[dim] = slice(0, n)
    Y_n = Y[tuple(slices)]

    result = (Y_n * phase).real

    # SCALE FIX: The even-symmetric mirror concatenation gives length 2N with each element
    # counted twice, so the FFT-based result is exactly 2× larger than the mathematical DCT.
    # Correcting here so coefficients match scipy.fft.dct and compose properly with _idct_1d.
    result *= 0.5

    # Handle orthonormal normalization
    if norm == 'ortho':
        scale = torch.ones(n, device=x.device, dtype=x.dtype)
        scale[0] = torch.sqrt(torch.tensor(1.0 / n, device=x.device, dtype=x.dtype))
        scale[1:] = torch.sqrt(torch.tensor(2.0 / n, device=x.device, dtype=x.dtype))
        scale_shape = [1] * ndim
        scale_shape[dim] = n
        scale = scale.view(scale_shape)
        result = result * scale

    return result


def _idct_1d(x: torch.Tensor, n: int = None, dim: int = -1, norm: str = 'ortho') -> torch.Tensor:
    """
    1D IDCT-II (Inverse DCT, equivalent to DCT-III for orthonormal DCT-II).

    Direct computation using the inverse definition:
        x[j] = sum_{k=0}^{N-1} c[k] * X[k] * cos(pi*(2j+1)*k/(2N))
    where c[0] = sqrt(1/N), c[k] = sqrt(2/N) for k >= 1 (orthonormal).

    This is the exact inverse of _dct_1d with orthonormal normalization.

    Args:
        x: Input DCT coefficients
        n: Transform length (defaults to input size along dim)
        dim: Dimension along which to compute IDCT
        norm: Normalization mode ('ortho' for orthonormal, or None)

    Returns:
        Reconstructed signal along specified dimension
    """
    if n is None:
        n = x.shape[dim]

    # Ensure float32, restore dtype at end
    input_dtype = x.dtype
    x = x.float()

    # Move dim to last position
    if dim != x.dim() - 1 and dim != -1:
        x = x.transpose(dim, x.dim() - 1)

    # For orthonormal DCT-II, the inverse uses the same cosine basis but
    # with normalization applied first, then the transpose basis.
    # x[j] = sum_k c[k] * X[k] * cos(pi*(2j+1)*k/(2N))
    # This is: cos_basis^T @ (scale * X) where cos_basis has rows=j, cols=k

    # Build cosine basis: same as forward (transpose-symmetric structure)
    j = torch.arange(n, device=x.device, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    k = torch.arange(n, device=x.device, dtype=torch.float32).unsqueeze(0)  # (1, N)
    cosine_basis = torch.cos(torch.pi * (2 * j + 1) * k / (2 * n))

    # Apply normalization first
    if norm == 'ortho':
        scale = torch.ones(n, device=x.device, dtype=torch.float32)
        scale[0] = torch.sqrt(torch.tensor(1.0 / n, dtype=torch.float32))
        scale[1:] = torch.sqrt(torch.tensor(2.0 / n, dtype=torch.float32))
        x_scaled = x * scale
    else:
        x_scaled = x

    # Result: (*batch, N) @ (N, N)^T = (*batch, N)
    # Since cosine_basis[j,k] is symmetric in structure, we transpose it
    result = torch.matmul(x_scaled, cosine_basis.transpose(-1, -2))

    # Restore original dimension order and dtype
    if dim != x.dim() - 1 and dim != -1:
        result = result.transpose(dim, result.dim() - 1)
    return result.to(input_dtype)


def _create_spherical_frequency_masks(
    fft_t: int,
    fft_h: int,
    fft_w: int,
    low_cutoff: float = 0.15,
    high_cutoff: float = 0.5,
    device=None,
    temporal_scale: float = 0.3,  # Scale temporal frequency (lower = less temporal penalty)
):
    """
    Create 3D frequency masks for spatio-temporal FFT with separate temporal scaling.

    Uses proper frequency coordinates accounting for FFT wraparound:
    - Full FFT dims (t, h): frequencies wrap around; index 0 is DC, then positive,
      then negative (high freq). Use |fftfreq| for radial distance.
    - Real FFT dim (w): frequencies are [0, ..., 0.5]; no wraparound.
    
    Key improvement over pure spherical: temporal_scale down-weights temporal
    frequency in the radius calculation. This prevents moving objects (high temporal
    freq, low spatial freq) from being incorrectly classified as "high frequency"
    and penalized.

    Args:
        fft_t: Temporal FFT size (T for full FFT)
        fft_h: Height FFT size
        fft_w: Width FFT size (W//2+1 for rfft on last dim)
        low_cutoff: Inner radius (0-1) for low-frequency band
        high_cutoff: Outer radius (0-1) for mid-frequency band
        device: Device for tensors
        temporal_scale: Weight for temporal frequency (0-1). 
                       1.0 = pure spherical (all dims equal)
                       0.3 = temporal down-weighted (recommended for video)
                       0.0 = spatial-only (ignores temporal frequency)

    Returns:
        low_mask, mid_mask, high_mask: each (fft_t, fft_h, fft_w)
    """
    # Full FFT dims: use absolute frequency (DC at 0, high freq at ±0.5)
    t_freqs = torch.abs(torch.fft.fftfreq(fft_t, device=device))
    h_freqs = torch.abs(torch.fft.fftfreq(fft_h, device=device))
    # Real FFT dim: frequencies are [0, ..., 0.5]; use rfftfreq for correct scaling.
    # BUG FIX B: previously used arange(fft_w)/(fft_w-1) which mapped to [0, 1.0],
    # but rfftn's last dim maps to [0, 0.5]. This caused 2× elongation of spherical
    # masks along the W axis. We reconstruct W from fft_w = W//2+1.
    W = (fft_w - 1) * 2 if fft_w > 1 else 2
    w_freqs = torch.fft.rfftfreq(W, device=device)

    # Meshgrid: (fft_t, fft_h, fft_w)
    t_grid, h_grid, w_grid = torch.meshgrid(t_freqs, h_freqs, w_freqs, indexing='ij')

    # Scaled radial frequency: temporal_scale reduces temporal frequency contribution
    # This prevents motion (high temporal freq) from being classified as "high frequency"
    scaled_t = t_grid * temporal_scale
    radius = torch.sqrt(scaled_t**2 + h_grid**2 + w_grid**2)
    max_radius = radius.max().clamp(min=1e-8)
    radius_norm = radius / max_radius

    low_mask = (radius_norm <= low_cutoff).float()
    mid_mask = ((radius_norm > low_cutoff) & (radius_norm <= high_cutoff)).float()
    high_mask = (radius_norm > high_cutoff).float()

    return low_mask, mid_mask, high_mask


def _create_dct_frequency_masks(
    dct_t: int,
    dct_h: int,
    dct_w: int,
    low_cutoff: float = 0.15,
    high_cutoff: float = 0.5,
    device=None,
    temporal_scale: float = 0.3,  # Scale temporal frequency (lower = less temporal penalty)
):
    """
    Create 3D frequency masks for spatio-temporal DCT with separate temporal scaling.

    Per SSVAE paper: "we adopt a 3D DCT to analyze the spatio-temporal
    frequency spectrum." DCT frequencies are monotonically increasing
    from DC (0,0,0) to Nyquist, with no wraparound.
    
    Key improvement: temporal_scale down-weights temporal frequency to prevent
    moving objects from being incorrectly penalized.

    Args:
        dct_t: Temporal DCT size (T)
        dct_h: Height DCT size (H)
        dct_w: Width DCT size (W)
        low_cutoff: Inner radius (0-1) for low-frequency band
        high_cutoff: Outer radius (0-1) for mid-frequency band
        device: Device for tensors
        temporal_scale: Weight for temporal frequency (0-1). 
                       1.0 = pure spherical (all dims equal)
                       0.3 = temporal down-weighted (recommended for video)
                       0.0 = spatial-only (ignores temporal frequency)

    Returns:
        low_mask, mid_mask, high_mask: each (dct_t, dct_h, dct_w)
    """
    # DCT frequencies: indices 0..N-1 map to 0..(N-1)/(N-1) = 0..1
    # No wraparound; index 0 is DC, highest index is highest freq
    t_freqs = torch.arange(dct_t, device=device) / max(dct_t - 1, 1)
    h_freqs = torch.arange(dct_h, device=device) / max(dct_h - 1, 1)
    w_freqs = torch.arange(dct_w, device=device) / max(dct_w - 1, 1)

    # Meshgrid: (dct_t, dct_h, dct_w)
    t_grid, h_grid, w_grid = torch.meshgrid(t_freqs, h_freqs, w_freqs, indexing='ij')

    # Scaled radial frequency: temporal_scale reduces temporal frequency contribution
    scaled_t = t_grid * temporal_scale
    radius = torch.sqrt(scaled_t**2 + h_grid**2 + w_grid**2)
    max_radius = radius.max().clamp(min=1e-8)
    radius_norm = radius / max_radius

    low_mask = (radius_norm <= low_cutoff).float()
    mid_mask = ((radius_norm > low_cutoff) & (radius_norm <= high_cutoff)).float()
    high_mask = (radius_norm > high_cutoff).float()

    return low_mask, mid_mask, high_mask


def _spectral_loss_3d_video(
    pred_latents: torch.Tensor,
    latents: torch.Tensor,
    low_weight: float,
    mid_weight: float,
    high_weight: float,
    low_cutoff: float,
    high_cutoff: float,
    use_phase: bool,
    temporal_scale: float = 0.3,
) -> torch.Tensor:
    """
    3D spatio-temporal spectral loss for video (B, C, T, H, W).

    Applies 3D FFT over (T, H, W) to analyze frequency content across
    all three dimensions simultaneously. This is the correct approach
    per SSVAE paper, which notes that 2D-only methods "do not
    adequately address the temporal dimension in video latents."

    Benefits:
    - Penalizes temporal flickering (high temporal frequencies)
    - Enforces low-frequency bias in motion (smooth trajectories)
    - Captures spatio-temporal correlations that 2D FFT misses

    Args:
        pred_latents: (B, C, T, H, W) predicted clean latents
        latents: (B, C, T, H, W) ground truth clean latents
        low_weight, mid_weight, high_weight: per-band loss weights
        low_cutoff, high_cutoff: spherical frequency band boundaries
        use_phase: include phase in loss

    Returns:
        loss_spatial: (B, C, T, H, W) loss tensor
    """
    B, C, T, H, W = latents.shape

    # Permute to (B*C, T, H, W) for batched 3D FFT
    pred = pred_latents.reshape(B * C, T, H, W)
    target = latents.reshape(B * C, T, H, W)

    # 3D FFT over (T, H, W)
    # rfftn with last two dims real -> output shape: (B*C, T, H, W//2+1)
    pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm='ortho')
    target_fft = torch.fft.rfftn(target, dim=(-3, -2, -1), norm='ortho')

    fft_t, fft_h, fft_w = pred_fft.shape[-3], pred_fft.shape[-2], pred_fft.shape[-1]

    # Create 3D masks with temporal scaling
    low_mask, mid_mask, high_mask = _create_spherical_frequency_masks(
        fft_t, fft_h, fft_w,
        low_cutoff, high_cutoff,
        device=pred_fft.device,
        temporal_scale=temporal_scale
    )

    # Expand masks: (1, fft_t, fft_h, fft_w)
    mask_exp_shape = (1, fft_t, fft_h, fft_w)

    band_losses = []
    masks = [low_mask, mid_mask, high_mask]
    weights = [low_weight, mid_weight, high_weight]

    for mask, weight in zip(masks, weights):
        mask_exp = mask.view(mask_exp_shape).to(pred_fft.device)
        pred_band = pred_fft * mask_exp
        target_band = target_fft * mask_exp

        if use_phase:
            band_loss = (
                F.mse_loss(pred_band.real, target_band.real, reduction='none') +
                F.mse_loss(pred_band.imag, target_band.imag, reduction='none')
            )
        else:
            band_loss = F.mse_loss(pred_band.abs(), target_band.abs(), reduction='none')

        band_losses.append(band_loss * weight)

    # Sum band losses: (B*C, T, H, W//2+1)
    total_fft_loss = torch.stack(band_losses, dim=1).sum(dim=1)

    # Convert to complex and inverse FFT back to spatio-temporal domain
    loss_fft_complex = torch.view_as_complex(
        torch.stack([total_fft_loss, torch.zeros_like(total_fft_loss)], dim=-1)
    )

    # irfftn with s to restore original shape
    loss_spatial = torch.fft.irfftn(
        loss_fft_complex, s=(T, H, W), dim=(-3, -2, -1), norm='ortho'
    ).abs()  # (B*C, T, H, W)

    # Reshape back to (B, C, T, H, W)
    loss_spatial = loss_spatial.reshape(B, C, T, H, W)

    return loss_spatial


def _spectral_loss_3d_video_dct(
    pred_latents: torch.Tensor,
    latents: torch.Tensor,
    low_weight: float,
    mid_weight: float,
    high_weight: float,
    low_cutoff: float,
    high_cutoff: float,
    use_phase: bool,
    temporal_scale: float = 0.3,
) -> torch.Tensor:
    """
    3D spatio-temporal spectral loss for video using DCT (SSVAE-compliant).

    Applies 3D DCT over (T, H, W) as specified in the SSVAE paper:
    "we adopt a 3D DCT to analyze the spatio-temporal frequency spectrum."

    DCT advantages over FFT:
    - Real-valued coefficients (no complex arithmetic overhead)
    - Mirror boundary conditions (no wraparound artifacts)
    - Matches SSVAE paper methodology exactly

    Args:
        pred_latents: (B, C, T, H, W) predicted clean latents
        latents: (B, C, T, H, W) ground truth clean latents
        low_weight, mid_weight, high_weight: per-band loss weights
        low_cutoff, high_cutoff: spherical frequency band boundaries
        use_phase: ignored for DCT (real-valued only)

    Returns:
        loss_spatial: (B, C, T, H, W) loss tensor
    """
    B, C, T, H, W = latents.shape

    # Reshape to (B*C, T, H, W) for batched 3D DCT
    pred = pred_latents.reshape(B * C, T, H, W)
    target = latents.reshape(B * C, T, H, W)

    # 3D DCT over (T, H, W) using 1D DCT composition
    # DCT-II with orthonormal normalization
    pred_dct = _dct_1d(_dct_1d(_dct_1d(pred, n=T, dim=-3, norm='ortho'),
                               n=H, dim=-2, norm='ortho'),
                       n=W, dim=-1, norm='ortho')
    target_dct = _dct_1d(_dct_1d(_dct_1d(target, n=T, dim=-3, norm='ortho'),
                                 n=H, dim=-2, norm='ortho'),
                         n=W, dim=-1, norm='ortho')

    # Create 3D masks with temporal scaling
    low_mask, mid_mask, high_mask = _create_dct_frequency_masks(
        T, H, W,
        low_cutoff, high_cutoff,
        device=pred_dct.device,
        temporal_scale=temporal_scale
    )

    # Expand masks: (1, T, H, W)
    mask_exp_shape = (1, T, H, W)

    band_losses = []
    masks = [low_mask, mid_mask, high_mask]
    weights = [low_weight, mid_weight, high_weight]

    for mask, weight in zip(masks, weights):
        mask_exp = mask.view(mask_exp_shape).to(pred_dct.device)
        pred_band = pred_dct * mask_exp
        target_band = target_dct * mask_exp

        # DCT is real-valued; no phase component
        band_loss = F.mse_loss(pred_band, target_band, reduction='none')
        band_losses.append(band_loss * weight)

    # Sum band losses: (B*C, T, H, W) - still in DCT frequency domain
    total_dct_loss = torch.stack(band_losses, dim=1).sum(dim=1)

    # Inverse DCT to convert loss from frequency domain back to spatio-temporal domain.
    # This ensures that applying a per-frame mask (e.g., I2V conditioning mask that
    # zeros out the first temporal position) correctly masks temporal/spatial positions
    # rather than incorrectly zeroing out a specific frequency component (temporal DC).
    # Matches the FFT path's behavior of applying irfftn before returning.
    loss_spatial = _idct_1d(_idct_1d(_idct_1d(total_dct_loss, n=T, dim=-3, norm='ortho'),
                                     n=H, dim=-2, norm='ortho'),
                            n=W, dim=-1, norm='ortho')

    # Reshape back to (B, C, T, H, W) and take absolute value, matching the FFT path.
    # The IDCT can produce negative values for loss computed in frequency domain;
    # abs() ensures non-negative loss and correct semantic for per-pixel masks.
    loss_spatial = loss_spatial.reshape(B, C, T, H, W).abs()

    return loss_spatial


def spectral_flow_loss(
    model_pred,
    latents,
    noise,
    batch_flow=None,
    timesteps=None,
    flow_loss_module=None,
    vae_temporal_stride=4,
    vae_spatial_stride=8,
    # Spectral params
    low_weight=1.0,
    mid_weight=1.0,
    high_weight=2.0,
    low_cutoff=0.15,
    high_cutoff=0.5,
    use_phase=True,
    lcr_weight=0.0,
    spectral_transform='dct',  # 'dct' (default, SSVAE-compliant) or 'fft'
    prediction_target='velocity',  # 'velocity' or 'x0'
    temporal_scale=0.3,  # Scale temporal frequency for video (0.0-1.0)
    spectral_weight=1.0,  # Overall spectral component weight (scales entire spectral loss)
    # Flow params
    flow_weight=0.1,
    flow_max_timestep=800,
    motion_weighted=True,
    reverse_gate=False,  # if True, flow loss weighted higher at high-noise timesteps
    adaptive=False,
    current_flow_weight=None,
):
    """
    Combined spectral + optical flow loss for video diffusion training.

    Spectral loss handles spatial frequency distribution (structure vs texture).
    Flow loss handles temporal motion consistency via latent-space flow warping.

    Args:
        spectral_transform: 'dct' for DCT-based (SSVAE-compliant, default), 'fft' for FFT-based
        prediction_target: 'velocity' (model predicts ε - x₀) or 'x0' (model predicts x₀ directly)
        flow_loss_module: Pre-cached FlowConsistencyLoss from SDTrainer.
                          If None, creates a new one (fallback for testing).
        reverse_gate: if True, flow loss is weighted higher at high-noise timesteps
                     (useful for enforcing motion consistency even in high-noise regime)
        current_flow_weight: Adaptive weight override. None = use flow_weight.

    Returns:
        tuple: (total_loss, flow_deviation, spatial_loss_val, flow_loss_val,
                spectral_component, flow_component)
        - total_loss: combined loss tensor (per-pixel)
        - flow_deviation: scalar raw flow loss for rejection/adaptive logic
        - spatial_loss_val: scalar spectral loss for logging
        - flow_loss_val: scalar weighted flow loss (with flow_weight applied) for logging
        - spectral_component: spectral loss tensor (for gradient-selective rejection)
        - flow_component: flow loss tensor (for gradient-selective rejection)
    """
    # Convert to float32 for precision
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()

    # Reconstruct predicted clean latents based on prediction target:
    # - velocity: model_pred = ε - x₀ ⇒ x₀ = noise - model_pred
    # - x0: model_pred = x₀ directly
    if prediction_target == 'x0':
        pred_latents = model_pred
    else:
        pred_latents = noise - model_pred

    is_video = len(latents.shape) == 5

    # === SPECTRAL LOSS ===
    #
    # NOTE: The original implementation used 2D FFT (spatial only) on each frame
    # independently. This was a quick prototype that treated video as a stack of
    # images. The SSVAE paper explicitly states this is incorrect:
    #
    #   "we adopt a 3D DCT to analyze the spatio-temporal frequency spectrum... they
    #    do not adequately address the temporal dimension in video latents."
    #
    # Problems with 2D-only FFT:
    # - Ignores temporal frequency components entirely
    # - Cannot penalize temporal flickering or inconsistent motion at the frequency level
    # - Leaves high-frequency temporal noise unchecked
    #
    # The 3D FFT below correctly operates over (T, H, W), enforcing low-frequency
    # bias in the temporal axis as well as spatial axes. This forces the model to
    # learn smoother motion transitions — a proven benefit from SSVAE's 3x
    # convergence speedup.
    #
    # The 2D implementation is preserved in comments below for reference.

    # Early exit for spectral component if all weights are zero
    if low_weight == 0 and mid_weight == 0 and high_weight == 0:
        loss_spatial = torch.zeros_like(latents)
    elif is_video:
        # Choose transform based on config
        if spectral_transform == 'dct':
            loss_spatial = _spectral_loss_3d_video_dct(
                pred_latents, latents,
                low_weight, mid_weight, high_weight,
                low_cutoff, high_cutoff,
                use_phase
            )
        else:
            # Default: FFT
            loss_spatial = _spectral_loss_3d_video(
                pred_latents, latents,
                low_weight, mid_weight, high_weight,
                low_cutoff, high_cutoff,
                use_phase
            )
    else:
        # Image case: 2D FFT is correct
        B, C, H, W = latents.shape
        pred_reshaped = pred_latents
        target_reshaped = latents

        pred_fft = torch.fft.rfft2(pred_reshaped, norm='ortho')
        target_fft = torch.fft.rfft2(target_reshaped, norm='ortho')

        fft_h, fft_w = pred_fft.shape[-2], pred_fft.shape[-1]
        low_mask, mid_mask, high_mask = _create_radial_frequency_masks(
            fft_h, fft_w, low_cutoff, high_cutoff
        )

        band_losses = []
        weights = [low_weight, mid_weight, high_weight]

        for mask, weight in zip([low_mask, mid_mask, high_mask], weights):
            mask_exp = mask.unsqueeze(0).unsqueeze(0).to(pred_fft.device)
            pred_band = pred_fft * mask_exp
            target_band = target_fft * mask_exp

            if use_phase:
                band_loss = F.mse_loss(pred_band.real, target_band.real, reduction='none') + \
                           F.mse_loss(pred_band.imag, target_band.imag, reduction='none')
            else:
                band_loss = F.mse_loss(pred_band.abs(), target_band.abs(), reduction='none')

            band_losses.append(band_loss * weight)

        total_fft_loss = torch.stack(band_losses, dim=1).sum(dim=1)
        loss_fft_complex = torch.view_as_complex(
            torch.stack([total_fft_loss, torch.zeros_like(total_fft_loss)], dim=-1)
        )
        loss_spatial = torch.fft.irfft2(loss_fft_complex, s=(H, W), norm='ortho').abs()

    # ============================================================
    # LEGACY 2D-ONLY IMPLEMENTATION (COMMENTED OUT)
    # ============================================================
    # This was the original approach: apply 2D FFT per frame.
    # Preserved for reference and rollback if needed.
    #
    # if is_video:
    #     B, C, T, H, W = latents.shape
    #     # Reshape to treat all frames as batch for FFT
    #     pred_reshaped = pred_latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    #     target_reshaped = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    # else:
    #     B, C, H, W = latents.shape
    #     pred_reshaped = pred_latents
    #     target_reshaped = latents
    #
    # # FFT on spatial dimensions only
    # pred_fft = torch.fft.rfft2(pred_reshaped, norm='ortho')
    # target_fft = torch.fft.rfft2(target_reshaped, norm='ortho')
    #
    # fft_h, fft_w = pred_fft.shape[-2], pred_fft.shape[-1]
    # low_mask, mid_mask, high_mask = _create_radial_frequency_masks(
    #     fft_h, fft_w, low_cutoff, high_cutoff
    # )
    #
    # band_losses = []
    # weights = [low_weight, mid_weight, high_weight]
    #
    # for mask, weight in zip([low_mask, mid_mask, high_mask], weights):
    #     mask_exp = mask.unsqueeze(0).unsqueeze(0).to(pred_fft.device)
    #     pred_band = pred_fft * mask_exp
    #     target_band = target_fft * mask_exp
    #
    #     if use_phase:
    #         band_loss = F.mse_loss(pred_band.real, target_band.real, reduction='none') + \
    #                    F.mse_loss(pred_band.imag, target_band.imag, reduction='none')
    #     else:
    #         band_loss = F.mse_loss(pred_band.abs(), target_band.abs(), reduction='none')
    #
    #     band_losses.append(band_loss * weight)
    #
    # total_fft_loss = torch.stack(band_losses, dim=1).sum(dim=1)
    # loss_fft_complex = torch.view_as_complex(
    #     torch.stack([total_fft_loss, torch.zeros_like(total_fft_loss)], dim=-1)
    # )
    # loss_spatial = torch.fft.irfft2(loss_fft_complex, s=(H, W), norm='ortho').abs()
    #
    # if is_video:
    #     loss_spatial = loss_spatial.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    # ============================================================

    # LCR loss
    if lcr_weight > 0.0:
        lcr_loss_scalar = _calculate_lcr_loss_ssvae_style(
            pred_latents,
            patch_size=2,
            alpha=0.75
        )
        spectral_mean = loss_spatial.mean()
        lcr_loss_scaled = lcr_loss_scalar * lcr_weight * spectral_mean
        lcr_loss_uniform = lcr_loss_scaled * torch.ones_like(loss_spatial)
        loss_spatial = loss_spatial + lcr_loss_uniform

    # === OPTICAL FLOW LOSS (temporal motion) ===
    flow_loss = torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)
    flow_deviation = 0.0

    if (is_video and batch_flow is not None and flow_weight > 0
            and timesteps is not None):
        B, C, T_lat, H_lat, W_lat = model_pred.shape

        if T_lat >= 2:
            from toolkit.optical_flow.flow_loss import FlowConsistencyLoss

            # Handle expand_timesteps (Wan22 5B): timesteps may be (B, seq_len)
            # instead of (B,). Use scalar timesteps for the timestep gate.
            if timesteps.dim() == 2:
                # Per-pixel timesteps — use first token's timestep as representative
                t = timesteps[:, 0].float()
            else:
                t = timesteps.float()

            # Compute gate for early-exit check (same logic as FlowConsistencyLoss.forward)
            if reverse_gate:
                gate = torch.clamp(t / flow_max_timestep, min=0.0, max=1.0)
            else:
                gate = torch.clamp(1.0 - (t / flow_max_timestep), min=0.0)

            if gate.sum() > 1e-6:
                # x0 reconstruction for flow loss:
                # FlowConsistencyLoss.forward() uses: pred_x0 = noisy_latents - sigma * noise_pred
                # This is correct when noisy_latents=x_t and noise_pred=velocity.
                #
                # For velocity prediction (default):
                #   x_t = (1-σ)·x0 + σ·ε, pass model_pred (velocity) directly
                #   pred_x0 = x_t - σ·v = x0 ✓
                #
                # For x0 prediction:
                #   model_pred = x0 directly; convert to velocity form for the flow module:
                #   v_equiv = (x_t - x0_pred) / σ
                #   pred_x0 = x_t - σ·v_equiv = x_t - (x_t - x0_pred) = x0_pred ✓
                sigma = (t / 1000.0).view(B, 1, 1, 1, 1).to(latents.dtype)
                x_t = (1.0 - sigma) * latents + sigma * noise  # (B, C, T_lat, H_lat, W_lat)

                if flow_loss_module is None:
                    flow_module = FlowConsistencyLoss(
                        vae_temporal_stride=vae_temporal_stride,
                        vae_spatial_stride=vae_spatial_stride
                    ).to(model_pred.device, dtype=model_pred.dtype)
                else:
                    # Only move if device/dtype differs (avoid wasteful .to() every step)
                    param = next(flow_loss_module.parameters(), None)
                    if param is None or param.device != model_pred.device or param.dtype != model_pred.dtype:
                        flow_module = flow_loss_module.to(model_pred.device, dtype=model_pred.dtype)
                    else:
                        flow_module = flow_loss_module

                # Handle prediction_target: convert x0 prediction to velocity form if needed
                if prediction_target == 'x0':
                    # model_pred is x0; convert to equivalent velocity
                    # v = (x_t - x0) / sigma
                    flow_noise_pred = (x_t - model_pred) / sigma.clamp(min=1e-8)
                else:
                    flow_noise_pred = model_pred

                flow_loss = flow_module(
                    noise_pred=flow_noise_pred,
                    noisy_latents=x_t,
                    timesteps=t,
                    batch_flow=batch_flow.float().to(model_pred.device),
                    max_timestep=flow_max_timestep,
                    motion_weighted=motion_weighted,
                    reverse_gate=reverse_gate
                )
                flow_deviation = flow_loss.item()

    # === COMBINE ===
    if adaptive:
        # When adaptive, use the dynamically-adjusted weight.
        # current_flow_weight defaults to None, falling back to flow_weight.
        effective_flow_weight = current_flow_weight if current_flow_weight is not None else flow_weight
    else:
        # When not adaptive, use the configured base weight
        effective_flow_weight = flow_weight

    # Keep spectral and flow components separate so rejection can zero only
    # the flow gradient without killing the spectral learning signal.
    spectral_component = loss_spatial * spectral_weight
    flow_component = flow_loss * effective_flow_weight
    total_loss = spectral_component + flow_component

    # Return in original dtype for mixed precision compatibility
    original_dtype = latents.dtype
    spectral_component = spectral_component.to(original_dtype)
    flow_component = flow_component.to(original_dtype)
    total_loss = total_loss.to(original_dtype)

    return (total_loss, flow_deviation, loss_spatial.mean().item() * spectral_weight,
            flow_loss.item() * effective_flow_weight,
            spectral_component, flow_component)


def mse_spectral_flow_loss(
    model_pred,
    latents,
    noise,
    batch_flow=None,
    timesteps=None,
    flow_loss_module=None,
    vae_temporal_stride=4,
    vae_spatial_stride=8,
    # MSE params
    mse_weight=1.0,
    # Spectral params
    low_weight=1.0,
    mid_weight=1.0,
    high_weight=2.0,
    low_cutoff=0.15,
    high_cutoff=0.5,
    use_phase=True,
    lcr_weight=0.0,
    spectral_transform='dct',  # 'dct' (default, SSVAE-compliant) or 'fft'
    prediction_target='velocity',  # 'velocity' or 'x0'
    temporal_scale=0.3,  # Scale temporal frequency for video (0.0-1.0)
    spectral_weight=1.0,  # Overall spectral component weight (scales entire spectral loss)
    # Flow params
    flow_weight=0.1,
    flow_max_timestep=800,
    motion_weighted=True,
    reverse_gate=False,  # if True, flow loss weighted higher at high-noise timesteps
    adaptive=False,
    current_flow_weight=None,
):
    """
    Combined MSE + spectral + optical flow loss for video diffusion training.

    MSE loss provides standard diffusion training signal.
    Spectral loss handles spatial frequency distribution (structure vs texture).
    Flow loss handles temporal motion consistency via latent-space flow warping.

    Args:
        mse_weight: Weight for the MSE component
        spectral_transform: 'dct' for DCT-based (SSVAE-compliant, default), 'fft' for FFT-based
        prediction_target: 'velocity' (model predicts ε - x₀) or 'x0' (model predicts x₀ directly)
        flow_loss_module: Pre-cached FlowConsistencyLoss from SDTrainer.
                          If None, creates a new one (fallback for testing).
        reverse_gate: if True, flow loss is weighted higher at high-noise timesteps
                     (useful for enforcing motion consistency even in high-noise regime)
        current_flow_weight: Adaptive weight override. None = use flow_weight.

    Returns:
        tuple: (total_loss, flow_deviation, mse_loss_val, spatial_loss_val, flow_loss_val,
                mse_component, spectral_component, flow_component)
        - total_loss: combined loss tensor (per-pixel)
        - flow_deviation: scalar raw flow loss for rejection/adaptive logic
        - mse_loss_val: scalar MSE loss for logging
        - spatial_loss_val: scalar spectral loss for logging
        - flow_loss_val: scalar weighted flow loss (with flow_weight applied) for logging
        - mse_component: MSE loss tensor (for gradient-selective rejection)
        - spectral_component: spectral loss tensor (for gradient-selective rejection)
        - flow_component: flow loss tensor (for gradient-selective rejection)
    """
    # Convert to float32 for precision
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()

    # Reconstruct predicted clean latents based on prediction target
    if prediction_target == 'x0':
        pred_latents = model_pred
    else:
        pred_latents = noise - model_pred

    is_video = len(latents.shape) == 5

    # === MSE LOSS ===
    # Standard MSE between predicted and target latents
    # Early exit if weight is zero to skip unnecessary computation
    if mse_weight == 0:
        mse_loss = torch.zeros_like(latents)
    else:
        mse_loss = F.mse_loss(pred_latents, latents, reduction='none')  # (B, C, T, H, W) or (B, C, H, W)

    # === SPECTRAL LOSS ===
    # Early exit for spectral component if all weights are zero
    if low_weight == 0 and mid_weight == 0 and high_weight == 0:
        loss_spatial = torch.zeros_like(latents)
    elif is_video:
        # Choose transform based on config
        if spectral_transform == 'dct':
            loss_spatial = _spectral_loss_3d_video_dct(
                pred_latents, latents,
                low_weight, mid_weight, high_weight,
                low_cutoff, high_cutoff,
                use_phase,
                temporal_scale
            )
        else:
            # Default: FFT
            loss_spatial = _spectral_loss_3d_video(
                pred_latents, latents,
                low_weight, mid_weight, high_weight,
                low_cutoff, high_cutoff,
                use_phase,
                temporal_scale
            )
    else:
        # Image case: 2D FFT is correct
        B, C, H, W = latents.shape
        pred_reshaped = pred_latents
        target_reshaped = latents

        pred_fft = torch.fft.rfft2(pred_reshaped, norm='ortho')
        target_fft = torch.fft.rfft2(target_reshaped, norm='ortho')

        fft_h, fft_w = pred_fft.shape[-2], pred_fft.shape[-1]
        low_mask, mid_mask, high_mask = _create_radial_frequency_masks(
            fft_h, fft_w, low_cutoff, high_cutoff
        )

        band_losses = []
        weights = [low_weight, mid_weight, high_weight]

        for mask, weight in zip([low_mask, mid_mask, high_mask], weights):
            mask_exp = mask.unsqueeze(0).unsqueeze(0).to(pred_fft.device)
            pred_band = pred_fft * mask_exp
            target_band = target_fft * mask_exp

            if use_phase:
                band_loss = F.mse_loss(pred_band.real, target_band.real, reduction='none') + \
                           F.mse_loss(pred_band.imag, target_band.imag, reduction='none')
            else:
                band_loss = F.mse_loss(pred_band.abs(), target_band.abs(), reduction='none')

            band_losses.append(band_loss * weight)

        total_fft_loss = torch.stack(band_losses, dim=1).sum(dim=1)
        loss_fft_complex = torch.view_as_complex(
            torch.stack([total_fft_loss, torch.zeros_like(total_fft_loss)], dim=-1)
        )
        loss_spatial = torch.fft.irfft2(loss_fft_complex, s=(H, W), norm='ortho').abs()

    # LCR loss
    if lcr_weight > 0.0:
        lcr_loss_scalar = _calculate_lcr_loss_ssvae_style(
            pred_latents,
            patch_size=2,
            alpha=0.75
        )
        spectral_mean = loss_spatial.mean()
        lcr_loss_scaled = lcr_loss_scalar * lcr_weight * spectral_mean
        lcr_loss_uniform = lcr_loss_scaled * torch.ones_like(loss_spatial)
        loss_spatial = loss_spatial + lcr_loss_uniform

    # === OPTICAL FLOW LOSS (temporal motion) ===
    flow_loss = torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)
    flow_deviation = 0.0

    if (is_video and batch_flow is not None and flow_weight > 0
            and timesteps is not None):
        B, C, T_lat, H_lat, W_lat = model_pred.shape

        if T_lat >= 2:
            from toolkit.optical_flow.flow_loss import FlowConsistencyLoss

            # Handle expand_timesteps (Wan22 5B): timesteps may be (B, seq_len)
            # instead of (B,). Use scalar timesteps for the timestep gate.
            if timesteps.dim() == 2:
                # Per-pixel timesteps — use first token's timestep as representative
                t = timesteps[:, 0].float()
            else:
                t = timesteps.float()

            # Compute gate for early-exit check (same logic as FlowConsistencyLoss.forward)
            if reverse_gate:
                gate = torch.clamp(t / flow_max_timestep, min=0.0, max=1.0)
            else:
                gate = torch.clamp(1.0 - (t / flow_max_timestep), min=0.0)

            if gate.sum() > 1e-6:
                # x0 reconstruction for flow loss
                sigma = (t / 1000.0).view(B, 1, 1, 1, 1).to(latents.dtype)
                x_t = (1.0 - sigma) * latents + sigma * noise  # (B, C, T_lat, H_lat, W_lat)

                if flow_loss_module is None:
                    flow_module = FlowConsistencyLoss(
                        vae_temporal_stride=vae_temporal_stride,
                        vae_spatial_stride=vae_spatial_stride
                    ).to(model_pred.device, dtype=model_pred.dtype)
                else:
                    # Only move if device/dtype differs
                    param = next(flow_loss_module.parameters(), None)
                    if param is None or param.device != model_pred.device or param.dtype != model_pred.dtype:
                        flow_module = flow_loss_module.to(model_pred.device, dtype=model_pred.dtype)
                    else:
                        flow_module = flow_loss_module

                # Handle prediction_target: convert x0 prediction to velocity form if needed
                if prediction_target == 'x0':
                    # model_pred is x0; convert to equivalent velocity
                    flow_noise_pred = (x_t - model_pred) / sigma.clamp(min=1e-8)
                else:
                    flow_noise_pred = model_pred

                flow_loss = flow_module(
                    noise_pred=flow_noise_pred,
                    noisy_latents=x_t,
                    timesteps=t,
                    batch_flow=batch_flow.float().to(model_pred.device),
                    max_timestep=flow_max_timestep,
                    motion_weighted=motion_weighted,
                    reverse_gate=reverse_gate
                )
                flow_deviation = flow_loss.item()

    # === COMBINE ===
    if adaptive:
        # When adaptive, use the dynamically-adjusted weight.
        # current_flow_weight defaults to None, falling back to flow_weight.
        effective_flow_weight = current_flow_weight if current_flow_weight is not None else flow_weight
    else:
        # When not adaptive, use the configured base weight
        effective_flow_weight = flow_weight

    # Keep all three components separate so PCGrad can project all three gradients
    mse_component = mse_loss * mse_weight
    spectral_component = loss_spatial * spectral_weight
    flow_component = flow_loss * effective_flow_weight
    total_loss = mse_component + spectral_component + flow_component

    # Return in original dtype for mixed precision compatibility
    original_dtype = latents.dtype
    mse_component = mse_component.to(original_dtype)
    spectral_component = spectral_component.to(original_dtype)
    flow_component = flow_component.to(original_dtype)
    total_loss = total_loss.to(original_dtype)

    return (total_loss, flow_deviation,
            mse_loss.mean().item() * mse_weight,
            loss_spatial.mean().item() * spectral_weight,
            flow_loss.item() * effective_flow_weight,
            mse_component, spectral_component, flow_component)
