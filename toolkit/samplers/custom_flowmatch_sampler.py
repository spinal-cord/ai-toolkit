import math
from typing import Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np
from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0
        self.timestep_type = "linear"

        # Store the UNSHIFTED sigma_min (i.e., 1/num_train_timesteps).
        # The parent __init__ stores the already-shifted sigma_min, which causes
        # a double-shift when set_timesteps applies the shift transformation again.
        # We use this to build the correct single-shift schedule.
        self._unshifted_sigma_min = 1.0 / self.config.num_train_timesteps

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000
            # Bell-Shaped Mean-Normalized Timestep Weighting
            # bsmntw? need a better name

            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

            # Shift minimum to 0
            y_shifted = y - y.min()

            # Scale to make mean 1
            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # only do half bell
            hbsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # flatten second half to max
            hbsmntw_weighing[num_timesteps //
                             2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            # Create linear timesteps from 1000 to 1
            timesteps = torch.linspace(1000, 1, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ):
        """
        Override set_timesteps to fix the double-shift bug.

        The parent class computes sigma_min from the full 1000-step shifted schedule
        in __init__, then in set_timesteps linearly spaces between sigma_max and
        sigma_min (both already shifted) and applies the shift transformation AGAIN.
        This results in a double-shifted sigma schedule.

        The correct approach (matching Wan2GP's EulerScheduler):
          1. Linearly space N values in the UNSHIFTED range [1/num_train_timesteps, 1.0]
          2. Apply the shift transformation ONCE
          3. timesteps = sigmas * num_train_timesteps
          4. Append terminal sigma=0

        This ensures the final step has a small, appropriate step size rather than
        a 5x+ oversized jump to zero.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length"
                )

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided."
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                # FIX: Use the UNSHIFTED range [1, num_train_timesteps] as the base.
                # The parent class used [sigma_min*1000, sigma_max*1000] where sigma_min
                # was already shifted, causing a double-shift when shift is applied below.
                timesteps = np.linspace(
                    self.config.num_train_timesteps,
                    1,
                    num_inference_steps,
                    dtype=np.float32,
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        # Apply the shift transformation ONCE (or dynamic shifting)
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # If required, stretch the sigmas schedule
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        # If required, convert to karras/exponential/beta schedules
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        # Convert to tensors
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        # Append the terminal sigma value (0 for normal, 1 for inverted)
        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False, timestep_type="linear") -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item()
                        for t in timesteps]

        # Get the weights for the timesteps
        if timestep_type == "weighted":
            weights = torch.tensor(
                [default_weighing_scheme[i] for i in step_indices],
                device=timesteps.device,
                dtype=timesteps.dtype
            )
        elif v2:
            weights = self.linear_timesteps_weights2[step_indices].flatten()
        else:
            weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        # forward ODE
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
        # reverse ODE
        # noisy_model_input = (1 - t_01) * noise + t_01 * original_samples
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type='linear',
        latents=None,
        patch_size=1
    ):
        self.timestep_type = timestep_type
        if timestep_type == 'linear' or timestep_type == 'weighted':
            timesteps = torch.linspace(1000, 1, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        elif timestep_type == 'sigmoid':
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = ((1 - t) * 1000)

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)

            return timesteps
        elif timestep_type in ['flux_shift', 'lumina2_shift', 'shift']:
            # FIX: Use the UNSHIFTED range [1, num_train_timesteps] as the base,
            # then apply shift ONCE. The previous code used self.sigma_min (already
            # shifted in __init__) as the lower bound, causing a double-shift.
            # This matches the inference set_timesteps behavior and Wan2GP's
            # EulerScheduler.
            timesteps = np.linspace(
                self.config.num_train_timesteps,
                1,
                num_timesteps,
                dtype=np.float32,
            )

            sigmas = timesteps / self.config.num_train_timesteps

            if self.config.use_dynamic_shifting:
                if latents is None:
                    raise ValueError('latents is None')

                # for flux we double up the patch size before sending her in to simulate the latent reduction
                h = latents.shape[2]
                w = latents.shape[3]
                image_seq_len = h * w // (patch_size**2)

                mu = calculate_shift(
                    image_seq_len,
                    self.config.get("base_image_seq_len", 256),
                    self.config.get("max_image_seq_len", 4096),
                    self.config.get("base_shift", 0.5),
                    self.config.get("max_shift", 1.16),
                )
                sigmas = self.time_shift(mu, 1.0, sigmas)
            else:
                sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

            if self.config.shift_terminal:
                sigmas = self.stretch_shift_to_terminal(sigmas)

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)

            sigmas = torch.from_numpy(sigmas).to(
                dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps

            if self.config.invert_sigmas:
                sigmas = 1.0 - sigmas
                timesteps = sigmas * self.config.num_train_timesteps
                sigmas = torch.cat(
                    [sigmas, torch.ones(1, device=sigmas.device)])
            else:
                sigmas = torch.cat(
                    [sigmas, torch.zeros(1, device=sigmas.device)])

            self.timesteps = timesteps.to(device=device)
            self.sigmas = sigmas

            self.timesteps = timesteps.to(device=device)
            return timesteps

        elif timestep_type == 'lognorm_blend':
            # disgtribute timestepd to the center/early and blend in linear
            alpha = 0.75

            lognormal = LogNormal(loc=0, scale=0.333)

            # Sample from the distribution
            t1 = lognormal.sample((int(num_timesteps * alpha),)).to(device)

            # Scale and reverse the values to go from 1000 to 0
            t1 = ((1 - t1/t1.max()) * 1000)

            # add half of linear
            t2 = torch.linspace(1000, 1, int(
                num_timesteps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device=device)
            return timesteps
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")
