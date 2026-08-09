import os
import itertools
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from toolkit.print import print_acc
from toolkit.metadata import get_meta_for_safetensors
from toolkit.optical_flow.base import get_flow_model_info, get_flow_model_names, default_flow_model_name


class OpticalFlowCachingMixin:
    """Mixin for AiToolkitDataset that handles optical flow caching."""

    def __init__(self, **kwargs):
        if hasattr(super(), "__init__"):
            super().__init__(**kwargs)
        self.is_caching_optical_flow = getattr(
            self.dataset_config, "cache_optical_flow_to_disk", False
        )

    def cache_optical_flow_all(self):
        """Cache optical flow for all files in the dataset."""
        if not self.is_caching_optical_flow:
            return

        # Skip if not video data
        if not self.is_video:
            print_acc("Skipping optical flow caching: not a video dataset")
            return

        flow_model = self.dataset_config.optical_flow_model

        # Validate model is registered
        model_info = get_flow_model_info(flow_model)
        if model_info is None:
            from toolkit.optical_flow import get_flow_model_names, default_flow_model_name

            available = ", ".join(get_flow_model_names())
            print_acc(
                f"WARNING: Unknown flow model '{flow_model}', using default '{default_flow_model_name()}'"
            )
            print_acc(f"Available models: {available}")
            flow_model = default_flow_model_name()
            model_info = get_flow_model_info(flow_model)

        # Create estimator via factory (auto-downloads)
        default_iters = model_info.default_iters if model_info else 12
        # Use num_workers from dataset_config (same pool used for latent caching)
        num_workers = max(1, self.dataset_config.num_workers)
        print_acc(
            f"Caching optical flow for {self.dataset_path} "
            f"(model={flow_model}, iters={default_iters}, "
            f"workers={num_workers})"
        )

        # Import create_flow_estimator here to avoid circular import
        from toolkit.optical_flow import create_flow_estimator

        estimator = create_flow_estimator(
            model_name=flow_model,
            device=self.sd.device,
            dtype=self.sd.torch_dtype,
        )

        def _prep(prep_item):
            """Check if flow is already cached, otherwise prepare for caching."""
            prep_flow_path = prep_item.get_flow_path(recalculate=True)
            if os.path.exists(prep_flow_path):
                return prep_item, prep_flow_path, False  # already cached

            # Load raw frames at the EXACT transform used for latents
            # This ensures flow is computed on the same pixels as latents
            prep_item.load_and_process_image(self.transform, only_load_latents=False, force_load_images=True)
            return prep_item, prep_flow_path, True

        pbar = tqdm(total=len(self.file_list), desc="Caching optical flow to disk")
        executor = ThreadPoolExecutor(max_workers=num_workers)

        try:
            pending = deque()
            file_iter = iter(self.file_list)

            # Queue initial items
            for queued in itertools.islice(
                file_iter, min(num_workers + 2, len(self.file_list))
            ):
                pending.append(executor.submit(_prep, queued))

            while pending:
                file_item, flow_path, needs_compute = pending.popleft().result()
                nxt = next(file_iter, None)
                if nxt is not None:
                    pending.append(executor.submit(_prep, nxt))

                if needs_compute:
                    self._cache_one_flow(file_item, flow_path, estimator, default_iters)

                file_item.is_flow_cached = True
                pbar.update(1)

        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            pbar.close()
            del estimator
            torch.cuda.empty_cache()

        print_acc(f"Optical flow caching complete for {self.dataset_path}")

    def _cache_one_flow(self, file_item, flow_path, estimator, default_iters):
        """Cache flow for a single file item."""
        frames = file_item.tensor

        # Handle different tensor shapes
        if frames.ndim == 5:
            frames = frames[0]  # (1, T, 3, H, W) -> (T, 3, H, W)
        if frames.ndim == 4 and frames.shape[0] == 1:
            frames = frames.squeeze(0)

        # Check we have multiple frames
        if frames.ndim != 4 or frames.shape[0] < 2:
            print_acc(
                f"WARNING: Skipping flow cache for {file_item.path} - not enough frames"
            )
            return

        T, C, H, W = frames.shape

        # SEA-RAFT needs H,W divisible by 64 (general requirement for most RAFT-based models)
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64

        if pad_h or pad_w:
            frames = torch.nn.functional.pad(
                frames, (0, pad_w, 0, pad_h), mode="replicate"
            )

        # Compute flow using the estimator's compute_pairwise_flow method
        flow = estimator.compute_pairwise_flow(
            frames.to(self.sd.device, self.sd.torch_dtype),
            iters=default_iters,
        )  # (T-1, 2, H_pad, W_pad) fp16

        # Crop back to bucket size
        if pad_h or pad_w:
            flow = flow[:, :, :H, :W]

        # FIX: Safetensors requires contiguous tensors. Slicing breaks contiguity.
        flow = flow.contiguous()

        # Save to disk
        state_dict = OrderedDict([("flow", flow.cpu())])
        meta = get_meta_for_safetensors(file_item.get_flow_info_dict())
        os.makedirs(os.path.dirname(flow_path), exist_ok=True)
        save_file(state_dict, flow_path, metadata=meta)

        # Cache in memory
        file_item._cached_flow = flow

        # Free pixel data if not needed
        if not file_item.dataset_config.load_image_when_caching_latents:
            file_item.tensor = None
