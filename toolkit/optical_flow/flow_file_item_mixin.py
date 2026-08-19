import os
import json
import base64
import hashlib
from collections import OrderedDict
from typing import Union
import torch
from safetensors.torch import load_file
from toolkit import dataset_crypto


class OpticalFlowFileItemDTOMixin:
    """Mixin for FileItemDTO that handles optical flow caching."""

    def __init__(self, *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self._cached_flow: Union[torch.Tensor, None] = None  # (T-1, 2, H, W) fp16
        self._flow_path: Union[str, None] = None
        self.is_flow_cached = False
        self.flow_version = 1  # bump to invalidate cache on format changes

    def get_flow_info_dict(self) -> 'OrderedDict':
        """
        Build hash input dict for flow cache key.
        Must include the SAME keys as get_latent_info_dict() so that
        crop/flip/num_frames/fps changes invalidate BOTH caches identically.
        """
        # Start with base info from latent caching
        item = OrderedDict([
            ("filename", os.path.basename(self.path)),
            ("scale_to_width", self.scale_to_width),
            ("scale_to_height", self.scale_to_height),
            ("crop_x", self.crop_x),
            ("crop_y", self.crop_y),
            ("crop_width", self.crop_width),
            ("crop_height", self.crop_height),
            ("flow_version", self.flow_version),
        ])

        # Include frame count and FPS if relevant
        if self.dataset_config.auto_frame_count:
            item["auto_frame_count"] = True
        elif self.dataset_config.num_frames > 1:
            item["num_frames"] = self.dataset_config.num_frames

        if self.dataset_config.fps != 24:
            item["fps"] = self.dataset_config.fps

        if self.dataset_config.do_i2v:
            item["do_i2v"] = True

        # Include flip flags (they affect flow direction)
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True

        # Include flow model info
        item["flow_model"] = self.dataset_config.optical_flow_model

        return item

    def get_flow_path(self, recalculate=False) -> str:
        """Get the cache path for this file's flow data."""
        if self._flow_path is not None and not recalculate:
            return self._flow_path

        img_dir = os.path.dirname(self.path)
        flow_dir = os.path.join(img_dir, '_flow_cache')
        hash_dict = self.get_flow_info_dict()

        filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
        hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
        hash_str = base64.urlsafe_b64encode(
            hashlib.md5(hash_input).digest()).decode('ascii').replace('=', '')

        self._flow_path = os.path.join(flow_dir, f'{filename_no_ext}_{hash_str}.safetensors')
        return self._flow_path

    def cleanup_flow(self):
        """Release the per-item flow tensor.

        Streaming: flow is always cached on disk and loaded on the fly
        (decrypted in RAM if the dataset is encrypted); the per-item tensor
        only exists while the item is inside the rotating prefetch ring, so
        always release it here.
        """
        self._cached_flow = None

    def get_flow(self, device=None) -> Union[torch.Tensor, None]:
        """Load and return cached flow tensor."""
        if not self.is_flow_cached:
            return None

        if self._cached_flow is None:
            state_dict = dataset_crypto.load_safetensors(self.get_flow_path(), device='cpu')
            self._cached_flow = state_dict['flow']  # (T-1, 2, H, W) fp16

        if device is not None:
            return self._cached_flow.to(device)
        return self._cached_flow
