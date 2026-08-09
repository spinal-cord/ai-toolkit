import os
import torch
import torch.nn.functional as F
from typing import Optional
from huggingface_hub import snapshot_download
from toolkit.paths import TOOLKIT_ROOT
from toolkit.print import print_acc
from toolkit.optical_flow.base import BaseFlowEstimator, register_flow_model, OpticalFlowModelInfo


class SeaRaftFlowEstimator(BaseFlowEstimator):
    """
    SEA-RAFT optical flow estimator.
    https://github.com/princeton-vl/SEA-RAFT

    Downloaded from HuggingFace: MemorySlices/Tartan-C-T-TSKH-spring540x960-{M/S}
    """

    def __init__(
        self,
        variant: str = "M",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Force float32 for higher precision flow computation and to avoid mixed-precision bugs
        # in F.grid_sample. SEA-RAFT is ~10-20M parameters, so float32 VRAM overhead is negligible.
        dtype = torch.float32
        super().__init__(device, dtype)
        self.variant = variant
        self.model = None
        self.model_dir = None

    def download_and_load(self):
        """Download SEA-RAFT weights and initialize the model."""
        repo = f"MemorySlices/Tartan-C-T-TSKH-spring540x960-{self.variant}"
        local_dir = os.path.join(
            TOOLKIT_ROOT, "models", "optical_flow", f"sea_raft_{self.variant.lower()}"
        )
        print_acc(
            f"Downloading SEA-RAFT-{self.variant} flow model from HuggingFace ({repo})..."
        )
        self.model_dir = snapshot_download(repo_id=repo, local_dir=local_dir)
        print_acc(f"SEA-RAFT-{self.variant} loaded from {self.model_dir}")

        # Load config and model (config.json is optional; model has sensible defaults)
        from toolkit.optical_flow.sea_raft_impl.config import load_cfg
        from toolkit.optical_flow.sea_raft_impl.model import SEA_RAFT

        config_path = os.path.join(self.model_dir, "config.json")
        cfg = load_cfg(config_path) if os.path.exists(config_path) else None
        self.model = SEA_RAFT(cfg).to(self.device, self.dtype).eval()

        # Load checkpoint
        from safetensors.torch import load_file

        ckpt_path = os.path.join(self.model_dir, "model.safetensors")
        state = load_file(ckpt_path, device="cpu")
        self.model.load_state_dict(state, strict=False)

        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute_pairwise_flow(
        self, frames: torch.Tensor, iters: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute pairwise optical flow for consecutive frames.

        Args:
            frames: (T, 3, H, W) float in [-1, 1] on self.device, H,W multiples of 64.

        Returns:
            flow: (T-1, 2, H, W) float16 on CPU, in pixel units (frame_t -> frame_t+1).
        """
        if iters is None:
            iters = 12

        flows = []
        for i in range(frames.shape[0] - 1):
            img1 = frames[i : i + 1].to(self.device, self.dtype)
            img2 = frames[i + 1 : i + 2].to(self.device, self.dtype)
            # Frames are in [-1, 1] from RescaleTransform.
            # SEA_RAFT.forward expects [0, 1] and normalizes to [-1, 1] internally
            # for neural net processing (standard practice). Convert before calling.
            img1 = (img1 + 1.0) / 2.0
            img2 = (img2 + 1.0) / 2.0
            # SEA-RAFT forward signature: (img1, img2, iters, test_mode=True)
            flow_pred, _ = self.model(img1, img2, iters=iters, test_mode=True)
            flows.append(flow_pred[0].detach().cpu())
        return torch.stack(flows).half()


# Register SEA-RAFT models in the global registry
# Note: This must happen after SeaRaftFlowEstimator is defined
def _register_sea_raft_models():
    register_flow_model(
        OpticalFlowModelInfo(
            name="sea-raft-m",
            display_name="SEA-RAFT-M (Medium)",
            repo_id="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
            description="SEA-RAFT Medium model. Best balance of speed and accuracy for video training.",
            recommended_use="Recommended for most use cases. ~3GB VRAM, good accuracy at 480p.",
            requires_memory_gb=3.0,
            default_iters=12,
        )
    )

    register_flow_model(
        OpticalFlowModelInfo(
            name="sea-raft-s",
            display_name="SEA-RAFT-S (Small)",
            repo_id="MemorySlices/Tartan-C-T-TSKH-spring540x960-S",
            description="SEA-RAFT Small model. Faster but slightly less accurate than -M.",
            recommended_use="Use when VRAM is limited. ~2GB VRAM.",
            requires_memory_gb=2.0,
            default_iters=12,
        )
    )


# Register on module import
_register_sea_raft_models()
