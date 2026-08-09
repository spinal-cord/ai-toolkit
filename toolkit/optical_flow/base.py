import abc
import os
import torch
from typing import Optional, Dict, Any, List, Tuple
from huggingface_hub import snapshot_download
from toolkit.paths import TOOLKIT_ROOT
from toolkit.print import print_acc


class OpticalFlowModelInfo:
    """Metadata about an optical flow model for UI and config."""

    def __init__(
        self,
        name: str,
        display_name: str,
        repo_id: str,
        description: str = "",
        recommended_use: str = "",
        requires_memory_gb: float = 3.0,
        default_iters: int = 12,
        supported_resolutions: List[Tuple[int, int]] = None,
    ):
        self.name = name
        self.display_name = display_name
        self.repo_id = repo_id
        self.description = description
        self.recommended_use = recommended_use
        self.requires_memory_gb = requires_memory_gb
        self.default_iters = default_iters
        self.supported_resolutions = supported_resolutions or [(480, 832)]


class BaseFlowEstimator(abc.ABC):
    """
    Abstract base class for optical flow estimators.
    New flow models should subclass this and implement the abstract methods.
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype

    @abc.abstractmethod
    def download_and_load(self):
        """Download model weights and initialize the model."""
        pass

    @abc.abstractmethod
    def compute_pairwise_flow(
        self, frames: torch.Tensor, iters: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute pairwise optical flow for consecutive frames.

        Args:
            frames: (T, 3, H, W) float in [-1, 1] on self.device

        Returns:
            flow: (T-1, 2, H, W) float16 on CPU, in pixel units
        """
        pass

    def _download_model(self, repo_id: str, local_dir_name: str) -> str:
        """Helper to download a model from HuggingFace with caching."""
        local_dir = os.path.join(TOOLKIT_ROOT, "models", "optical_flow", local_dir_name)
        print_acc(f"Downloading optical flow model from HuggingFace ({repo_id})...")
        model_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print_acc(f"Model loaded from {model_dir}")
        return model_dir


# Global registry for optical flow models
_FLOW_MODEL_REGISTRY: Dict[str, OpticalFlowModelInfo] = {}


def register_flow_model(info: OpticalFlowModelInfo):
    """Register an optical flow model."""
    _FLOW_MODEL_REGISTRY[info.name] = info


def get_flow_model_info(name: str) -> Optional[OpticalFlowModelInfo]:
    """Get model info by name."""
    return _FLOW_MODEL_REGISTRY.get(name)


def list_flow_models() -> List[OpticalFlowModelInfo]:
    """List all registered flow models."""
    return list(_FLOW_MODEL_REGISTRY.values())


def get_flow_model_names() -> List[str]:
    """Get list of all registered flow model names."""
    return list(_FLOW_MODEL_REGISTRY.keys())


def default_flow_model_name() -> str:
    """Get the default flow model name (first registered)."""
    names = get_flow_model_names()
    return names[0] if names else "sea-raft-m"
