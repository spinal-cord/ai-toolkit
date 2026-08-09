"""
Optical flow module for ai-toolkit.

Provides caching and consistency loss for video diffusion training.
Extensible design for multiple flow models via registry pattern.

Usage:
    from toolkit.optical_flow import (
        create_flow_estimator,
        list_flow_models,
        FlowConsistencyLoss,
    )

    # Create an estimator (auto-downloads on first use)
    estimator = create_flow_estimator("sea-raft-m", device="cuda")

    # List available models
    models = list_flow_models()
"""

import torch

# Import base classes and registry
from toolkit.optical_flow.base import (
    BaseFlowEstimator,
    OpticalFlowModelInfo,
    register_flow_model,
    get_flow_model_info,
    list_flow_models,
    get_flow_model_names,
    default_flow_model_name,
)

# Import concrete implementations (their registration happens at import time)
from toolkit.optical_flow import sea_raft

# Import mixins (after sea_raft is imported so registry is populated)
from toolkit.optical_flow.flow_caching_mixin import OpticalFlowCachingMixin
from toolkit.optical_flow.flow_file_item_mixin import OpticalFlowFileItemDTOMixin

# Import loss module
from toolkit.optical_flow.flow_loss import FlowConsistencyLoss, load_flow_loss


def create_flow_estimator(
    model_name: str, device: str = "cuda", dtype=None, **kwargs
) -> BaseFlowEstimator:
    """
    Factory function to create a flow estimator by model name.

    Args:
        model_name: Registered model name (e.g., "sea-raft-m")
        device: Device to run inference on
        dtype: Data type for inference
        **kwargs: Additional args passed to the estimator constructor

    Returns:
        Initialized and loaded flow estimator

    Raises:
        ValueError: If model_name is not registered
    """
    from toolkit.optical_flow.sea_raft import SeaRaftFlowEstimator

    if dtype is None:
        dtype = torch.bfloat16

    info = get_flow_model_info(model_name)
    if info is None:
        available = ", ".join(get_flow_model_names())
        raise ValueError(
            f"Unknown optical flow model '{model_name}'. "
            f"Available models: {available}"
        )

    # Route to appropriate estimator based on model name
    if model_name.startswith("sea-raft-"):
        variant = "M" if model_name == "sea-raft-m" else "S"
        estimator = SeaRaftFlowEstimator(
            variant=variant, device=device, dtype=dtype, **kwargs
        )
    else:
        raise ValueError(
            f"No estimator implementation for model '{model_name}'. "
            f"Please add a new estimator class and register it."
        )

    # Download and load the model
    estimator.download_and_load()
    return estimator


# Legacy compatibility: FLOW_MODEL_REGISTRY still works but use registry functions instead
FLOW_MODEL_REGISTRY = {
    "sea-raft-m": {
        "repo": "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        "class": sea_raft.SeaRaftFlowEstimator,
        "variant": "M",
    },
    "sea-raft-s": {
        "repo": "MemorySlices/Tartan-C-T-TSKH-spring540x960-S",
        "class": sea_raft.SeaRaftFlowEstimator,
        "variant": "S",
    },
}
