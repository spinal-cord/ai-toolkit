# Optical Flow Models for ai-toolkit

This document describes the extensible optical flow model system used for the `spectral_flow` loss type.

## Architecture

The optical flow system uses a registry pattern for easy extension:

```
toolkit/optical_flow/
├── __init__.py           # Package init, factory functions
├── base.py               # BaseFlowEstimator ABC, registry functions
├── sea_raft.py           # SEA-RAFT implementation (current only model)
├── sea_raft_impl/        # Vendored SEA-RAFT core
│   ├── __init__.py
│   ├── config.py
│   └── model.py
├── flow_caching_mixin.py # Dataset-level flow caching
├── flow_file_item_mixin.py # Per-file flow caching
└── flow_loss.py          # FlowConsistencyLoss for training
```

## How It Works

1. **Registry**: Each flow model registers itself via `register_flow_model()` with metadata
2. **Factory**: `create_flow_estimator(model_name)` creates and loads the appropriate estimator
3. **Caching**: Flow is precomputed and cached per-file during dataset setup
4. **Loss**: `FlowConsistencyLoss` uses cached flow for temporal consistency during training

## Adding a New Flow Model

To add support for a new optical flow model (e.g., GMFlow, Unimatch):

### 1. Create a new estimator class

Create a new file, e.g., `toolkit/optical_flow/gmflow.py`:

```python
import torch
from toolkit.optical_flow.base import BaseFlowEstimator, register_flow_model, OpticalFlowModelInfo

class GMFlowEstimator(BaseFlowEstimator):
    """GMFlow optical flow estimator."""

    def __init__(self, variant: str = "large", device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.variant = variant
        self.model = None

    def download_and_load(self):
        # Download model weights and initialize
        model_dir = self._download_model(
            repo_id="your-org/gmflow",
            local_dir_name=f"gmflow_{self.variant}"
        )
        # Load your model
        self.model = YourGMFlowModel().to(self.device, self.dtype).eval()
        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute_pairwise_flow(self, frames: torch.Tensor,
                               iters: int = None) -> torch.Tensor:
        """
        Compute pairwise optical flow for consecutive frames.

        Args:
            frames: (T, 3, H, W) float in [-1, 1] on self.device

        Returns:
            flow: (T-1, 2, H, W) float16 on CPU, in pixel units
        """
        flows = []
        for i in range(frames.shape[0] - 1):
            img1 = frames[i:i+1]
            img2 = frames[i+1:i+2]
            flow = self.model(img1, img2)  # Your model's forward pass
            flows.append(flow[0].detach().cpu())
        return torch.stack(flows).half()

# Register the model(s)
def _register_gmflow_models():
    register_flow_model(OpticalFlowModelInfo(
        name="gmflow-large",
        display_name="GMFlow-Large",
        repo_id="your-org/gmflow-large",
        description="GMFlow Large model. High accuracy.",
        recommended_use="Best accuracy for complex motion. ~4GB VRAM.",
        requires_memory_gb=4.0,
        default_iters=20,
    ))

_register_gmflow_models()
```

### 2. Import the new module

Add to `toolkit/optical_flow/__init__.py`:

```python
# Import concrete implementations
from toolkit.optical_flow import sea_raft
from toolkit.optical_flow import gmflow  # NEW
```

### 3. Update the factory function

Update `create_flow_estimator()` in `__init__.py`:

```python
def create_flow_estimator(model_name: str, ...):
    from toolkit.optical_flow.sea_raft import SeaRaftFlowEstimator
    from toolkit.optical_flow.gmflow import GMFlowEstimator  # NEW

    if model_name.startswith("sea-raft-"):
        variant = "M" if model_name == "sea-raft-m" else "S"
        estimator = SeaRaftFlowEstimator(...)
    elif model_name.startswith("gmflow-"):  # NEW
        variant = "large" if "large" in model_name else "small"
        estimator = GMFlowEstimator(variant=variant, ...)
    else:
        raise ValueError(...)

    estimator.download_and_load()
    return estimator
```

### 4. Update UI options

Add to the selector in `ui/src/app/jobs/new/SimpleJob.tsx`:

```tsx
options={[
    { value: 'sea-raft-m', label: 'SEA-RAFT-M (Medium, Recommended)' },
    { value: 'sea-raft-s', label: 'SEA-RAFT-S (Small, Faster)' },
    { value: 'gmflow-large', label: 'GMFlow-Large (Highest Accuracy)' },  // NEW
]}
```

## Current Available Models

| Model | Name | VRAM | Use Case |
|-------|------|------|----------|
| SEA-RAFT-M | `sea-raft-m` | ~3GB | Recommended default |
| SEA-RAFT-S | `sea-raft-s` | ~2GB | VRAM-constrained |

## Requirements for New Models

All flow estimators must:

1. Inherit from `BaseFlowEstimator`
2. Implement `download_and_load()` - download weights and initialize model
3. Implement `compute_pairwise_flow(frames)` - return `(T-1, 2, H, W)` float16 flow
4. Register via `register_flow_model()` with `OpticalFlowModelInfo`
5. Accept frames in `[-1, 1]` range (standard ai-toolkit normalization)
6. Return flow in pixel units (not normalized)

## Model Metadata Fields

`OpticalFlowModelInfo` fields:

- `name`: Unique identifier (used in config)
- `display_name`: Human-readable name (shown in UI)
- `repo_id`: HuggingFace repo ID for weights
- `description`: Brief description
- `recommended_use`: When to use this model
- `requires_memory_gb`: Estimated VRAM requirement
- `default_iters`: Default number of refinement iterations
- `supported_resolutions`: List of (height, width) tuples

## Testing a New Model

1. Test the estimator directly:
   ```python
   from toolkit.optical_flow import create_flow_estimator
   estimator = create_flow_estimator("your-model-name")
   ```

2. Test caching on a small dataset:
   ```yaml
   datasets:
     - folder_path: /small_test_dataset
       num_frames: 9
       cache_optical_flow_to_disk: true
       optical_flow_model: your-model-name
   ```

3. Test training with spectral_flow loss:
   ```yaml
   train:
     loss_type: spectral_flow
     spectral_flow_weight: 0.1
   ```
