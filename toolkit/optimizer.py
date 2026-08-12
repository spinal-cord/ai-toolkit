import torch
import inspect
from typing import Dict, Any


def _filter_optimizer_params(
    optimizer_class,
    params,
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Filter optimizer kwargs to only include recognized parameters.
    Prevents crashes when users pass wrong parameters (e.g., 'weight_decay' to an optimizer that doesn't accept it).
    """
    try:
        sig = inspect.signature(optimizer_class.__init__)
        valid_params = set(sig.parameters.keys())
        # Remove 'self' from valid params
        valid_params.discard('self')
        # Keep only params that the optimizer actually accepts
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        if len(filtered) < len(kwargs):
            ignored = set(kwargs.keys()) - set(filtered.keys())
            print(f"[INFO] Ignoring unknown optimizer parameters: {ignored}")
        return filtered
    except (ValueError, TypeError):
        # If we can't inspect the signature, return all params and let the optimizer handle it
        return kwargs


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None
):
    if optimizer_params is None:
        optimizer_params = {}
    lower_type = optimizer_type.lower()
    if lower_type.startswith("dadaptation"):
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0
        if lower_type.endswith('lion'):
            filtered_params = _filter_optimizer_params(dadaptation.DAdaptLion, params, optimizer_params)
            optimizer = dadaptation.DAdaptLion(params, eps=1e-8, lr=use_lr, **filtered_params)
        elif lower_type.endswith('adam'):
            filtered_params = _filter_optimizer_params(dadaptation.DAdaptLion, params, optimizer_params)
            optimizer = dadaptation.DAdaptLion(params, eps=1e-8, lr=use_lr, **filtered_params)
        elif lower_type == 'dadaptation':
            # backwards compatibility
            filtered_params = _filter_optimizer_params(dadaptation.DAdaptAdam, params, optimizer_params)
            optimizer = dadaptation.DAdaptAdam(params, eps=1e-8, lr=use_lr, **filtered_params)
            # warn user that dadaptation is deprecated
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        filtered_params = _filter_optimizer_params(Prodigy8bit, params, optimizer_params)
        optimizer = Prodigy8bit(params, lr=use_lr, eps=1e-8, **filtered_params)
    elif lower_type.startswith("adamw_fp8"):
        from toolkit.optimizers.adamw_fp8 import AdamWFP8
        print("Using adamw_fp8")
        use_lr = learning_rate
        
        filtered_params = _filter_optimizer_params(AdamWFP8, params, optimizer_params)
        optimizer = AdamWFP8(params, lr=use_lr, eps=1e-8, **filtered_params)
    elif lower_type.startswith("adamw_bf16"):
        from toolkit.optimizers.adamw_bf16 import AdamWBF16
        print("Using adamw_bf16")
        use_lr = learning_rate
        
        filtered_params = _filter_optimizer_params(AdamWBF16, params, optimizer_params)
        optimizer = AdamWBF16(params, lr=use_lr, eps=1e-8, **filtered_params)
    elif lower_type.startswith("prodigy"):
        from prodigyopt import Prodigy

        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        safe_params = {k: v for k, v in optimizer_params.items() if k not in ("use_bias_correction", "d0", "d_coef", "safeguard_warmup")}
        filtered_params = _filter_optimizer_params(Prodigy, params, safe_params)
        optimizer = Prodigy(params, lr=use_lr, eps=1e-8, use_bias_correction=True, d0=5e-5, d_coef=1.0, safeguard_warmup=True, **filtered_params)
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit
        filtered_params = _filter_optimizer_params(Adam8bit, params, optimizer_params)
        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-8, **filtered_params)
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit
        filtered_params = _filter_optimizer_params(Adam8bit, params, optimizer_params)
        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-8, decouple=True, **filtered_params)
    elif lower_type.endswith("8bit"):
        import bitsandbytes

        if lower_type == "adam8bit":
            filtered_params = _filter_optimizer_params(bitsandbytes.optim.Adam8bit, params, optimizer_params)
            return bitsandbytes.optim.Adam8bit(params, lr=learning_rate, eps=1e-8, **filtered_params)
        if lower_type == "ademamix8bit":
            filtered_params = _filter_optimizer_params(bitsandbytes.optim.AdEMAMix8bit, params, optimizer_params)
            return bitsandbytes.optim.AdEMAMix8bit(params, lr=learning_rate, eps=1e-8, **filtered_params)
        elif lower_type == "adamw8bit":
            filtered_params = _filter_optimizer_params(bitsandbytes.optim.AdamW8bit, params, optimizer_params)
            return bitsandbytes.optim.AdamW8bit(params, lr=learning_rate, eps=1e-8, **filtered_params)
        elif lower_type == "lion8bit":
            filtered_params = _filter_optimizer_params(bitsandbytes.optim.Lion8bit, params, optimizer_params)
            return bitsandbytes.optim.Lion8bit(params, lr=learning_rate, **filtered_params)
        else:
            raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type == 'adam':
        filtered_params = _filter_optimizer_params(torch.optim.Adam, params, optimizer_params)
        optimizer = torch.optim.Adam(params, lr=torch.tensor(learning_rate), eps=1e-8, **filtered_params)
    elif lower_type == 'adamw':
        filtered_params = _filter_optimizer_params(torch.optim.AdamW, params, optimizer_params)
        optimizer = torch.optim.AdamW(params, lr=torch.tensor(learning_rate), eps=1e-8, **filtered_params)
    elif lower_type == 'adamw_fused':
        fused_params = dict(optimizer_params)
        fused_params.setdefault("fused", True)
        try:
            filtered_params = _filter_optimizer_params(torch.optim.AdamW, params, fused_params)
            optimizer = torch.optim.AdamW(params, lr=torch.tensor(learning_rate), eps=1e-8, **filtered_params)
        except TypeError:
            # Older torch builds do not support fused AdamW; fallback safely.
            fused_params.pop("fused", None)
            filtered_params = _filter_optimizer_params(torch.optim.AdamW, params, fused_params)
            optimizer = torch.optim.AdamW(params, lr=torch.tensor(learning_rate), eps=1e-8, **filtered_params)
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            filtered_params = _filter_optimizer_params(Lion, params, optimizer_params)
            return Lion(params, lr=learning_rate, **filtered_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        filtered_params = _filter_optimizer_params(torch.optim.Adagrad, params, optimizer_params)
        optimizer = torch.optim.Adagrad(params, lr=torch.tensor(learning_rate), **filtered_params)
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        if 'relative_step' not in optimizer_params:
            optimizer_params['relative_step'] = False
        if 'scale_parameter' not in optimizer_params:
            optimizer_params['scale_parameter'] = False
        if 'warmup_init' not in optimizer_params:
            optimizer_params['warmup_init'] = False
        filtered_params = _filter_optimizer_params(Adafactor, params, optimizer_params)
        optimizer = Adafactor(params, lr=torch.tensor(learning_rate), **filtered_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        filtered_params = _filter_optimizer_params(Automagic, params, optimizer_params)
        optimizer = Automagic(params, lr=torch.tensor(learning_rate), **filtered_params)
    elif lower_type == 'automagic2':
        from toolkit.optimizers.automagic2 import Automagic2
        filtered_params = _filter_optimizer_params(Automagic2, params, optimizer_params)
        optimizer = Automagic2(params, lr=torch.tensor(learning_rate), **filtered_params)
    elif lower_type == 'automagic3':
        from toolkit.optimizers.automagic3 import Automagic3
        # Detect if this is a multistage/MoE training by checking param
        # group names. When expert-aware mode is enabled, each layer
        # within each expert gets its own adaptive LR, which is important
        # for Wan 2.2 14B MoE models where different blocks (early vs.
        # late layers) often need different adaptation rates.
        is_moe = any(
            isinstance(p, dict) and p.get("name", "") in ("high_noise_loras", "low_noise_loras")
            for p in params
        ) if isinstance(params, list) else False
        # Filter params but keep expert_aware separate since it's computed dynamically
        auto_params = dict(optimizer_params)
        auto_params.pop('expert_aware', None)
        filtered_params = _filter_optimizer_params(Automagic3, params, auto_params)
        optimizer = Automagic3(
            params,
            lr=torch.tensor(learning_rate),
            expert_aware=is_moe,
            **filtered_params,
        )
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    return optimizer
