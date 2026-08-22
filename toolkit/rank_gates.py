"""
SparseForge-inspired rank gate annealing for LoRA training.

This module implements soft, curvature-aware rank gating for LoRA adapters,
allowing gradual elimination of redundant ranks during training instead of
hard truncation. Based on SparseForge (2026) with adaptations for diffusion
training per wan_sparse_annealing.md.

Key concepts:
- Per-rank gates m_r ∈ [0,1] annealed from 1 → {0,1} (GatedLoRA), and
  per-element gates for full-finetune .diff tensors such as 1D layer norms
  (GatedDiff) — every tensor in the LoRA is covered
- PER-TENSOR annealing: at annealing start (after LR warmup) each tensor's
  final target k is computed from its own energy spectrum
  (compute_tensor_budgets) — the count of components whose contribution
  (S²/ΣS² for LoRA pairs, x²/Σx² for diff elements) is at least
  target_min_contribution (default 1e-4)
- Fisher diagonal via EMA of g² (cheaper than Hutchinson HVP)
- OBD-style scoring: s = |g·w| + 0.5·F ⊙ w² (curvature × magnitude²)
- Progressive quenching: soft gates → hard TopK → final binarization
- Binary preference penalty L_mid = Σ m(1-m)
- Dual-track: gates updated by their own rule, not autograd from task loss
"""

from typing import List, Optional, Dict, Tuple
from math import isfinite
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatedLoRA(nn.Module):
    """
    Wraps a LoRA module with per-rank soft gates.
    
    Forward: W_eff = B @ diag(m) @ A,  where m ∈ [0,1]^r are learnable gates.
    
    The gates are NOT optimized by the task loss autograd (that would trivially
    keep all gates at 1). Instead, they are updated by a dedicated rule based on
    curvature-aware importance scores (SparseForge Algorithm 1), called every
    T_update steps.
    
    At the end of training, gates are hardened to {0,1}, effectively selecting
    which rank components survive.
    
    IMPORTANT: Gate parameters (self.gates) MUST be excluded from the optimizer
    parameter groups. They have requires_grad=False so they are not in the autograd
    graph for task loss. Gate updates are applied directly to .data by the dedicated
    SparseForge rule. See prepare_optimizer_params in kohya_lora.py
    and lora_special.py for the exclusion logic.
    """
    
    def __init__(
        self,
        lora_name: str,
        lora_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        owner_transformer: Optional[str] = None,
    ):
        """
        Args:
            lora_name: Name of the wrapped LoRA module.
            lora_dim: Rank of the LoRA (number of gates).
            device: Device for gate parameters.
            dtype: Dtype for gate parameters (UNUSED — gates always float32 for precision).
            owner_transformer: Which expert owns this LoRA ('transformer_1', 'transformer_2', or None).
        """
        super().__init__()
        self.lora_name = lora_name
        self.r = lora_dim
        self.owner_transformer = owner_transformer
        
        # Per-rank gates: starts at 1 (all ranks active), anneals to {0,1}
        # Gates are NOT optimized by task loss autograd (they are excluded from the
        # optimizer parameter groups AND set requires_grad=False so they are not
        # in the autograd graph). Gate updates come from the dedicated SparseForge
        # rule which writes .data directly.
        # Gates are ALWAYS float32 to avoid bf16/fp16 quantization issues around 0.5 boundary.
        # They are cast to the compute dtype at the multiply site in forward.
        self.gates = nn.Parameter(
            torch.ones(self.r, device=device, dtype=torch.float32),
            requires_grad=False
        )
        # Mark this parameter as a gate parameter so it can be excluded from optimizer groups
        self.gates.rank_gate_gates = True
        
        # Track which gates are "dead" (permanently zeroed after hardening)
        self._hardened = False
        
        # Snapshot of soft gates at hardening start for correct interpolation
        self._soft_snapshot: Optional[Tensor] = None
        
        # References to LoRA weight matrices (plain list, NOT registered as nn.Module parameters)
        # This avoids nn.Module.__setattr__ intercepting and registering them in state_dict.
        self._lora_refs: List[Tensor] = []
        
        # Per-tensor final target (number of components to keep), computed at
        # annealing start from the tensor's energy spectrum (see
        # compute_tensor_budgets). None → QuenchSchedule falls back to the
        # global target_rank_ratio.
        self.k_final: Optional[int] = None
        self.budget_source: Optional[str] = None
    
    def apply_gates_to_matrices(self, A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply gates to LoRA matrices.
        
        For LoRA with A: (r, in_dim), B: (out_dim, r):
        W_eff = B @ diag(m) @ A = B @ (A * m[None,:])
        
        Args:
            A: Down projection matrix (r, in_dim) or (r, in_dim, k, k) for conv.
            B: Up projection matrix (out_dim, r) or (out_dim, r, 1, 1) for conv.
        
        Returns:
            (A_gated, B) where A_gated has gates applied.
        """
        m = self.gates.to(A.dtype)  # cast from float32 to compute dtype
        
        # Handle different shapes:
        # Linear: A is (r, in_dim), B is (out_dim, r)
        # Conv: A is (r, in_dim, k, k), B is (out_dim, r, 1, 1)
        if A.dim() == 2:
            # Linear: scale rows of A by gates
            A_gated = A * m.unsqueeze(1)  # (r, in_dim)
            return A_gated, B
        elif A.dim() == 4:
            # Conv: scale channels of A by gates
            A_gated = A * m.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (r, in_dim, k, k)
            return A_gated, B
        else:
            raise ValueError(f"Unexpected A shape: {A.shape}")
    
    def effective_weight(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Compute effective LoRA weight: W_eff = B @ diag(m) @ A.
        
        For linear: standard matrix mult.
        For conv: element-wise product along rank dimension, summed.
        """
        A_gated, B_gated = self.apply_gates_to_matrices(A, B)
        if A.dim() == 2:
            return B_gated @ A_gated
        elif A.dim() == 4:
            # Conv: A_gated (r, in_dim, k, k), B_gated (out_dim, r, 1, 1)
            # W_eff[out_dim, in_dim, k, k] = sum_r B[out_dim, r] * A_gated[r, :, :, :]
            return (B_gated.unsqueeze(2) * A_gated.unsqueeze(0)).sum(dim=2)
        else:
            raise ValueError(f"Unexpected A shape: {A.shape}")
    
    def harden_gates(self, threshold: float = 0.5):
        """
        Final hardening: convert soft gates to binary {0,1}.
        Called at the end of training.
        """
        if self._hardened:
            return
        with torch.no_grad():
            self.gates.data = (self.gates.data > threshold).float()
            self._hardened = True
    
    def is_hardened(self) -> bool:
        return self._hardened
    
    def active_ranks(self) -> int:
        """Number of currently active (non-zero) ranks."""
        return int((self.gates > 0.5).sum().item())
    
    def gate_histogram(self, bins: int = 10) -> Tensor:
        """Histogram of gate values for monitoring."""
        return torch.histc(self.gates, bins=bins, min=0.0, max=1.0)
    
    def extra_repr(self) -> str:
        return f"r={self.r}, active={self.active_ranks()}, hardened={self._hardened}"


class GatedDiff(nn.Module):
    """
    Per-element soft gate for a full-finetune (`.diff`) tensor.
    
    Covers every non-LoRA tensor in the network: 1D tensors (layer-norm
    gamma/beta, biases) and higher-dimensional full-finetune deltas (e.g. conv
    patch-embedding weights). These tensors are stored in `.diff` format (a
    delta added to the base weight) and have no LoRA rank axis, so the natural
    "rank" is the flat element index ordered by energy contribution:
    
        contribution_i = x_i^2 / sum_j x_j^2
    
    Forward application happens in FullModule.forward:
        eff_weight = base + (diff * m) * multiplier
    where m ∈ [0,1]^n is the per-element gate vector (flattened to the tensor
    shape). This keeps autograd flowing through the trainable diff while the
    gate slowly suppresses low-contribution elements.
    
    Like GatedLoRA, gates are NOT optimized by the task loss autograd; they
    are updated by the dedicated SparseForge rule (update_rank_gates) and MUST
    be excluded from the optimizer parameter groups (is_gate_parameter).
    """
    
    def __init__(
        self,
        lora_name: str,
        num_elements: int,
        device: torch.device,
        dtype: torch.dtype,
        owner_transformer: Optional[str] = None,
        param_ref: Optional[Tensor] = None,
        key_suffix: str = "diff",
    ):
        """
        Args:
            lora_name: Name of the owning FullModule (state_dict prefix).
            num_elements: Number of gated components (flattened tensor size).
            device: Device for gate parameters.
            dtype: Dtype (UNUSED — gates always float32 for precision).
            owner_transformer: Which expert owns this tensor (or None if shared).
            param_ref: The gated diff parameter (plain reference, NOT registered
                       as an nn.Module attribute to avoid state_dict duplication).
            key_suffix: state_dict key suffix of the gated parameter ('diff' or
                       'diff_b'), so truncation/folding can locate it.
        """
        super().__init__()
        self.lora_name = lora_name
        self.r = num_elements  # number of gated components (flat elements)
        self.owner_transformer = owner_transformer
        self.key_suffix = key_suffix
        
        # Per-element gates: starts at 1 (all elements active), anneals to {0,1}.
        # ALWAYS float32 to avoid bf16/fp16 quantization issues around the 0.5
        # boundary; cast to the compute dtype at the multiply site.
        self.gates = nn.Parameter(
            torch.ones(num_elements, device=device, dtype=torch.float32),
            requires_grad=False
        )
        # Mark as gate parameter so it is excluded from optimizer groups
        self.gates.rank_gate_gates = True
        
        self._hardened = False
        self._soft_snapshot: Optional[Tensor] = None
        
        # Reference to the gated diff parameter (plain attr, NOT registered)
        self._param_ref: Optional[Tensor] = param_ref
        
        # Per-tensor final target, set by compute_tensor_budgets
        self.k_final: Optional[int] = None
        self.budget_source: Optional[str] = None
    
    def apply_gates(self, tensor: Tensor) -> Tensor:
        """Element-wise gate application: tensor * m (m reshaped to tensor)."""
        m = self.gates.to(tensor.dtype).view(tensor.shape)
        return tensor * m
    
    def harden_gates(self, threshold: float = 0.5):
        """Final hardening: convert soft gates to binary {0,1}."""
        if self._hardened:
            return
        with torch.no_grad():
            self.gates.data = (self.gates.data > threshold).float()
            self._hardened = True
    
    def is_hardened(self) -> bool:
        return self._hardened
    
    def active_ranks(self) -> int:
        """Number of currently active (non-zero) elements."""
        return int((self.gates > 0.5).sum().item())
    
    def extra_repr(self) -> str:
        return f"n={self.r}, active={self.active_ranks()}, hardened={self._hardened}"


def _lora_budget(gl: 'GatedLoRA', target: float, device: torch.device) -> Optional[int]:
    """
    Final target rank for a LoRA pair from the SVD energy of the effective
    matrix B@A.
    
    Uses the factorized SVD  S = svd( diag(S_B) Vh_B @ U_A diag(S_A) )  which
    is O(r^2 * dim) instead of a full (out x in) SVD — identical to the
    recommended-rank computation in lib_om/lora_statistics_fix.py.
    
    Returns max(1, #{i : S_i^2 / sum(S^2) >= target}), or None if the weight
    refs are not available (caller falls back to target_rank_ratio).
    """
    if len(gl._lora_refs) < 2:
        return None
    A, B = gl._lora_refs
    A = A.detach().float().to(device)
    B = B.detach().float().to(device)
    if A.dim() == 4:  # conv: (r, in, k, k) -> (r, in*k*k)
        A = A.reshape(A.shape[0], -1)
    if B.dim() == 4:  # conv: (out, r, 1, 1) -> (out, r)
        B = B.reshape(B.shape[0], -1)
    U_B, S_B, Vh_B = torch.linalg.svd(B, full_matrices=False)
    U_A, S_A, _ = torch.linalg.svd(A, full_matrices=False)
    M = (S_B.unsqueeze(1) * Vh_B) @ (U_A * S_A.unsqueeze(0))
    S = torch.linalg.svdvals(M)
    S_sq = S * S
    total = S_sq.sum()
    if total <= 0:
        return 1
    k = int((S_sq / total >= target).sum().item())
    return max(1, k)


def _diff_budget(gl: 'GatedDiff', target: float, device: torch.device) -> Optional[int]:
    """
    Final target for a .diff tensor from per-element energy x^2/sum(x^2).
    
    Returns max(1, #{i : x_i^2 / sum(x^2) >= target}), or None if the param
    reference is not available (caller falls back to target_rank_ratio).
    """
    if gl._param_ref is None:
        return None
    x = gl._param_ref.detach().float().to(device).flatten()
    S_sq = x * x
    total = S_sq.sum()
    if total <= 0:
        return 1
    k = int((S_sq / total >= target).sum().item())
    return max(1, k)


def compute_tensor_budgets(
    gated_list: List,
    target_min_contribution: float = 1e-4,
    fallback_ratio: float = 0.3,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> Dict[str, int]:
    """
    Compute per-tensor final target component counts (k_final) from the
    CURRENT weights' energy spectrum.
    
    Called ONCE at annealing start (after LR warmup) so that each tensor's
    target reflects its trained state, not its initialization. Every tensor
    gets its own target (per-tensor annealing, not a global ratio):
    
    - GatedLoRA: k = #{i : S_i^2 / sum(S^2) >= target} for S = svd(B@A)
    - GatedDiff: k = #{i : x_i^2 / sum(x^2) >= target} (per element)
    
    Tensors with a degenerate (all-zero) spectrum get k_final = 1. Tensors
    whose budget fails to compute keep k_final = None and fall back to
    QuenchSchedule.target_rank_ratio (via fallback_ratio for reporting).
    
    Args:
        gated_list: List of GatedLoRA / GatedDiff instances.
        target_min_contribution: Minimum fraction of total energy a component
            must contribute to be kept (default 1e-4 = 0.01%).
        fallback_ratio: Ratio used only for the summary report of tensors
            that failed budget computation.
        device: Device for the one-shot SVD/energy computation. None = auto
            (CUDA when available, else CPU).
        progress: Show a tqdm progress bar.
    
    Returns:
        Summary dict: {'tensors', 'kept_components', 'total_components'}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from tqdm import tqdm
    
    items = list(gated_list)
    pbar = tqdm(items, desc="[RankGates] Per-tensor rank budgets", disable=not progress)
    for gl in pbar:
        try:
            if isinstance(gl, GatedDiff):
                gl.k_final = _diff_budget(gl, target_min_contribution, device)
            else:
                gl.k_final = _lora_budget(gl, target_min_contribution, device)
            gl.budget_source = "energy_spectrum" if gl.k_final is not None else "ratio_fallback"
        except Exception as e:
            print(f"[RankGates WARNING] budget computation failed for '{gl.lora_name}': {e} "
                  f"- falling back to target_rank_ratio")
            gl.k_final = None
            gl.budget_source = "ratio_fallback"
    pbar.close()
    
    # Summary, grouped by owner expert
    by_owner: Dict[Optional[str], List] = {}
    for gl in items:
        by_owner.setdefault(gl.owner_transformer, []).append(gl)
    
    print(f"\n[RankGates] Per-tensor rank budgets computed "
          f"(target_min_contribution={target_min_contribution}, device={device}):")
    total_c = kept_c = 0
    for owner in sorted(by_owner.keys(), key=lambda o: (o is None, o or "")):
        glist = by_owner[owner]
        label = owner or "shared"
        tot = sum(g.r for g in glist)
        fin = sum((g.k_final if g.k_final is not None
                   else max(1, int(round(g.r * fallback_ratio)))) for g in glist)
        total_c += tot
        kept_c += fin
        n_diff = sum(1 for g in glist if isinstance(g, GatedDiff))
        print(f"  {label}: {len(glist)} tensors ({len(glist) - n_diff} LoRA, {n_diff} diff), "
              f"final target {fin}/{tot} components ({100.0 * fin / max(1, tot):.1f}%)")
    print(f"  Total: {kept_c}/{total_c} components ({100.0 * kept_c / max(1, total_c):.1f}%)")
    
    return {"tensors": len(items), "kept_components": kept_c, "total_components": total_c}


def is_gate_parameter(param) -> bool:
    """
    Check if a parameter is a rank gate parameter.
    
    Gate parameters are marked with a 'rank_gate_gates' attribute for identification.
    """
    return hasattr(param, 'rank_gate_gates')


def get_params_excluding_gates(module):
    """
    Yield all parameters of a module except rank gate parameters.
    
    Used when building optimizer parameter groups to ensure gates are NOT
    updated by the optimizer (they are updated by the dedicated SparseForge rule).
    """
    for name, param in module.named_parameters():
        if not is_gate_parameter(param):
            yield param


class FisherTracker:
    """
    Tracks Fisher information diagonal via EMA of squared gradients.
    
    For diffusion training where we have gradients every step, this is much
    cheaper than the Hutchinson HVP estimator used in SparseForge for post-training
    pruning. Near a local optimum, F ≈ E[g²] approximates the Hessian diagonal.
    
    Per wan_sparse_annealing.md: for mid-training diffusion where ∇L is not small,
    we use the full Taylor expansion (WoodFisher-style):
        s = |g_ema ⊙ w| + 0.5 · fisher_ema ⊙ w²
    """
    
    def __init__(self, decay: float = 0.999, use_first_order: bool = True):
        """
        Args:
            decay: EMA decay factor. Higher = smoother but slower to adapt.
            use_first_order: If True, include |g·w| term (recommended for diffusion).
        """
        self.decay = decay
        self.use_first_order = use_first_order
        self.grad_ema: Dict[int, Tensor] = {}  # param_id → EMA of gradients
        self.fisher_ema: Dict[int, Tensor] = {}  # param_id → EMA of g²
    
    def update(self, params):
        """
        Update EMA statistics for all parameters with gradients.
        Call after backward() but before optimizer.step().
        
        Args:
            params: Can be a list of tensors OR a list of optimizer param group dicts
                    (each with 'params' key containing tensors).
        """
        # Handle optimizer param group dicts: [{'params': [tensors...], 'lr': ...}, ...]
        if len(params) > 0 and isinstance(params[0], dict):
            params = (p for group in params for p in group.get('params', []))
        
        for p in params:
            if p.grad is None:
                continue
            pid = id(p)
            g = p.grad.detach().float()  # store EMAs in float32 to avoid bf16 overflow
            g2 = g ** 2
            
            if pid not in self.grad_ema:
                self.grad_ema[pid] = g.clone()
                self.fisher_ema[pid] = g2.clone()
            else:
                self.grad_ema[pid].mul_(self.decay).add_(g, alpha=1 - self.decay)
                self.fisher_ema[pid].mul_(self.decay).add_(g2, alpha=1 - self.decay)
    
    def obd_score(self, param: Tensor) -> Tensor:
        """
        Compute OBD-style importance score for a parameter.
        
        Uses full Taylor expansion (WoodFisher-style) for diffusion training:
            s = |g_ema ⊙ w| + 0.5 · fisher_ema ⊙ w²
        
        This handles the case where gradients are not small mid-training.
        
        Args:
            param: The parameter tensor.
        
        Returns:
            Score tensor of same shape as param (in float32 for numerical stability).
        """
        pid = id(param)
        w = param.data.float()  # compute in float32
        
        if pid not in self.fisher_ema:
            # Fallback to magnitude-based score if no stats yet
            return w ** 2
        
        fisher = self.fisher_ema[pid]  # already float32
        second_order = 0.5 * fisher * (w ** 2)
        
        if self.use_first_order and pid in self.grad_ema:
            grad = self.grad_ema[pid]  # already float32
            first_order = torch.abs(grad * w)
            return first_order + second_order
        else:
            return second_order
    
    def reset(self):
        """Reset all tracked statistics."""
        self.grad_ema.clear()
        self.fisher_ema.clear()
    
    def clear_param(self, param):
        """Remove statistics for a specific parameter."""
        pid = id(param)
        self.grad_ema.pop(pid, None)
        self.fisher_ema.pop(pid, None)


class LearningAwareScheduler:
    """
    Per-expert, LOSS- and LR-aware annealing schedule (replaces hardcoded
    start/end/hardening percentages).

    Each expert gets its OWN instance with its own step counter, its own loss
    EMAs, its own LR tracking and its own phase state, so experts anneal
    completely independently of each other.

    The trainer calls ``record(loss, lr, step, total_steps)`` once per step the
    expert is active. The scheduler then:

    * **Annealing START** — automatically, when the expert's loss has plateaued
      (fast EMA ≈ slow EMA for ``plateau_confirm_steps`` consecutive steps, and
      after ``min_anneal_steps`` / warmup).
    * **Annealing END** — driven by the LR itself: progress = how far the LR has
      decayed from its value at anneal-start down to ``end_lr_fraction * peak``.
      If the LR is constant (no decay), a step-based clock of length
      ``anneal_max_duration`` is used instead.
    * **Hardening START** — automatically, when the LR has decayed below
      ``hardening_lr_fraction * peak`` (learning is essentially finished); for a
      constant LR it triggers ``hardening_min_steps`` before the end.

    The same phase quantities the legacy QuenchSchedule exposed as step
    functions (target rank, temperature, beta, lambda_mid, hardening_x) are now
    properties computed from the detected timeline.
    """

    PHASE_HEATING = "heating"
    PHASE_ANNEALING = "annealing"
    PHASE_HARDENING = "hardening"
    PHASE_DONE = "done"

    def __init__(
        self,
        expert_name: str,
        total_steps: int,
        *,
        target_rank_ratio: float = 0.3,
        temperature: float = 1.0,
        gamma: float = 0.95,
        alpha: float = 0.1,
        lambda_mid_max: float = 0.01,
        update_every: int = 15,
        eta_pen: float = 0.01,
        # learning-aware knobs
        plateau_relative_threshold: float = 5e-3,
        plateau_confirm_steps: int = 50,
        min_anneal_steps: int = 200,
        end_lr_fraction: float = 0.2,
        anneal_max_duration: int = 1500,
        hardening_lr_fraction: float = 0.05,
        hardening_min_steps: int = 150,
        loss_ema_fast: float = 0.10,
        loss_ema_slow: float = 0.02,
        start_after_warmup: bool = True,
        warmup_steps: int = 0,
        # Optional MANUAL timeline (per-expert steps). If manual_start_step is
        # not None, detection is bypassed and the timeline is fixed to these
        # values (legacy / user-specified start_step, end_step, hardening).
        manual_start_step: Optional[int] = None,
        manual_end_step: Optional[int] = None,
        manual_hardening_start: Optional[int] = None,
    ):
        self.expert_name = expert_name
        self.total_steps = max(1, int(total_steps))
        self.manual = manual_start_step is not None
        self.manual_start_step = manual_start_step
        self.manual_end_step = manual_end_step
        self.manual_hardening_start = manual_hardening_start

        # schedule math params (kept from the legacy QuenchSchedule)
        self.target_rank_ratio = target_rank_ratio
        self.T0 = temperature
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_mid_max = lambda_mid_max
        self.update_every = max(1, int(update_every))
        self.eta_pen = eta_pen

        # learning-aware knobs
        self.plateau_relative_threshold = plateau_relative_threshold
        self.plateau_confirm_steps = max(1, int(plateau_confirm_steps))
        self.min_anneal_steps = max(1, int(min_anneal_steps))
        self.end_lr_fraction = end_lr_fraction
        self.anneal_max_duration = max(1, int(anneal_max_duration))
        self.hardening_lr_fraction = hardening_lr_fraction
        self.hardening_min_steps = max(1, int(hardening_min_steps))
        self.loss_ema_fast = loss_ema_fast
        self.loss_ema_slow = loss_ema_slow

        # The annealing floor also respects the LR warmup so the first gate
        # decisions / budget computation use stable gradients + primed Fisher.
        self.anneal_floor = self.min_anneal_steps
        if start_after_warmup:
            self.anneal_floor = max(self.anneal_floor, int(warmup_steps) + 1)

        # --- mutable state (all per-expert, fully independent) ---
        self.step = 0
        self.record_count = 0          # number of record() calls this expert
        self.update_count = 0          # number of gate updates this expert got
        self.loss_ema_fast_v: Optional[float] = None
        self.loss_ema_slow_v: Optional[float] = None
        self.lr_current: Optional[float] = None
        self.lr_peak: Optional[float] = None
        self._flat_counter = 0
        self._lr_ever_decayed = False

        # detected timeline (per-expert)
        self.anneal_start_step: Optional[int] = None
        self.lr_at_anneal_start: Optional[float] = None
        self.annealing_progress: float = 0.0
        self.hardening_start_step: Optional[int] = None
        self.hardening_progress: float = 0.0

        # set by the trainer once per-tensor budgets are computed for this expert
        self.budgets_computed: bool = False

        self._log_every = 500

    # ------------------------------------------------------------------
    # main entry point — call once per step the expert is active
    # ------------------------------------------------------------------
    def record(self, loss: Optional[float], lr: Optional[float], step: int,
               total_steps: Optional[int] = None) -> Dict[str, float]:
        if total_steps is not None:
            self.total_steps = max(1, int(total_steps))
        self.step = int(step)
        self.record_count += 1

        # ---- loss EMAs (plateau detection) ----
        if loss is not None and isfinite(loss):
            loss = float(loss)
            if self.loss_ema_fast_v is None:
                self.loss_ema_fast_v = loss
                self.loss_ema_slow_v = loss
            else:
                self.loss_ema_fast_v = (self.loss_ema_fast_v * (1 - self.loss_ema_fast)
                                        + loss * self.loss_ema_fast)
                self.loss_ema_slow_v = (self.loss_ema_slow_v * (1 - self.loss_ema_slow)
                                        + loss * self.loss_ema_slow)

        # ---- LR tracking ----
        if lr is not None and isfinite(lr) and lr >= 0:
            lr = float(lr)
            self.lr_current = lr
            if self.lr_peak is None or lr > self.lr_peak:
                self.lr_peak = lr
            # detect whether the LR has ever meaningfully decayed below its peak
            if self.lr_peak is not None and self.lr_peak > 0:
                if lr < self.lr_peak * 0.999:
                    self._lr_ever_decayed = True

        self._detect_timeline()
        return self.status()

    # ------------------------------------------------------------------
    def _lr_ratio(self) -> float:
        if self.lr_peak is not None and self.lr_peak > 0 and self.lr_current is not None:
            return self.lr_current / self.lr_peak
        return 1.0

    def _detect_timeline(self):
        # ================= MANUAL TIMELINE (legacy / user-specified) ==========
        if self.manual:
            if self.anneal_start_step is None and self.step >= self.manual_start_step:
                self.anneal_start_step = self.manual_start_step
                self.lr_at_anneal_start = (self.lr_current
                                           if self.lr_current is not None else self.lr_peak)
            if self.anneal_start_step is not None:
                denom = max(1, self.manual_end_step - self.manual_start_step) \
                    if self.manual_end_step else self.anneal_max_duration
                self.annealing_progress = _clamp01(
                    (self.step - self.anneal_start_step) / denom)
            if (self.annealing_progress >= 1.0 and self.hardening_start_step is None
                    and self.manual_hardening_start is not None
                    and self.step >= self.manual_hardening_start):
                self.hardening_start_step = self.manual_hardening_start
            if self.hardening_start_step is not None:
                denom = max(1, self.total_steps - self.hardening_start_step)
                self.hardening_progress = _clamp01(
                    (self.step - self.hardening_start_step) / denom)
            return

        # ================= ANNEALING START =================
        if self.anneal_start_step is None:
            if (self.step >= self.anneal_floor
                    and self.loss_ema_slow_v is not None
                    and self.loss_ema_slow_v > 0):
                rel_improve = ((self.loss_ema_slow_v - self.loss_ema_fast_v)
                               / (abs(self.loss_ema_slow_v) + 1e-8))
                if rel_improve < self.plateau_relative_threshold:
                    self._flat_counter += 1
                else:
                    self._flat_counter = 0
                if self._flat_counter >= self.plateau_confirm_steps:
                    self.anneal_start_step = self.step
                    self.lr_at_anneal_start = (self.lr_current
                                               if self.lr_current is not None else self.lr_peak)
                    print(f"[RankGates:{self.expert_name}] ANNEALING START at per-expert "
                          f"step {self.step} (loss plateau detected: "
                          f"fast={self.loss_ema_fast_v:.5f} slow={self.loss_ema_slow_v:.5f}, "
                          f"lr={self.lr_current}, lr_peak={self.lr_peak})")
            return

        # ================= ANNEALING PROGRESS =================
        lr_target = (self.lr_peak * self.end_lr_fraction) if self.lr_peak else 0.0
        lr_driven = (self._lr_ever_decayed
                     and self.lr_at_anneal_start is not None
                     and self.lr_current is not None
                     and self.lr_at_anneal_start > lr_target * 1.02)
        if lr_driven:
            denom = self.lr_at_anneal_start - lr_target
            if denom > 1e-12:
                self.annealing_progress = _clamp01((self.lr_at_anneal_start - self.lr_current) / denom)
            else:
                self.annealing_progress = _clamp01(
                    (self.step - self.anneal_start_step) / self.anneal_max_duration)
        else:
            # constant LR (or no scheduler): step-based clock
            self.annealing_progress = _clamp01(
                (self.step - self.anneal_start_step) / self.anneal_max_duration)

        # ================= HARDENING =================
        if self.annealing_progress >= 1.0 and self.hardening_start_step is None:
            lr_ratio = self._lr_ratio()
            trigger = False
            if self._lr_ever_decayed and self.lr_current is not None:
                if lr_ratio < self.hardening_lr_fraction:
                    trigger = True
            else:
                # constant LR: trigger near the end
                if self.step >= self.total_steps - self.hardening_min_steps:
                    trigger = True
            if trigger:
                self.hardening_start_step = self.step
                print(f"[RankGates:{self.expert_name}] HARDENING START at per-expert "
                      f"step {self.step} (lr_ratio={lr_ratio:.4f})")

        if self.hardening_start_step is not None:
            denom = max(1, self.total_steps - self.hardening_start_step)
            self.hardening_progress = _clamp01((self.step - self.hardening_start_step) / denom)

    # ------------------------------------------------------------------
    # phase quantities (mirror the legacy QuenchSchedule interface)
    # ------------------------------------------------------------------
    @property
    def phase(self) -> str:
        if self.hardening_start_step is not None:
            return self.PHASE_DONE if self.hardening_progress >= 1.0 else self.PHASE_HARDENING
        if self.anneal_start_step is not None:
            # annealing in progress OR completed (awaiting LR-driven hardening)
            return self.PHASE_ANNEALING
        return self.PHASE_HEATING

    def target_rank(self, r: int, k_final: Optional[int] = None) -> int:
        """Target number of active components at the current annealing progress."""
        p = self.annealing_progress
        if k_final is None:
            k_final = max(1, int(round(r * self.target_rank_ratio)))
        k_final = max(1, min(k_final, r))
        return int(round(r - (r - k_final) * p))

    @property
    def temperature(self) -> float:
        return self.T0 * (self.gamma ** self.update_count)

    @property
    def beta(self) -> float:
        p = self.annealing_progress
        return p * p * (3 - 2 * p)  # smoothstep

    @property
    def lambda_mid(self) -> float:
        p = self.annealing_progress
        return self.lambda_mid_max * (p ** 2)

    @property
    def hardening_x(self) -> float:
        """Soft->hard interpolation factor: 1 (soft) before hardening, 0 after."""
        if self.hardening_start_step is None:
            return 1.0
        return 1.0 - self.hardening_progress

    # ------------------------------------------------------------------
    def should_update(self) -> bool:
        """Gate update due: annealing in progress, every update_every records."""
        return (self.anneal_start_step is not None
                and self.annealing_progress < 1.0
                and self.record_count % self.update_every == 0)

    def is_hardening(self) -> bool:
        return self.hardening_start_step is not None and self.hardening_progress < 1.0

    def is_complete(self) -> bool:
        return self.hardening_start_step is not None and self.hardening_progress >= 1.0

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, float]:
        """Snapshot of the per-expert state for logging."""
        return {
            "step": self.step,
            "phase": self.phase,
            "anneal_start": self.anneal_start_step if self.anneal_start_step is not None else -1,
            "annealing_progress": self.annealing_progress,
            "hardening_start": self.hardening_start_step if self.hardening_start_step is not None else -1,
            "hardening_progress": self.hardening_progress,
            "loss_ema_fast": self.loss_ema_fast_v if self.loss_ema_fast_v is not None else float("nan"),
            "loss_ema_slow": self.loss_ema_slow_v if self.loss_ema_slow_v is not None else float("nan"),
            "lr_current": self.lr_current if self.lr_current is not None else float("nan"),
            "lr_peak": self.lr_peak if self.lr_peak is not None else float("nan"),
            "lr_ratio": self._lr_ratio(),
            "temperature": self.temperature,
            "update_count": self.update_count,
        }


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class QuenchSchedule:
    """
    SparseForge-style progressive quenching schedules.
    
    Implements three coupled schedules from the paper:
    1. Target rank: r → r_final via cubic ramp over [t0, t1]
    2. Temperature: geometric decay T ← γT each update (sharpens sigmoid)
    3. Beta blend: smoothstep 0→1 blending soft gate → hard TopK target
    
    Plus the binary preference penalty λ_mid that ramps up late.
    
    The process: Heating (explore, soft gates) → Progressive Quench → Final Hardening Window.
    """
    
    def __init__(
        self,
        total_steps: int,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        target_rank_ratio: float = 0.3,
        temperature: float = 1.0,
        gamma: float = 0.95,
        alpha: float = 0.1,
        lambda_mid_max: float = 0.01,
        update_every: int = 15,
        hardening_window: int = 500,
        eta_pen: float = 0.01,
    ):
        """
        Args:
            total_steps: Total training steps.
            start_step: When to begin annealing. Defaults to 5% of total_steps.
            end_step: When to complete annealing. Defaults to 75% of total_steps.
            target_rank_ratio: Final active rank fraction (0.3 = keep 30% of ranks, aggressive pruning).
            temperature: Initial sigmoid temperature.
            gamma: Temperature decay per update step (0.95 = faster sharpening).
            alpha: EMA update rate for gates (0.1 = faster gate updates).
            lambda_mid_max: Maximum binary preference penalty (0.01 = stronger decisiveness push).
            update_every: Update gates every N steps (15 = more frequent updates).
            hardening_window: Steps for final soft→hard interpolation.
            eta_pen: Penalty coefficient for mid-preference nudge.
        """
        self.total_steps = total_steps
        
        # Default timing
        if start_step is None:
            self.start_step = max(100, int(total_steps * 0.05))
        else:
            self.start_step = start_step
        
        if end_step is None:
            self.end_step = min(total_steps - hardening_window, int(total_steps * 0.75))
        else:
            self.end_step = end_step
        
        self.target_rank_ratio = target_rank_ratio
        self.T0 = temperature
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_mid_max = lambda_mid_max
        self.update_every = update_every
        self.hardening_window = hardening_window
        self.eta_pen = eta_pen
        
        # Hardening window starts after end_step
        self.hardening_start = max(self.end_step, total_steps - hardening_window)
        self.hardening_end = total_steps
        
        # Validate configuration
        if self.start_step >= self.end_step:
            print(f"[RankGates WARNING] start_step ({self.start_step}) >= end_step ({self.end_step}). "
                  f"Annealing range is degenerate; all gates will jump to final values immediately.")
        if self.end_step > total_steps - hardening_window:
            actual_window = total_steps - self.end_step
            print(f"[RankGates WARNING] end_step ({self.end_step}) is too late; "
                  f"hardening window shrunk from {hardening_window} to {actual_window} steps.")
        if self.hardening_start >= self.hardening_end:
            print(f"[RankGates WARNING] Hardening window is empty "
                  f"(start={self.hardening_start}, end={self.hardening_end}). Gates will not interpolate.")
    
    def _phase(self, step: int) -> float:
        """
        Annealing phase: 0 before t0, 1 after t1, cubic between.
        """
        if step < self.start_step:
            return 0.0
        if step > self.end_step:
            return 1.0
        denom = self.end_step - self.start_step
        if denom <= 0:
            return 1.0  # degenerate case: start == end, treat as complete
        x = (step - self.start_step) / denom
        return 1 - (1 - x) ** 3  # Cubic ramp
    
    def target_rank_at(self, step: int, r: int, k_final: Optional[int] = None) -> int:
        """
        Target number of active components at given step.
        Anneals from r → k_final over [start_step, end_step] (cubic ramp).
        
        Args:
            step: Current global step.
            r: Total number of components of the tensor (rank or flat size).
            k_final: Per-tensor final target computed by compute_tensor_budgets
                     (per-tensor annealing). If None, falls back to the global
                     target_rank_ratio (legacy behavior).
        """
        p = self._phase(step)
        if k_final is None:
            k_final = max(1, int(round(r * self.target_rank_ratio)))
        k_final = max(1, min(k_final, r))
        return int(round(r - (r - k_final) * p))
    
    def temperature_at(self, step: int, update_count: Optional[int] = None) -> float:
        """
        Temperature for sigmoid gating. Geometric decay sharpens decisions over time.
        T ← γT each mask update.
        
        Args:
            step: Current training step (used for start_step check).
            update_count: Optional explicit count of updates this entity has received.
                          If provided, temperature is computed from this count instead
                          of inferring from step. This is important for per-expert
                          training where each expert only receives ~1/N of the updates.
        """
        if step < self.start_step:
            return self.T0
        if update_count is not None:
            # Use explicit update count (e.g., per-expert count)
            updates = max(0, update_count)
        else:
            # Infer from global step (original behavior, single-expert case)
            updates = max(0, (step - self.start_step) // self.update_every)
        return self.T0 * (self.gamma ** updates)
    
    def beta_at(self, step: int) -> float:
        """
        Blend factor: 0→1 via smoothstep.
        Controls mixing between soft exploratory gate G and hard structured target Ḡ.
        
        G ← (1-β)·G_soft + β·G_hard
        """
        p = self._phase(step)
        return p * p * (3 - 2 * p)  # Smoothstep
    
    def lambda_mid_at(self, step: int) -> float:
        """
        Binary preference penalty weight.
        Weak early (allow exploration), strong late (push toward {0,1}).
        """
        p = self._phase(step)
        return self.lambda_mid_max * (p ** 2)
    
    def hardening_x_at(self, step: int) -> float:
        """
        Hardening interpolation factor: 1→0 over hardening window.
        
        m_eff = x·m + (1-x)·1[m>θ]
        
        Returns 1.0 before hardening window, 0.0 after.
        """
        if step < self.hardening_start:
            return 1.0
        if step >= self.hardening_end:
            return 0.0
        x = (step - self.hardening_start) / (self.hardening_end - self.hardening_start)
        return 1.0 - x
    
    def should_update(self, step: int) -> bool:
        """Whether to update gates at this step."""
        return (
            step >= self.start_step and
            step < self.hardening_start and
            step % self.update_every == 0
        )
    
    def is_hardening(self, step: int) -> bool:
        """Whether we're in the final hardening window."""
        return self.hardening_start <= step < self.hardening_end
    
    def is_complete(self, step: int) -> bool:
        """Whether annealing is complete (gates should be binary)."""
        return step >= self.hardening_end

    # ------------------------------------------------------------------
    # Adapter interface matching LearningAwareScheduler, so that
    # update_rank_gates / apply_hardening_interpolation work with either.
    # The trainer binds the current step via bind() before each call.
    # ------------------------------------------------------------------
    def bind(self, step: int) -> "QuenchSchedule":
        self._bound_step = int(step)
        return self

    @property
    def _bound(self) -> int:
        return getattr(self, '_bound_step', self.start_step)

    def target_rank(self, r: int, k_final: Optional[int] = None) -> int:
        return self.target_rank_at(self._bound, r, k_final)

    @property
    def temperature(self) -> float:
        return self.temperature_at(self._bound)

    @property
    def beta(self) -> float:
        return self.beta_at(self._bound)

    @property
    def lambda_mid(self) -> float:
        return self.lambda_mid_at(self._bound)

    @property
    def hardening_x(self) -> float:
        return self.hardening_x_at(self._bound)


def update_rank_gates(
    gated_loras: List,
    fisher_tracker: FisherTracker,
    sched: QuenchSchedule,
    step: int,
    active_experts: Optional[set] = None,
    expert_update_counts: Optional[Dict[str, int]] = None,
) -> Tensor:
    """
    SparseForge Algorithm 1 translated to LoRA rank gates.
    
    Computes curvature-guided importance scores and updates gates via EMA.
    Called every sched.update_every steps during the annealing phase.
    
    For dual expert training (Wan 2.2 14B), gates are updated ONLY for the
    currently active expert(s). Frozen experts' gates are completely untouched
    to maintain per-expert independence (two independent LoRAs being trained).
    
    For each gated tensor (GatedLoRA or GatedDiff) of an active expert:
    1. Compute per-component OBD score (per-rank for LoRA pairs, per-element
       for .diff tensors): s = |g·w| + 0.5·F ⊙ w² (curvature × energy)
    2. Standardize scores tensor-wise
    3. Compute soft gate G = sigmoid((s - τ_k) / T) around k-th largest score,
       where k is the PER-TENSOR target at this step (sched.target_rank_at
       with the tensor's k_final from compute_tensor_budgets)
    4. Compute hard TopK target Ḡ
    5. Blend: G ← (1-β)·G_soft + β·Ḡ
    6. Apply mid-preference nudge
    7. EMA update: m ← (1-α)·m + α·G
    
    Args:
        gated_loras: List of GatedLoRA / GatedDiff modules to update.
        fisher_tracker: Fisher diagonal tracker.
        sched: QuenchSchedule for current step values.
        step: Current training step (global step).
        active_experts: Set of active expert names (e.g. {'transformer_1'}). Only
                       gates for these experts will be updated. Frozen experts
                       are skipped entirely to maintain per-expert independence.
        expert_update_counts: Optional dict mapping expert name → count of gate
                              updates that expert has received. Used for correct
                              temperature decay in per-expert training.
    
    Returns:
        L_mid: Binary preference loss term (Σ m(1-m)). Logged for monitoring only;
               NOT added to training loss. Per SparseForge eq. 4, this is used
               as a diagnostic to track gate binarization progress.
    """
    if not gated_loras:
        return torch.tensor(0.0)
    
    # Track per-expert stats for logging
    expert_stats = {}  # owner -> {total_ranks, active_ranks, avg_score}
    
    with torch.no_grad():  # gate updates are a dedicated rule, not autograd
        for gl in gated_loras:
            # Skip frozen experts during per-expert training.
            # When active_experts is specified (Wan 2.2 multistage), only update gates
            # for the currently active expert(s). Frozen experts must remain completely
            # untouched to maintain per-expert independence (training two separate LoRAs).
            if active_experts is not None and gl.owner_transformer is not None:
                if gl.owner_transformer not in active_experts:
                    continue
            
            # Skip already-hardened gates (finalized at end of training)
            if gl.is_hardened():
                continue
            
            r = gl.r
            
            # IMPORTANT: score on the ORIGINAL parameters, not copies.
            # FisherTracker keys by id(param), and .to(device) creates a new tensor
            # with a new id. Under layer offloading, weights may live on CPU-pinned
            # storage while gates are on GPU — scoring the copies would silently
            # fall back to magnitude-only (w**2), defeating the whole purpose.
            gate_device = gl.gates.device
            
            if isinstance(gl, GatedDiff):
                # .diff tensor (full finetune): per-element OBD score over the
                # flat diff vector (1D norms/biases or N-dim conv deltas).
                param = gl._param_ref
                if param is None or param.numel() == 0:
                    continue
                s = fisher_tracker.obd_score(param).flatten().to(gate_device)  # (n,)
            else:
                # LoRA pair: score on the underlying weight matrices via plain
                # list refs (NOT registered as nn.Module parameters to avoid
                # state_dict bloat).
                if len(gl._lora_refs) < 2:
                    continue  # refs not set up yet
                A = gl._lora_refs[0]  # (r, in_dim) - original nn.Parameter
                B = gl._lora_refs[1]  # (out_dim, r) - original nn.Parameter
                score_A = fisher_tracker.obd_score(A).sum(dim=1).to(gate_device)  # (r,)
                score_B = fisher_tracker.obd_score(B).sum(dim=0).to(gate_device)  # (r,)
                s = score_A + score_B
            
            # Handle degenerate case
            if s.numel() == 0:
                continue
            
            # Tensor-wise standardization (their ˜s)
            # Use unbiased=False to avoid NaN for single-component tensors
            s_std = s.std(unbiased=False)
            s = (s - s.mean()) / (s_std + 1e-8)
            
            # Per-tensor target k components at the current annealing progress.
            # `sched` is this expert's LearningAwareScheduler (or a legacy
            # QuenchSchedule bound to the current step); its phase values are
            # derived from the detected loss-plateau / LR-decay timeline.
            # k_final was computed at annealing start by compute_tensor_budgets;
            # None → global target_rank_ratio fallback.
            k = sched.target_rank(r, gl.k_final)
            
            # Threshold: k-th largest score (0-indexed: index k-1)
            # Fix: use max(k-1, 0) not min(k, r-1) which was off-by-one
            tau = s.sort(descending=True).values[max(k - 1, 0)]
            
            # Temperature (per-expert, decays with this expert's own update count).
            T = max(sched.temperature, 1e-6)
            
            # Soft gate: sigmoid around threshold (their eq. 3)
            G_soft = torch.sigmoid((s - tau) / T)
            
            # Hard structured target: TopK (their Ḡ)
            G_hard = torch.zeros(r, device=s.device, dtype=s.dtype)
            if k > 0:
                G_hard[s.topk(k).indices] = 1.0
            
            # Progressive structure blend: β: 0→1 on smoothstep (their §4.3)
            beta = sched.beta
            G = (1 - beta) * G_soft + beta * G_hard
            
            # Binary-preference nudge: gradient of L_mid = Σ m(1-m) w.r.t. target
            # (their eq. 4)
            lam_mid = sched.lambda_mid
            G = (G - sched.eta_pen * lam_mid * (gl.gates.data - G)).clamp(0, 1)
            
            # EMA update preserves continuity (their eq. 5)
            gl.gates.data.mul_(1 - sched.alpha).add_(G, alpha=sched.alpha)
            
            # Track per-expert stats
            owner = gl.owner_transformer or "shared"
            if owner not in expert_stats:
                expert_stats[owner] = {"total_ranks": 0, "active_ranks": 0, "scores": []}
            expert_stats[owner]["total_ranks"] += r
            expert_stats[owner]["active_ranks"] += int((gl.gates > 0.5).sum().item())
            expert_stats[owner]["scores"].append(s.mean().item())
    
    # Compute L_mid for training loss (only for active experts during per-expert training)
    L_mid_sum = torch.tensor(0.0, device=gated_loras[0].gates.device)
    active_gate_count = 0
    for gl in gated_loras:
        # Skip frozen experts during per-expert training
        if active_experts is not None and gl.owner_transformer is not None:
            if gl.owner_transformer not in active_experts:
                continue
        m = gl.gates
        L_mid_sum = L_mid_sum + (m * (1 - m)).mean()
        active_gate_count += 1
    
    if active_gate_count > 0:
        L_mid = L_mid_sum / active_gate_count
    else:
        L_mid = torch.tensor(0.0, device=gated_loras[0].gates.device)
    
    # Log per-expert stats periodically
    if step % 1000 == 0 and expert_stats:
        print(f"\n[RankGates Update step={step}]")
        for owner, stats in expert_stats.items():
            active_pct = 100 * stats["active_ranks"] / max(1, stats["total_ranks"])
            avg_score = sum(stats["scores"]) / max(1, len(stats["scores"]))
            is_active = "ACTIVE" if active_experts and owner in active_experts else ""
            print(f"  Expert {owner}: {stats['active_ranks']}/{stats['total_ranks']} active "
                  f"({active_pct:.1f}%), avg_score={avg_score:.3f} {is_active}")
    
    return L_mid


def apply_hardening_interpolation(
    gated_loras: List[GatedLoRA],
    sched: QuenchSchedule,
    step: int,
    threshold: float = 0.5,
):
    """
    Apply the final hardening interpolation during the hardening window.
    
    Uses a snapshot of soft gates taken at hardening start to avoid compounding:
        m_eff = x·m_snapshot + (1-x)·1[m_snapshot>θ]
    
    where x: 1→0 over the hardening window.
    
    Applied to ALL gated LoRAs regardless of which expert is active.
    The interpolation is idempotent (snapshot-based, no gradients), so applying
    it to frozen experts during the hardening window is safe and necessary:
    with switch_boundary_every > hardening_window, an expert could otherwise be
    inactive for the entire window and its gates would jump abruptly from soft
    to binary at finalize_gates, defeating the purpose of the hardening window.
    """
    if not gated_loras:
        return
    
    # LearningAwareScheduler exposes hardening_x as a property (per-expert
    # timeline); legacy QuenchSchedule computes it from the bound step.
    x = sched.hardening_x
    
    with torch.no_grad():
        for gl in gated_loras:
            # Snapshot soft gates at hardening start (only once per GatedLoRA)
            if gl._soft_snapshot is None:
                gl._soft_snapshot = gl.gates.data.clone()
            
            m_snapshot = gl._soft_snapshot
            m_hard = (m_snapshot > threshold).float()
            m_eff = x * m_snapshot + (1 - x) * m_hard
            gl.gates.data.copy_(m_eff)


def finalize_gates(
    gated_loras: List[GatedLoRA],
    threshold: float = 0.5,
):
    """
    Finalize gates to binary {0,1}.
    Called at the very end of training.
    
    Finalizes ALL gated_loras (no active_experts filter) — at training end,
    both experts should be finalized. The previous per-expert filtering was
    buggy: once any expert finalized, the any(gl.is_hardened()) guard in
    SDTrainer blocked the other expert forever.

    Args:
        gated_loras: List of GatedLoRA modules to finalize.
        threshold: Threshold for binarization (gate > threshold → 1, else → 0).
    """
    for gl in gated_loras:
        if not gl.is_hardened():
            gl.harden_gates(threshold)


def get_gated_loras_by_expert(
    gated_loras: List[GatedLoRA]
) -> Dict[Optional[str], List[GatedLoRA]]:
    """
    Partition gated LoRAs by their expert owner.
    
    Returns:
        Dict mapping owner_transformer → list of GatedLoRA.
        Keys: 'transformer_1', 'transformer_2', or None (shared).
    """
    by_expert: Dict[Optional[str], List[GatedLoRA]] = {}
    for gl in gated_loras:
        owner = gl.owner_transformer
        if owner not in by_expert:
            by_expert[owner] = []
        by_expert[owner].append(gl)
    return by_expert


def log_gate_stats(gated_loras: List[GatedLoRA], step: int, prefix: str = "rank_gates"):
    """
    Log statistics about gate states for monitoring.
    """
    if not gated_loras:
        return
    
    total_ranks = sum(gl.r for gl in gated_loras)
    active_ranks = sum(gl.active_ranks() for gl in gated_loras)
    avg_gate = torch.stack([gl.gates.mean() for gl in gated_loras]).mean().item()
    
    # Per-expert stats
    by_expert = get_gated_loras_by_expert(gated_loras)
    
    stats = {
        f"{prefix}/total_ranks": total_ranks,
        f"{prefix}/active_ranks": active_ranks,
        f"{prefix}/active_ratio": active_ranks / max(1, total_ranks),
        f"{prefix}/avg_gate_value": avg_gate,
    }
    
    for owner, glist in by_expert.items():
        if not glist:
            continue
        label = owner or "shared"
        t = sum(gl.r for gl in glist)
        a = sum(gl.active_ranks() for gl in glist)
        stats[f"{prefix}/expert_{label}/total_ranks"] = t
        stats[f"{prefix}/expert_{label}/active_ranks"] = a
        stats[f"{prefix}/expert_{label}/active_ratio"] = a / max(1, t)
    
    # Print summary every 1000 steps
    if step % 1000 == 0:
        print(f"[RankGates step={step}] "
              f"total={total_ranks}, active={active_ranks} "
              f"({100*active_ranks/max(1,total_ranks):.1f}%), "
              f"avg_gate={avg_gate:.3f}")
        for owner, glist in by_expert.items():
            if not glist:
                continue
            label = owner or "shared"
            t = sum(gl.r for gl in glist)
            a = sum(gl.active_ranks() for gl in glist)
            print(f"  Expert {label}: {a}/{t} active "
                  f"({100*a/max(1,t):.1f}%)")
    
    return stats


def truncate_state_dict(
    state_dict: Dict[str, Tensor],
    gated_loras: List,
    threshold: float = 0.5,
    min_rank: int = 1,
) -> Tuple[Dict[str, Tensor], List[Tuple[str, int, int]]]:
    """
    Produce a FULLY-TRUNCATED copy of a network state dict.

    Unlike the normal final save (which only FOLDS the gates, zeroing dead
    components but keeping the original tensor shapes), this physically reduces
    the LoRA rank: dead rows of ``lora_down`` and the matching columns of
    ``lora_up`` are REMOVED, and ``alpha`` is rescaled so the per-rank scaling
    (alpha / rank) is preserved. The result is a genuinely smaller LoRA that any
    standard loader can consume.

    For ``.diff`` (full-finetune) tensors there is no rank axis to shrink; the
    gates are simply folded (dead elements -> 0 diff, i.e. base weight).

    Args:
        state_dict: The network state dict (as returned by ``self.state_dict()``).
        gated_loras: The list of GatedLoRA / GatedDiff modules whose gates drive
                     the truncation. Only modules present in the state dict are
                     affected.
        threshold: Gate value above which a rank is kept.
        min_rank: Minimum ranks to keep per LoRA (at least the single strongest).

    Returns:
        (new_state_dict, summary) where summary is a list of
        (lora_name, old_rank, new_rank) for each truncated LoRA.
    """
    out: Dict[str, Tensor] = {}
    for k, v in state_dict.items():
        # skip gate tensors entirely (they are folded, never saved)
        if '.rank_gate.' in k or '.rank_gate_b.' in k:
            continue
        out[k] = v

    summary: List[Tuple[str, int, int]] = []

    for gl in gated_loras:
        if isinstance(gl, GatedDiff):
            # Dense delta: `gl` IS the GatedDiff (gl.gates are the per-element
            # gates). Fold element-wise (no dimension removal possible for a
            # dense delta; dead elements simply become a 0-diff = base weight).
            key = f"{gl.lora_name}.{gl.key_suffix}"
            if key in out:
                v = out[key].float()
                m = gl.gates.detach().float().view(v.shape)
                out[key] = (v * m).to(state_dict[key].dtype)
            continue

        # GatedLoRA: `gl` IS the GatedLoRA (gl.gates are the per-rank gates).
        # Physically reduce the rank.
        key_A = f"{gl.lora_name}.lora_down.weight"
        key_B = f"{gl.lora_name}.lora_up.weight"
        if key_A not in out or key_B not in out:
            continue

        m = gl.gates.detach().float().view(-1)
        keep = (m > threshold).nonzero(as_tuple=True)[0].tolist()
        if len(keep) < min_rank:
            # keep at least the single strongest rank
            keep = [int(m.argmax().item())]
        k = len(keep)
        r = gl.r

        A = out[key_A].float()
        B = out[key_B].float()

        # Fold gates into A first (surviving ranks keep their gated values,
        # dropped ranks are removed). Rank axis is dim 0 for both linear
        # (r, in) and conv (r, in, k, k).
        m_b = m.view(-1, 1, 1, 1) if A.dim() == 4 else m.view(-1, 1)
        A = A * m_b

        # Slice: A rows (dim 0), B columns (dim 1) for both linear (out, r)
        # and conv (out, r, 1, 1).
        A_new = A[keep].to(state_dict[key_A].dtype)
        B_new = B[:, keep].to(state_dict[key_B].dtype)
        out[key_A] = A_new
        out[key_B] = B_new

        # Rescale alpha so that alpha / rank is preserved.
        key_alpha = f"{gl.lora_name}.alpha"
        if key_alpha in out and r > 0:
            out[key_alpha] = (out[key_alpha].float() * (k / r)).to(
                state_dict[key_alpha].dtype)

        summary.append((gl.lora_name, r, k))

    return out, summary
