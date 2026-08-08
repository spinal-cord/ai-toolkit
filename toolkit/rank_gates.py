"""
SparseForge-inspired rank gate annealing for LoRA training.

This module implements soft, curvature-aware rank gating for LoRA adapters,
allowing gradual elimination of redundant ranks during training instead of
hard truncation. Based on SparseForge (2026) with adaptations for diffusion
training per wan_sparse_annealing.md.

Key concepts:
- Per-rank gates m_r ∈ [0,1] annealed from 1 → {0,1}
- Fisher diagonal via EMA of g² (cheaper than Hutchinson HVP)
- OBD-style scoring: s = F ⊙ w² (curvature × magnitude²)
- Progressive quenching: soft gates → hard TopK → final binarization
- Binary preference penalty L_mid = Σ m(1-m)
- Dual-track: gates updated by their own rule, not autograd from task loss
"""

from typing import List, Optional, Dict, Tuple
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
    
    def target_rank_at(self, step: int, r: int) -> int:
        """
        Target number of active ranks at given step.
        Anneals from r → r * target_rank_ratio over [start_step, end_step].
        """
        p = self._phase(step)
        r_final = max(1, int(round(r * self.target_rank_ratio)))
        return int(round(r - (r - r_final) * p))
    
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


def update_rank_gates(
    gated_loras: List[GatedLoRA],
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
    
    For each GatedLoRA of an active expert:
    1. Compute per-rank OBD score: s = F ⊙ w² (curvature × energy)
    2. Standardize scores layer-wise
    3. Compute soft gate G = sigmoid((s - τ_k) / T) around k-th largest score
    4. Compute hard TopK target Ḡ
    5. Blend: G ← (1-β)·G_soft + β·Ḡ
    6. Apply mid-preference nudge
    7. EMA update: m ← (1-α)·m + α·G
    
    Args:
        gated_loras: List of GatedLoRA modules to update.
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
            
            # Get the underlying LoRA module's weight matrices via plain list refs
            # (NOT registered as nn.Module parameters to avoid state_dict bloat)
            if len(gl._lora_refs) < 2:
                continue  # refs not set up yet
            A = gl._lora_refs[0]  # (r, in_dim) - original nn.Parameter
            B = gl._lora_refs[1]  # (out_dim, r) - original nn.Parameter
            
            # IMPORTANT: score on the ORIGINAL parameters, not copies.
            # FisherTracker keys by id(param), and .to(device) creates a new tensor
            # with a new id. Under layer offloading, A/B may live on CPU-pinned
            # storage while gates are on GPU — scoring the copies would silently
            # fall back to magnitude-only (w**2), defeating the whole purpose.
            gate_device = gl.gates.device
            score_A = fisher_tracker.obd_score(A).sum(dim=1).to(gate_device)  # (r,)
            score_B = fisher_tracker.obd_score(B).sum(dim=0).to(gate_device)  # (r,)
            s = score_A + score_B
            
            # Handle degenerate case
            if s.numel() == 0:
                continue
            
            # Layer-wise standardization (their ˜s)
            # Use unbiased=False to avoid NaN for r=1 layers
            s_std = s.std(unbiased=False)
            s = (s - s.mean()) / (s_std + 1e-8)
            
            # Target k ranks at this step
            k = sched.target_rank_at(step, r)
            
            # Threshold: k-th largest score (0-indexed: index k-1)
            # Fix: use max(k-1, 0) not min(k, r-1) which was off-by-one
            tau = s.sort(descending=True).values[max(k - 1, 0)]
            
            # Temperature at this step.
            # Use per-expert update count if available (for correct decay in dual-expert training).
            owner = gl.owner_transformer or "shared"
            update_count = expert_update_counts.get(owner) if expert_update_counts else None
            T = sched.temperature_at(step, update_count=update_count)
            
            # Soft gate: sigmoid around threshold (their eq. 3)
            G_soft = torch.sigmoid((s - tau) / T)
            
            # Hard structured target: TopK (their Ḡ)
            G_hard = torch.zeros(r, device=s.device, dtype=s.dtype)
            if k > 0:
                G_hard[s.topk(k).indices] = 1.0
            
            # Progressive structure blend: β: 0→1 on smoothstep (their §4.3)
            beta = sched.beta_at(step)
            G = (1 - beta) * G_soft + beta * G_hard
            
            # Binary-preference nudge: gradient of L_mid = Σ m(1-m) w.r.t. target
            # (their eq. 4)
            lam_mid = sched.lambda_mid_at(step)
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
    
    x = sched.hardening_x_at(step)
    
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
