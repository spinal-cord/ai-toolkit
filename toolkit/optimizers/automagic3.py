"""
NOTE: This is experimental and under active development; expect breaking changes and bugs. Feedback welcome.
"""

import math
from typing import List, Optional, Callable, Any
import torch


class Automagic3(torch.optim.Optimizer):
    """
    Automagic v3.

    A single learning rate is kept per param group (typically: one lr for
    the whole run). The control principle: the lr RISES while elements hold
    a decisively consistent update direction at the current step size, FALLS
    while their signs decisively alternate (the overshoot signature: weights
    hopping across a minimum flip sign step to step -- shrinking the step is
    what makes a trajectory reappear at a finer scale), and HOLDS on
    everything in between, which is treated as noise.

    Each element keeps a window of its last H (= ``polarity_history``,
    default 8) update sign bits ("is the update positive", 1-bit packed) --
    H/8 bytes per element (one byte at the default), the only
    per-element optimizer state. A short window suffices because verdicts
    are pooled across the whole group: millions of voters make weak
    common-mode evidence visible long before any single element is
    decisive, and the window length is also the controller's reaction lag
    and warmup.

    Vote rule (per element)
    -----------------------
    Only the two perfectly decisive window states vote; everything else is
    noise:

      up    all H signs agree            +1 * |update|  ("step too small")
      down  all H-1 transitions flip     -1 * |update|  ("step too large":
            (perfect alternation)        the overshoot signature)
      else  any imperfect window          0  (noise)

    The two events are exact mirrors with IDENTICAL pure-noise probability
    (2 of the 2^H possible windows each; ~0.8% per element at H=8), so equal
    weights balance exactly -- no correction factors, no tiers. Per element
    the events are rare, but the verdict is pooled over the whole group
    (millions of elements -> tens of thousands of voters per step even
    under pure noise, mean zero), so the pooled signal is smooth and a real
    trend or real overshoot shifts it decisively. A majority being overshot
    always outvotes a persistent minority, which is what anchors the lr's
    absolute level without external rails. Weighting by |update| lets the
    elements actually moving the weights dominate; an exact-zero update
    records as the negative bit, but such dead/masked elements carry zero
    weight anyway. A tensor abstains entirely until its window has filled
    (the first H steps, and again after a history reset on resume).

    LR MODES
    --------
    Three pooling strategies are available via ``lr_mode``:

    ``per_block`` (default, recommended):
        Votes are pooled per *block*. A block is identified by ``block_fn``,
        which receives ``(param, group_idx)`` and returns a hashable block
        ID. The default ``block_fn`` maps each param group to its own block
        (equivalent to per_group), but a custom function can subdivide a
        group into finer blocks (e.g. per transformer layer). This preserves
        the symmetry cancellation that makes per-group necessary (Q/K
        coupled pairs in the same block cancel their opposing votes) while
        allowing different blocks to learn at different rates -- the key
        improvement over pure per-group, which over-couples unrelated
        modules (embeddings vs. attention vs. LM-head).

    ``per_tensor_soft``:
        Each tensor gets its own adaptive LR, but a soft spring pulls each
        tensor's log-LR toward the group-mean log-LR:
            log(lr_t) <- (1 - lambda) * log(lr_t) + signal_t + lambda * mean_log_lr
        At ``lambda=0`` this is pure per-tensor (diverges on Q/K); at
        ``lambda -> inf`` it collapses to per-group; at ``lambda ~ 0.1`` it
        allows per-tensor adaptation while keeping coupled pairs from
        diverging. This is a strict generalization of per-group.

    ``per_group``:
        The original v3 behavior: one LR per param group, all tensors in
        the group vote into a single pool. Safe but conservative -- it
        sacrifices cross-block / cross-module adaptivity.

    VOTE EMA SMOOTHING
    ------------------
    ``vote_ema_beta`` (default 0.9) applies an EMA to the pooled vote
    signal before converting it to a multiplicative LR factor. This
    prevents the log-space random walk that a pure integral controller
    exhibits on long runs under noisy votes. At ``vote_ema_beta=0`` the
    behavior is identical to the original (no smoothing).

    Parameters
    ----------
    lr : float
        Starting learning rate for every group.
    min_lr : float
        Lower bound on the adapted lr (default 1e-8).
    max_lr : float
        Upper bound on the adapted lr (default 1e3).
    beta2 : float
        EMA decay for the second moment, as in Adam/Adafactor.
    eps : float
        Floor added to the second moment before the rsqrt.
    clip_threshold : float
        Trust region on the update: its RMS is scaled to <= this, then every
        element is clamped to +/- this.
    weight_decay : float
        Decoupled (AdamW-style) weight decay; 0 disables it.
    polarity_history : int
        Sign-history window length H (2 to 64, default 8); H/8 bytes of
        state per element.
    fused : bool
        If True (default), each param is updated inside the backward pass.
    lr_mode : str
        LR pooling strategy: 'per_block' (default), 'per_tensor_soft', or
        'per_group'.
    block_fn : callable, optional
        Function ``(param, group_idx) -> block_id`` for per-block pooling.
        If None, each param group is one block (per_block == per_group).
    coupling_lambda : float
        Soft coupling strength for per_tensor_soft mode (default 0.1).
        Pulls each tensor's log-LR toward the group mean.
    vote_ema_beta : float
        EMA decay for the pooled vote signal (default 0.9). 0 disables
        smoothing (original behavior).
    state_dtype_fp32 : bool
        If True (default), store second-moment EMA state in fp32 for
        numerical stability on long runs. The original code stored in
        p.dtype, which accumulated rounding error in bf16.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-8,
        max_lr: float = 1e3,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        polarity_history: int = 8,
        fused: bool = True,
        lr_mode: str = 'per_block',
        block_fn: Optional[Callable] = None,
        coupling_lambda: float = 0.1,
        vote_ema_beta: float = 0.9,
        state_dtype_fp32: bool = True,
    ):
        if lr_mode not in ('per_group', 'per_block', 'per_tensor_soft'):
            raise ValueError(
                f"lr_mode must be 'per_group', 'per_block', or 'per_tensor_soft', got '{lr_mode}'"
            )
        if coupling_lambda < 0:
            raise ValueError(f"coupling_lambda must be >= 0, got {coupling_lambda}")
        if not (0 <= vote_ema_beta < 1):
            raise ValueError(f"vote_ema_beta must be in [0, 1), got {vote_ema_beta}")
        if min_lr > max_lr:
            raise ValueError(
                f"min_lr ({min_lr}) must be <= max_lr ({max_lr})"
            )
        if lr > 1e-3:
            print(
                f"Note: start lr {lr} is high; the controller will correct it "
                f"(the pooled vote will walk it down)."
            )
        defaults = dict(
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            polarity_history=max(2, min(64, int(polarity_history))),
        )
        super().__init__(params, defaults)

        self.fused = fused
        self.lr_mode = lr_mode
        self.coupling_lambda = coupling_lambda
        self.vote_ema_beta = vote_ema_beta
        self.state_dtype_fp32 = state_dtype_fp32

        if block_fn is not None:
            self._block_fn = block_fn
        else:
            # Default: each param group is one block (per_block == per_group)
            self._block_fn = lambda p, gi: gi

        self._rebuild_group_index()
        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.fused:
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                    self._hook_handles.append(handle)
                elif p.dtype != torch.float32:
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_accum_hook()
                    )
                    self._hook_handles.append(handle)

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(f"Total training parameters: {total:,}")
        print(f"Automagic3 LR mode: {self.lr_mode}")
        if self.lr_mode == 'per_block' and block_fn is not None:
            num_blocks = len(set(self._block_fn(p, gi)
                                 for gi, group in enumerate(self.param_groups)
                                 for p in group["params"]))
            print(f"  Custom block_fn: {num_blocks} blocks identified")
        if self.lr_mode == 'per_tensor_soft':
            print(f"  Coupling lambda: {self.coupling_lambda}")
        if self.vote_ema_beta > 0:
            print(f"  Vote EMA beta: {self.vote_ema_beta}")
        if self.state_dtype_fp32:
            print(f"  Second moment state in fp32")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    @staticmethod
    def _sr_truncate(v_fp32: torch.Tensor, drop_bits: int) -> torch.Tensor:
        as_int = v_fp32.view(torch.int32)
        as_int.add_(torch.randint_like(as_int, 1 << drop_bits))
        as_int.bitwise_and_(-(1 << drop_bits))
        return v_fp32

    @staticmethod
    def _stochastic_round(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        finfo = torch.finfo(dtype)
        absv = v.abs().clamp_(min=finfo.tiny)
        ulp = torch.exp2(torch.floor(torch.log2(absv))).mul_(finfo.eps)
        noise = torch.rand_like(v).sub_(0.5).mul_(ulp)
        return v.add_(noise).to(dtype)

    _PACK_CONSTS: dict = {}

    @classmethod
    def _pack_consts(cls, device):
        consts = cls._PACK_CONSTS.get(device)
        if consts is None:
            consts = (
                torch.tensor(
                    [1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8
                ),
                torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7], device=device, dtype=torch.uint8
                ),
            )
            cls._PACK_CONSTS[device] = consts
        return consts

    @classmethod
    def _pack_bits(cls, bits: torch.Tensor) -> torch.Tensor:
        weights, _ = cls._pack_consts(bits.device)
        flat = bits.reshape(-1).to(torch.uint8)
        pad = (-flat.numel()) % 8
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        return (flat.view(-1, 8) * weights).sum(-1, dtype=torch.uint8)

    @classmethod
    def _unpack_bits(cls, packed: torch.Tensor, numel: int) -> torch.Tensor:
        _, shifts = cls._pack_consts(packed.device)
        vals = (packed.unsqueeze(-1) >> shifts).bitwise_and_(1)
        return vals.view(-1)[:numel]

    def _rebuild_group_index(self) -> None:
        """Build param -> pool index based on lr_mode."""
        self._param_group_index = {
            p: gi for gi, group in enumerate(self.param_groups) for p in group["params"]
        }

        if self.lr_mode == 'per_tensor_soft':
            # Per-tensor: votes stored in per-param state, no pool accumulators
            self._param_pool_index = None
            self._pool_num = None
            self._pool_den = None
            self._pool_vote_ema = None
        else:
            # per_group or per_block: build pool index
            self._param_pool_index = {}
            pool_id_to_idx = {}

            for gi, group in enumerate(self.param_groups):
                for p in group["params"]:
                    if self.lr_mode == 'per_block':
                        pool_id = self._block_fn(p, gi)
                    else:  # per_group
                        pool_id = gi

                    if pool_id not in pool_id_to_idx:
                        pool_id_to_idx[pool_id] = len(pool_id_to_idx)
                    self._param_pool_index[p] = pool_id_to_idx[pool_id]

            num_pools = len(pool_id_to_idx)
            self._pool_num: List = [None] * num_pools
            self._pool_den: List = [None] * num_pools
            self._pool_vote_ema: List = [None] * num_pools

    @classmethod
    def _stochastic_copy_(cls, dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
        if dst.dtype == torch.bfloat16:
            dst.copy_(cls._sr_truncate(src_fp32, 16))
        elif dst.dtype == torch.float16:
            dst.copy_(cls._sr_truncate(src_fp32, 13))
        else:
            dst.copy_(cls._stochastic_round(src_fp32, dst.dtype))

    def _make_accum_hook(self):
        def _hook(p: torch.Tensor):
            if p.grad is None:
                return
            if hasattr(p, "_accum_grad"):
                acc = p._accum_grad.to(torch.float32).add_(p.grad.to(torch.float32))
                self._stochastic_copy_(p._accum_grad, acc)
            else:
                p._accum_grad = p.grad.clone()
            p.grad = None

        return _hook

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.tensor(
            min(max(float(group["lr"]), group["min_lr"]), group["max_lr"]),
            dtype=torch.float32,
            device=p.device,
        )
        H = group["polarity_history"]
        width = (p.numel() + 7) // 8
        state["sign_history"] = torch.zeros(
            (H, width), dtype=torch.uint8, device=p.device
        )
        state["hist_idx"] = 0
        state["hist_fill"] = 0

        # Second moment state: fp32 by default for numerical stability on
        # long runs (the original code stored in p.dtype, which accumulated
        # rounding error in bf16).
        state_dtype = torch.float32 if self.state_dtype_fp32 else p.dtype
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1], dtype=state_dtype, device=p.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:], dtype=state_dtype, device=p.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=state_dtype, device=p.device)

        # Per-tensor vote storage for per_tensor_soft mode
        if self.lr_mode == 'per_tensor_soft':
            state['vote_num'] = None
            state['vote_den'] = None
            if self.vote_ema_beta > 0:
                state['vote_ema'] = 0.0

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)

        return _hook

    # -------------------------------------------------------------- per-param

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic3 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = grad * grad

        # Second moment EMA (Adafactor-factored for >=2D, full for 1D).
        # State is fp32 when state_dtype_fp32=True (default), so no upcast
        # is needed on the hot path.
        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            if row_state.dtype == torch.float32:
                row_state.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
                col_state.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
                update = self._approx_sq_grad(row_state, col_state).mul_(grad)
            else:
                # Low-precision state (state_dtype_fp32=False): upcast for math
                row = row_state.to(torch.float32)
                col = col_state.to(torch.float32)
                row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
                row_state.copy_(row.to(row_state.dtype))
                col_state.copy_(col.to(col_state.dtype))
                update = self._approx_sq_grad(row, col).mul_(grad)
        else:
            v_state = state["exp_avg_sq"]
            if v_state.dtype == torch.float32:
                v_state.mul_(beta2).add_(sq, alpha=1.0 - beta2)
                update = v_state.add(eps).rsqrt().mul_(grad)
            else:
                v = v_state.to(torch.float32)
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
                v_state.copy_(v.to(v_state.dtype))
                update = v.add(eps).rsqrt().mul_(grad)

        # Update-RMS clip (trust region)
        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])

        # --- Direction-consistency vote ---
        cur_bits = update.gt(0.0)
        hist = state["sign_history"]
        idx = state["hist_idx"]
        H = hist.shape[0]
        lr_t = state["lr"]

        hist[idx].copy_(self._pack_bits(cur_bits))
        state["hist_idx"] = (idx + 1) % H
        fill = min(H, state["hist_fill"] + 1)
        state["hist_fill"] = fill

        if fill == H:
            # Compute vote from the packed sign history
            _, shifts = self._pack_consts(hist.device)
            # Avoid torch.roll allocation: use cat with index arithmetic
            chron = torch.cat([hist[idx:], hist[:idx]], dim=0)
            bits = (
                (chron.unsqueeze(-1) >> shifts)
                .bitwise_and_(1)
                .view(H, -1)[:, : update.numel()]
            )
            s1 = bits.sum(0, dtype=torch.int16)
            flips = (bits[1:] ^ bits[:-1]).sum(0, dtype=torch.int16)
            up = s1.eq(H).logical_or_(s1.eq(0))
            down = flips.eq(H - 1)
            w = update.abs().view(-1)
            num = (w * up).sum().sub_((w * down).sum())
            den = w.sum()

            if self.lr_mode == 'per_tensor_soft':
                # Store per-tensor vote; coupling applied in step()
                state['vote_num'] = num
                state['vote_den'] = den
            else:
                # Accumulate into pool (per_group or per_block)
                pi = self._param_pool_index.get(p) if self._param_pool_index is not None else None
                if pi is not None:
                    if self._pool_num[pi] is None:
                        self._pool_num[pi] = num
                        self._pool_den[pi] = den
                    else:
                        acc = self._pool_num[pi]
                        if num.device != acc.device:
                            num = num.to(acc.device)
                            den = den.to(acc.device)
                        acc.add_(num)
                        self._pool_den[pi].add_(den)

        state["step"] += 1

        # --- Apply weight update with current LR ---
        wd = group["weight_decay"]

        if p.dtype == torch.float32:
            if wd != 0.0:
                update.add_(p, alpha=wd)
            p.addcmul_(update, lr_t, value=-1.0)
        else:
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.add_(new_p_fp32, alpha=wd)
            new_p_fp32.addcmul_(update, lr_t, value=-1.0)
            self._stochastic_copy_(p, new_p_fp32)

        p.grad = None

    # ----------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.fused:
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    accum = getattr(p, "_accum_grad", None)
                    if accum is not None:
                        p.grad = accum
                        del p._accum_grad
                    if p.grad is None:
                        continue
                    self._update_param(p, group)
        self._apply_votes()
        return loss

    def _apply_votes(self) -> None:
        """Dispatch to mode-specific vote application."""
        if self.lr_mode == 'per_tensor_soft':
            self._apply_per_tensor_votes()
        else:
            self._apply_pool_votes()

    def _apply_pool_votes(self) -> None:
        """Apply pooled votes for per_group and per_block modes."""
        for pi in range(len(self._pool_num)):
            num = self._pool_num[pi]
            if num is None:
                continue
            den = self._pool_den[pi]
            signal = num.div_(den.clamp_(min=1e-30)).clamp_(-1.0, 1.0)

            # Vote EMA smoothing: prevents log-space random walk on long runs
            if self.vote_ema_beta > 0:
                ema = self._pool_vote_ema[pi]
                if ema is None or ema.device != signal.device:
                    ema = signal.clone()
                else:
                    ema.mul_(self.vote_ema_beta).add_(signal, alpha=1.0 - self.vote_ema_beta)
                self._pool_vote_ema[pi] = ema
                signal = ema

            factor = torch.exp(signal)

            # Apply the same factor to every param in this pool
            for p, pool_idx in self._param_pool_index.items():
                if pool_idx != pi:
                    continue
                st = self.state.get(p)
                if st is None or "lr" not in st:
                    continue
                lr_t = st["lr"]
                f = factor if factor.device == lr_t.device else factor.to(lr_t.device)
                lr_t.mul_(f).clamp_(
                    min=self.param_groups[self._param_group_index[p]]["min_lr"],
                    max=self.param_groups[self._param_group_index[p]]["max_lr"],
                )

            self._pool_num[pi] = None
            self._pool_den[pi] = None

    def _apply_per_tensor_votes(self) -> None:
        """Apply per-tensor votes with soft coupling toward group mean log-LR.

        For each param group:
          1. Compute the mean log-LR across all params (the anchor).
          2. For each param with a vote:
             log_lr_new = (1 - lambda) * log_lr_old + signal + lambda * mean_log_lr
          This allows per-tensor adaptation while keeping coupled pairs
          (e.g. Q/K) from diverging along symmetry directions.
        """
        for gi, group in enumerate(self.param_groups):
            # Collect all LRs in the group for the mean
            all_log_lrs = []
            params_with_votes = []
            for p in group["params"]:
                st = self.state.get(p)
                if st is None or "lr" not in st:
                    continue
                lr_val = float(st["lr"].item())
                all_log_lrs.append(math.log(max(lr_val, 1e-30)))
                if st.get('vote_num') is not None:
                    params_with_votes.append(p)

            if not all_log_lrs:
                continue

            mean_log_lr = sum(all_log_lrs) / len(all_log_lrs)

            for p in params_with_votes:
                st = self.state[p]
                num = st['vote_num']
                den = st['vote_den']
                signal_val = float(
                    (num / den.clamp(min=1e-30)).clamp(-1.0, 1.0).item()
                )

                # Vote EMA smoothing
                if self.vote_ema_beta > 0:
                    prev_ema = st.get('vote_ema', 0.0)
                    st['vote_ema'] = (
                        self.vote_ema_beta * prev_ema
                        + (1.0 - self.vote_ema_beta) * signal_val
                    )
                    signal_val = st['vote_ema']

                # Soft coupling: pull toward group mean log-LR
                log_lr = math.log(max(float(st['lr'].item()), 1e-30))
                log_lr = (
                    (1.0 - self.coupling_lambda) * log_lr
                    + signal_val
                    + self.coupling_lambda * mean_log_lr
                )
                new_lr = math.exp(log_lr)
                new_lr = max(group["min_lr"], min(group["max_lr"], new_lr))
                st['lr'].fill_(new_lr)

                # Clear consumed vote
                st['vote_num'] = None
                st['vote_den'] = None

    def get_learning_rates(self) -> List[float]:
        out = []
        for group in self.param_groups:
            lrs = [
                self.state[p]["lr"]
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(float(torch.stack(lrs).mean()) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        # Hyperparameters are NOT loaded from the checkpoint: constructor args
        # always win, so any setting can be changed mid-run.
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v

        # Mode-specific LR restoration
        if self.lr_mode == 'per_tensor_soft':
            # Preserve individual per-tensor LRs (they are meant to diverge).
            # Just ensure fp32 and clear any stale votes.
            for group in self.param_groups:
                for p in group["params"]:
                    st = self.state.get(p)
                    if st is None:
                        continue
                    if isinstance(st.get("lr"), torch.Tensor):
                        st["lr"] = st["lr"].to(torch.float32)
                    # Clear stale votes from mid-step saves
                    if st.get('vote_num') is not None:
                        st['vote_num'] = None
                    if st.get('vote_den') is not None:
                        st['vote_den'] = None
        elif self.lr_mode == 'per_block':
            # Unify LRs per block (params in the same block should have
            # identical LRs; use geometric median for robustness).
            block_lrs = {}
            for p, pool_idx in self._param_pool_index.items():
                st = self.state.get(p)
                if st is None or not isinstance(st.get("lr"), torch.Tensor):
                    continue
                st["lr"] = st["lr"].to(torch.float32)
                block_lrs.setdefault(pool_idx, []).append(st["lr"])

            for pool_idx, lrs in block_lrs.items():
                if lrs:
                    dev = lrs[0].device
                    med = (
                        torch.stack([t.to(torch.float32).to(dev) for t in lrs])
                        .log_()
                        .median()
                        .exp_()
                    )
                    for t in lrs:
                        t.copy_(med.to(t.device))
        else:
            # per_group: unify LRs per group (original behavior)
            for group in self.param_groups:
                lrs = [
                    st["lr"]
                    for p in group["params"]
                    if (st := self.state.get(p)) is not None
                    and isinstance(st.get("lr"), torch.Tensor)
                ]
                med = None
                if lrs:
                    dev = lrs[0].device
                    med = (
                        torch.stack([t.to(torch.float32).to(dev) for t in lrs])
                        .log_()
                        .median()
                        .exp_()
                    )
                for p in group["params"]:
                    st = self.state.get(p)
                    if st is None:
                        continue
                    if isinstance(st.get("lr"), torch.Tensor):
                        st["lr"] = st["lr"].to(torch.float32)
                        if med is not None:
                            st["lr"].copy_(med.to(st["lr"].device))

        # Sign history: keep when geometry matches, reset otherwise
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                numel = p.numel()
                H = group["polarity_history"]
                width = (numel + 7) // 8
                sh = st.get("sign_history")
                hist_ok = (
                    isinstance(sh, torch.Tensor)
                    and sh.shape == (H, width)
                    and isinstance(st.get("hist_idx"), int)
                    and 0 <= st["hist_idx"] < H
                    and isinstance(st.get("hist_fill"), int)
                    and 0 <= st["hist_fill"] <= H
                )
                if hist_ok:
                    st["sign_history"] = sh.to(torch.uint8)
                else:
                    st["sign_history"] = torch.zeros(
                        (H, width), dtype=torch.uint8, device=p.device
                    )
                    st["hist_idx"] = 0
                    st["hist_fill"] = 0

        # Rebuild pool index (parent replaced group dicts)
        self._rebuild_group_index()