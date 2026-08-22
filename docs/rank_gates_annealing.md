# Rank Gate Annealing (SparseForge-Inspired)

## Overview

Rank gate annealing is a SparseForge-inspired technique that prevents rank collapse during LoRA training by using soft, curvature-aware rank gating instead of hard truncation.

**Status: ENABLED BY DEFAULT** - Rank gates are automatically enabled for all LoRA training jobs. Set `rank_gates.enabled: false` in your config to disable.

> **Note on dual-expert training (Wan 2.2 14B I2V) — full per-expert independence:**
> Each expert gets its OWN annealing controller with a fully separate step
> counter, loss EMAs, LR tracking, phase timeline and Fisher/gradient EMA. The
> experts are trained completely independently and anneal on their own clocks:
> only the active expert's controller advances on a given step, so one expert's
> plateau / LR decay never affects the other. Both experts are finalized
> together at the final save.

## How It Works

## How It Works

### Core Concept

Instead of training all LoRA ranks equally or hard-truncating small singular values:

1. **Soft Gates**: Each rank gets a gate `m_r ∈ [0,1]`, starting at 1 (all active)
2. **Curvature-Aware Scoring**: Ranks are scored using OBD-style importance: `s = F ⊙ w²` (curvature × magnitude²)
3. **Progressive Annealing**: Gates gradually anneal from 1 → {0,1} based on scores
4. **Final Hardening**: At end of training, gates are binarized, selecting surviving ranks

This allows:
- Dead ranks to be "revived" if their score improves (impossible with hard truncation)
- Smooth transitions instead of projection shocks
- Principled decisions based on curvature, not just magnitude

### Fisher Information Tracking

Instead of expensive Hutchinson HVP (used in SparseForge for post-training pruning), we use EMA of squared gradients:

```python
fisher_ema[p] = decay * fisher_ema[p] + (1 - decay) * p.grad.detach() ** 2
```

For diffusion training where gradients are not small mid-training, we use the full Taylor expansion:
```
s = |g_ema ⊙ w| + 0.5 · fisher_ema ⊙ w²
```

### Annealing Schedule

Three coupled schedules from SparseForge:

1. **Target Rank**: `r → k_final` via smooth ramp over the annealing window
2. **Temperature**: Geometric decay `T ← γT` each update (sharpens sigmoid)
3. **Beta Blend**: Smoothstep `0→1` blending soft gate → hard TopK target

Plus binary preference penalty `L_mid = Σ m(1-m)` that ramps up late.

### Learning-Aware Auto Timing (default — no hardcoded steps)

All timing is derived from the **actual training dynamics**, per expert, by the
`LearningAwareScheduler`. There are no fixed `5%`/`75%` step percentages:

| Phase | Trigger (per expert) |
|---|---|
| **Annealing START** | The expert's loss has **plateaued**: fast loss EMA ≈ slow loss EMA (relative change `< plateau_relative_threshold`) for `plateau_confirm_steps` consecutive steps, and after `max(min_anneal_steps, warmup+1)` per-expert steps. |
| **Annealing progress → 1** | Driven by the **LR itself**: progress = how far the LR has decayed from its value at anneal-start down to `end_lr_fraction × peak_lr`. (If the LR is constant, a step clock of `anneal_max_duration` is used instead.) |
| **Hardening START** | The LR has decayed below `hardening_lr_fraction × peak_lr` (learning is essentially finished). For a constant LR, it triggers `hardening_min_steps` before the end. |
| **Hardening progress → 1** | Steps from hardening-start to the per-expert horizon. |

The per-expert horizon is `total_steps / (number of active experts)`.

**Manual override:** set `auto_timing: false` to fall back to a fixed timeline
from `start_step` / `end_step` / `hardening_window` (interpreted as per-expert
steps). This is the legacy behavior, kept for reproducibility.

### Per-Tensor Annealing (SVD-based targets)

Instead of one global `target_rank_ratio` applied to every layer, **every
tensor in the LoRA gets its own final target** `k_final`, computed once at
annealing start (the first post-warmup gate update) from the tensor's
**current** energy spectrum:

| Tensor type | "Rank" axis | Contribution of component *i* | Final target |
|---|---|---|---|
| LoRA pair (`lora_A`/`lora_B`) | SVD rank of `B@A` (factorized SVD) | `S_i² / ΣS²` | `max(1, #{i : S_i²/ΣS² ≥ target_min_rank_contribution})` |
| `.diff` tensor (full finetune) — 1D layer norms, biases, or N-dim conv deltas | Flat element index | `x_i² / Σx²` | `max(1, #{i : x_i²/Σx² ≥ target_min_rank_contribution})` |

This covers **all** tensors in the LoRA, including 1D tensors such as layer
norms that are stored in `.diff` format (not LoRA). For those, the gate is a
per-element vector `m ∈ [0,1]^n` applied as `eff_weight = base + (diff · m) ·
multiplier` inside `FullModule.forward`.

`target_min_rank_contribution` (default **1e-4** = 0.01% of total energy) is
the same threshold used by the offline LoRA statistics tool for its
`Recommended_Rank` column. `target_rank_ratio` remains only as a fallback for
tensors whose budget could not be computed.

### Warmup-Gated Start

With `start_after_warmup: true` (default), the annealing floor is raised to
`warmup_steps + 1` per-expert steps, so the plateau-confirmed start (and the
per-tensor budget computation at the first gate update) is based on stable
(non-ramping) gradients and an already-primed Fisher EMA.

### Truncated Checkpoint ("button")

Set `rank_gates.save_truncated: true` to have **every** save also emit a
fully-truncated variant, e.g. `<name>_truncated.safetensors`. Unlike the normal
final save — which only FOLDS the gates (zeroing dead components but keeping the
original tensor shapes) — the truncated checkpoint **physically reduces the
LoRA rank**: dead rows of `lora_down` and the matching columns of `lora_up` are
removed and `alpha` is rescaled so the per-rank scaling (`alpha / rank`) is
preserved. The result is a genuinely smaller LoRA that any standard loader can
consume. `.diff` tensors are folded as usual (a dense delta has no rank axis to
shrink; dead elements simply become a 0-diff, i.e. the base weight).

`rank_gates.truncation_threshold` (default `0.5`) is the gate value above which
a rank is kept.

## Configuration

### Default Settings (Enabled Automatically — Aggressive)

The current defaults are **aggressive**, tuned for strong pruning and fast convergence:

```yaml
network:
  type: lora
  linear: 128
  # rank_gates is ENABLED BY DEFAULT with these aggressive settings:
  # rank_gates:
  #   enabled: true
  #   auto_timing: true                  # Detect start/end/hardening from loss + LR
  #   target_min_rank_contribution: 1.0e-4  # Per-tensor target: keep components
  #                                         # contributing >= 0.01% of tensor energy
  #   start_after_warmup: true           # Raise the annealing floor past warmup
  #   plateau_relative_threshold: 5.0e-3 # Loss "flat" if <0.5% change over EMA window
  #   plateau_confirm_steps: 50          # Consecutive flat steps to start annealing
  #   min_anneal_steps: 200              # Per-expert floor before annealing may start
  #   end_lr_fraction: 0.2               # Anneal completes when LR < 20% of peak
  #   hardening_lr_fraction: 0.05        # Harden when LR < 5% of peak
  #   target_rank_ratio: 0.3             # Fallback ratio if a per-tensor budget fails
  #   temperature: 1.0
  #   gamma: 0.95                # Faster temperature decay
  #   alpha: 0.1                 # Faster gate updates
  #   lambda_mid_max: 0.01       # Stronger binary preference
  #   update_every: 15           # More frequent updates (per-expert steps)
  #   fisher_decay: 0.999        # Long memory EMA
  #   use_first_order: true
  #   save_truncated: false      # Also emit a rank-reduced checkpoint each save
```

### Conservative vs Aggressive Parameters

The rank gate annealing has several parameters that can be tuned between **conservative** (gentle, exploratory) and **aggressive** (decisive, pruney) settings:

| Parameter | Conservative Range | Aggressive Range | Effect |
|-----------|-------------------|------------------|--------|
| `target_rank_ratio` | 0.6–0.9 (keep more) | 0.2–0.4 (keep less) | Final fraction of ranks retained |
| `lambda_mid_max` | 0.001–0.005 | 0.01–0.05 | Strength of binary preference penalty |
| `alpha` | 0.01–0.05 | 0.1–0.2 | Gate EMA update rate (higher = faster) |
| `gamma` | 0.98–0.999 | 0.93–0.96 | Temperature decay per update (lower = faster sharpening) |
| `update_every` | 30–50 | 10–20 | Steps between gate updates (lower = more frequent) |
| `temperature` | 2.0–5.0 | 0.5–1.0 | Initial sigmoid temperature (higher = softer) |

#### Conservative Example (Gentle Pruning)

Use when:
- Training on a small dataset
- Want to preserve more rank diversity
- Early exploration phase
- Uncertain about final rank needs

```yaml
rank_gates:
  target_rank_ratio: 0.6
  lambda_mid_max: 0.003
  alpha: 0.03
  gamma: 0.98
  update_every: 40
  temperature: 2.0
```

**Behavior**: Keeps 60% of ranks, updates gates infrequently with smooth transitions, softer decisions.

#### Aggressive Example (Strong Pruning)

Use when:
- Training on a large dataset
- Want to eliminate noise-dominated ranks quickly
- Short training runs (< 5000 steps)
- Known that many ranks are redundant

```yaml
rank_gates:
  target_rank_ratio: 0.25
  lambda_mid_max: 0.015
  alpha: 0.15
  gamma: 0.94
  update_every: 12
  temperature: 0.7
```

**Behavior**: Keeps 25% of ranks, updates gates frequently with sharp transitions, decisive pruning.

#### Current Default (Balanced Aggressive)

```yaml
rank_gates:
  target_rank_ratio: 0.3
  lambda_mid_max: 0.01
  alpha: 0.1
  gamma: 0.95
  update_every: 15
  temperature: 1.0
```

**Behavior**: Keeps 30% of ranks, moderate update frequency, balanced sharpening.

### To Disable

```yaml
network:
  type: lora
  linear: 128
  rank_gates:
    enabled: false
```

### To Customize

```yaml
network:
  type: lora
  linear: 256
  rank_gates:
    enabled: true
    target_min_rank_contribution: 1.0e-4  # Min per-component energy contribution
                                           # to keep (per-tensor targets). Raise for
                                           # more aggressive pruning (e.g. 1.0e-3).
    start_after_warmup: true      # Push annealing start past LR warmup
    target_rank_ratio: 0.5        # Fallback only (if a per-tensor budget fails)
    start_step: 1000           # Begin annealing after warmup (auto: 5% of total)
    end_step: 15000            # Complete annealing here (auto: 75% of total)
    temperature: 1.0           # Higher = softer decisions (0.5–5.0)
    gamma: 0.95                # Temperature decay per update (0.93–0.999)
    alpha: 0.1                 # Gate EMA rate (0.01–0.2, higher = faster)
    lambda_mid_max: 0.01       # Binary preference strength (0.001–0.05)
    update_every: 15           # Update gates every N steps (10–50)
    fisher_decay: 0.999        # Fisher EMA decay (keep long memory)
    use_first_order: true      # Include |g·w| term (recommended for diffusion)
    hardening_window: 500      # Final soft→hard interpolation steps
    eta_pen: 0.01              # Mid-preference nudge strength
```

## Dual Expert Training (Wan 2.2 14B)

Rank gates maintain **complete per-expert independence** during dual expert
training. Each expert owns its own `LearningAwareScheduler` **and** its own
`FisherTracker`, so every piece of annealing state is separate:

- **Separate step counters**: each expert's timeline advances only on its own
  per-expert steps (`expert_step_counts`), not the global step.
- **Separate loss EMAs**: plateau detection uses only that expert's loss.
- **Separate LR tracking**: annealing-end / hardening are driven by that
  expert's own learning-rate decay.
- **Separate Fisher/gradient EMAs**: each expert's `FisherTracker` is fed only
  that expert's gated params (the frozen expert's params have no gradients).
- **Separate phase timelines**: one expert can be annealing while the other is
  still heating, or already hardening.
- When training one expert, only that expert's controller advances and gates
  update; the frozen expert is completely untouched.

Per-expert status is logged every 500 steps:
```
[RankGates:transformer_1] step=365 phase=annealing anneal=0.010 harden=0.000 loss_fast=0.30012 lr_ratio=0.706
[RankGates:transformer_2] step=210 phase=heating   anneal=0.000 harden=0.000 loss_fast=0.41200 lr_ratio=0.918
```

## Monitoring

Key metrics to watch (logged to TensorBoard if configured):

- `rank_gates/total_ranks`: Total number of gated ranks
- `rank_gates/active_ranks`: Currently active ranks (gate > 0.5)
- `rank_gates/active_ratio`: Fraction of active ranks
- `rank_gates/avg_gate_value`: Average gate value (should decrease over time)
- `rank_gates/expert_{label}/active_ratio`: Per-expert active ratio
- `loss/L_mid`: Binary preference penalty (should increase late in training)

## Example Config

See `/workspace/ai-toolkit/config/examples/train_lora_wan22_14b_rank_gates.yaml` for a complete example.

## Implementation Details

### Key Design Decisions

1. **Gates are nn.Parameters with requires_grad=False**: Updated by the dedicated SparseForge rule (L_mid is logged, not backpropagated)
2. **Gates NOT in optimizer**: Updated by dedicated SparseForge rule, not optimizer step
3. **Fisher via EMA, not Hutchinson**: Cheaper for mid-training diffusion
4. **Full Taylor expansion**: Handles non-negligible gradients in diffusion training
5. **Per-tensor scoring and targets**: Each tensor scores its own components independently; each tensor's final target `k_final` is computed from its own energy spectrum at annealing start (LoRA pairs: SVD of `B@A`; `.diff` tensors: per-element energy)
6. **`.diff` tensors covered**: 1D layer norms/biases and conv deltas get per-element `GatedDiff` gates applied in `FullModule.forward` (`eff = base + (diff·m)·mult`) and folded into the diff at final save
7. **Learning-aware, loss- and LR-driven timing**: no hardcoded step percentages — annealing starts on a loss plateau, ends on LR decay, and hardening starts on LR decay, all detected per expert by `LearningAwareScheduler`
8. **Complete per-expert independence**: separate controllers, Fisher trackers, step counters and phase timelines per expert (experts anneal on their own clocks)
9. **Warmup-gated floor**: the annealing floor is raised past LR warmup so the first decisions use stable gradients and a primed Fisher EMA
10. **Truncated checkpoint button**: `save_truncated` emits a rank-reduced LoRA (dead rows/cols physically removed, alpha rescaled), not just gate-folded weights

### Files Modified

- `toolkit/rank_gates.py`: Core implementation (GatedLoRA, GatedDiff, FisherTracker, LearningAwareScheduler, QuenchSchedule [legacy], compute_tensor_budgets, update_rank_gates, truncate_state_dict)
- `toolkit/config_modules.py`: RankGateConfig class (auto_timing, plateau/LR thresholds, save_truncated, per-tensor target, warmup floor)
- `toolkit/lora_special.py`: GatedLoRA/GatedDiff creation, FullModule forward/merge gate application
- `toolkit/network_mixins.py`: Gate application in forward pass, gate folding + rank truncation at final save
- `extensions_built_in/sd_trainer/SDTrainer.py`: Per-expert controller/Fisher setup, learning-aware per-expert update loop, truncated-checkpoint save

## References

- SparseForge: Efficient Semi-Structured LLM Sparsification via Annealing of Hessian-Guided Soft-Mask (2026)
- wan_sparse_annealing.md: Project-specific adaptation notes
