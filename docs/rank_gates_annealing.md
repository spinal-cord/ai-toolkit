# Rank Gate Annealing (SparseForge-Inspired)

## Overview

Rank gate annealing is a SparseForge-inspired technique that prevents rank collapse during LoRA training by using soft, curvature-aware rank gating instead of hard truncation.

**Status: ENABLED BY DEFAULT** - Rank gates are automatically enabled for all LoRA training jobs. Set `rank_gates.enabled: false` in your config to disable.

> **Note on dual-expert training (Wan 2.2 14B I2V):**
> The annealing schedule operates on global steps. Each expert's gates are updated
> only when that expert is active. Temperature decay is tracked per-expert to ensure
> correct behavior. Both experts reach their final state by `total_steps`.

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

1. **Target Rank**: `r → r × target_rank_ratio` via cubic ramp over `[start_step, end_step]`
2. **Temperature**: Geometric decay `T ← γT` each update (sharpens sigmoid)
3. **Beta Blend**: Smoothstep `0→1` blending soft gate → hard TopK target

Plus binary preference penalty `L_mid = Σ m(1-m)` that ramps up late.

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
  #   target_rank_ratio: 0.3     # Keep 30% of ranks (prune 70%)
  #   temperature: 1.0
  #   gamma: 0.95                # Faster temperature decay
  #   alpha: 0.1                 # Faster gate updates
  #   lambda_mid_max: 0.01       # Stronger binary preference
  #   update_every: 15           # More frequent updates
  #   fisher_decay: 0.999        # Long memory EMA
  #   use_first_order: true
  #   hardening_window: 500
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
    target_rank_ratio: 0.5     # Keep 50% of ranks (tune between 0.2–0.9)
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

Rank gates maintain **per-expert independence** during dual expert training:

- Each LoRA module's rank gate is tied to its expert (`transformer_1` or `transformer_2`)
- When training one expert, only that expert's gates are updated
- The frozen expert's gates are **completely untouched**: no annealing, no decay, no updates
- This ensures training two independent LoRAs for two independent models
- Fisher EMA tracks gradients per-parameter; frozen expert params have no gradients
- Per-expert statistics are logged every 1000 steps

Example log output (only active expert's gates updated each step):
```
[RankGates Update step=2000] (training transformer_1)
  Expert transformer_1: 1024/2048 active (50.0%), avg_score=0.023 ACTIVE

[RankGates Update step=2025] (training transformer_2)
  Expert transformer_2: 980/2048 active (47.9%), avg_score=-0.012 ACTIVE
```

Frozen expert's gate values remain constant until it becomes the active expert.

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

### Files Modified

- `toolkit/rank_gates.py`: Core implementation (GatedLoRA, FisherTracker, QuenchSchedule, update_rank_gates)
- `toolkit/config_modules.py`: RankGateConfig class, NetworkConfig.rank_gates field
- `toolkit/lora_special.py`: GatedLoRA creation and integration with LoRAModule
- `toolkit/network_mixins.py`: Gate application in forward pass
- `extensions_built_in/sd_trainer/SDTrainer.py`: Training loop integration

### Key Design Decisions

1. **Gates are nn.Parameters with requires_grad=True**: Allows L_mid to affect them via autograd
2. **Gates NOT in optimizer**: Updated by dedicated SparseForge rule, not optimizer step
3. **Fisher via EMA, not Hutchinson**: Cheaper for mid-training diffusion
4. **Full Taylor expansion**: Handles non-negligible gradients in diffusion training
5. **Per-module scoring**: Each LoRA module scores its own ranks independently

## References

- SparseForge: Efficient Semi-Structured LLM Sparsification via Annealing of Hessian-Guided Soft-Mask (2026)
- wan_sparse_annealing.md: Project-specific adaptation notes
