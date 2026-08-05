"""
Test that EMA correctly isolates per-expert updates for Wan 2.2 multistage models.

Verifies that when only one expert is active, the other expert's EMA shadow
parameters remain completely frozen (no decay drift).
"""
import torch
import math
from toolkit.ema import ExponentialMovingAverage

def approx_eq(a, b, tol=1e-5):
    return math.isclose(a, b, abs_tol=tol)


def test_ema_expert_isolation():
    """Test that inactive expert's EMA is frozen while active expert's EMA updates."""
    # Create 6 params: 2 for expert1, 2 for expert2, 2 shared
    p_e1_1 = torch.nn.Parameter(torch.tensor(1.0))
    p_e1_2 = torch.nn.Parameter(torch.tensor(2.0))
    p_e2_1 = torch.nn.Parameter(torch.tensor(10.0))
    p_e2_2 = torch.nn.Parameter(torch.tensor(20.0))
    p_shared_1 = torch.nn.Parameter(torch.tensor(100.0))
    p_shared_2 = torch.nn.Parameter(torch.tensor(200.0))

    # Expert map passed via constructor (no Parameter attributes modified)
    expert_map = [
        "transformer_1", "transformer_1",  # p_e1_1, p_e1_2
        "transformer_2", "transformer_2",  # p_e2_1, p_e2_2
        None, None,                        # p_shared_1, p_shared_2
    ]
    params = [p_e1_1, p_e1_2, p_e2_1, p_e2_2, p_shared_1, p_shared_2]
    ema = ExponentialMovingAverage(params, decay=0.5, expert_map=expert_map)

    # Initial shadow params match the real params
    assert approx_eq(ema.shadow_params[0].item(), 1.0)
    assert approx_eq(ema.shadow_params[1].item(), 2.0)
    assert approx_eq(ema.shadow_params[2].item(), 10.0)
    assert approx_eq(ema.shadow_params[3].item(), 20.0)

    # --- Step 1: expert 1 is active ---
    # expert 1 params change, expert 2 params stay static
    with torch.no_grad():
        p_e1_1.fill_(1.1)
        p_e1_2.fill_(2.1)
        # p_e2_1, p_e2_2 unchanged

    ema.update(active_experts={"transformer_1"})

    # Expert 1's EMA should have updated (tracking the changed values)
    # shadow_new = shadow_old * 0.5 + param_new * 0.5
    assert approx_eq(ema.shadow_params[0].item(), 1.05), f"Expected 1.05, got {ema.shadow_params[0].item()}"
    assert approx_eq(ema.shadow_params[1].item(), 2.05), f"Expected 2.05, got {ema.shadow_params[1].item()}"

    # Expert 2's EMA should be COMPLETELY FROZEN (not even decay drift!)
    assert ema.shadow_params[2].item() == 10.0, "Expert 2 EMA should be frozen!"
    assert ema.shadow_params[3].item() == 20.0, "Expert 2 EMA should be frozen!"

    # Shared params always update
    assert approx_eq(ema.shadow_params[4].item(), 100.0)
    assert approx_eq(ema.shadow_params[5].item(), 200.0)

    # --- Step 2: expert 2 is active ---
    with torch.no_grad():
        p_e2_1.fill_(10.5)
        p_e2_2.fill_(20.5)
        # p_e1_1, p_e1_2 unchanged from step 1

    ema.update(active_experts={"transformer_2"})

    # Expert 2's EMA should update
    # shadow_new = shadow_old * 0.5 + param_new * 0.5 = 10.0 * 0.5 + 10.5 * 0.5 = 10.25
    assert approx_eq(ema.shadow_params[2].item(), 10.25)
    assert approx_eq(ema.shadow_params[3].item(), 20.25)

    # Expert 1's EMA should be frozen (still at the values from step 1)
    assert approx_eq(ema.shadow_params[0].item(), 1.05), "Expert 1 EMA should be frozen!"
    assert approx_eq(ema.shadow_params[1].item(), 2.05), "Expert 1 EMA should be frozen!"

    print("✅ All EMA expert isolation tests passed!")


def test_ema_backward_compat():
    """Test that EMA without expert_map/active_experts updates all params (backward compat)."""
    p1 = torch.nn.Parameter(torch.tensor(1.0))
    p2 = torch.nn.Parameter(torch.tensor(2.0))

    ema = ExponentialMovingAverage([p1, p2], decay=0.5)  # no expert_map

    with torch.no_grad():
        p1.fill_(1.1)
        p2.fill_(2.1)

    # No active_experts arg → update all
    ema.update()

    assert approx_eq(ema.shadow_params[0].item(), 1.05)
    assert approx_eq(ema.shadow_params[1].item(), 2.05)

    print("✅ Backward compatibility test passed!")


if __name__ == "__main__":
    test_ema_expert_isolation()
    test_ema_backward_compat()
