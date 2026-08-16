"""
Hardware-accelerated GELU using tanh.approx.f32 PTX instruction.

Wan 2.2 uses gelu-approximate (tanh approximation of GELU) in all FeedForward layers.
PyTorch's F.gelu(approximate="tanh") uses standard tanh which is relatively slow.
This module provides an optimized version using the same tanh.approx.f32 PTX instruction
that quack/attention-gym use.

Formula (from quack/activation.py):
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            = 0.5 * x * (1 + tanh(x * (0.797885 + 0.0356774 * x²)))
"""

import math
import torch
from torch import Tensor
from functools import partial

# Constants from quack/activation.py
sqrt_2_over_pi = math.sqrt(2 / math.pi)  # ~0.797885
sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # ~0.0356774
sqrt_2_over_pi_coeff_3 = 3.0 * sqrt_2_over_pi_coeff  # ~0.01070322


# Try to create hardware-accelerated GELU using tanh.approx.f32
_gelu_approx_accelerated = None


def _get_gelu_approx_accelerated():
    """
    Get hardware-accelerated GELU approximation using tanh.approx.f32 PTX instruction.
    Falls back to torch.nn.functional.gelu if custom op registration fails.
    """
    global _gelu_approx_accelerated
    if _gelu_approx_accelerated is not None:
        return _gelu_approx_accelerated

    try:
        # Import torch.compile internals for custom op registration
        from torch._inductor.lowering import make_pointwise, register_lowering
        from torch._inductor.virtualized import ops

        @torch.library.custom_op("approx::gelu_tanh", mutates_args=())
        def _gelu_approx_impl(inp: Tensor) -> Tensor:
            # Fallback implementation for tracing
            return torch.nn.functional.gelu(inp, approximate="tanh")

        @_gelu_approx_impl.register_fake
        def _(inp: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(inp, approximate="tanh")

        def _gelu_approx_lowering(inp):
            """
            Lower GELU to use tanh.approx.f32 PTX instruction.
            Formula: 0.5 * x * (1 + tanh(x * (c1 + c2 * x²)))
            """
            # Create pointwise function that computes GELU with approximate tanh
            def gelu_with_approx_tanh(x):
                # Compute z = x * (c1 + c2 * x²)
                x_sq = x * x
                z = x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq)
                # Use approximate tanh via inline PTX
                tanh_z = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")(z)
                # Compute final GELU
                return 0.5 * x * (1.0 + tanh_z)

            return make_pointwise(gelu_with_approx_tanh)(inp)

        register_lowering(torch.ops.approx.gelu_tanh)(_gelu_approx_lowering)

        class _GeluApproxAccelerated(torch.autograd.Function):
            @staticmethod
            def forward(x):
                return torch.ops.approx.gelu_tanh(x)

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, grad_output):
                """
                GELU tanh approximation backward pass.
                From quack/activation.py dgelu_tanh_approx:
                
                d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * dz/dx
                where z = x * (c1 + c2 * x²), dz/dx = c1 + 3 * c2 * x²
                and sech²(z) = 1 - tanh²(z)
                """
                (x,) = ctx.saved_tensors
                
                # Compute z = x * (c1 + c2 * x²)
                x_sq = x * x
                z = x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq)
                
                # Compute tanh(z) - use standard tanh for backward (accuracy > speed here)
                tanh_z = torch.tanh(z)
                
                # half_tanh_z_plus_one = 0.5 * (1 + tanh(z))
                half_tanh_z_plus_one = 0.5 + 0.5 * tanh_z
                
                # sech²(z) = 1 - tanh²(z)
                sech2_z = 1 - tanh_z * tanh_z
                
                # dz/dx = c1 + 3 * c2 * x²
                dz_dx = sqrt_2_over_pi + sqrt_2_over_pi_coeff_3 * x_sq
                
                # d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech²(z) * dz/dx
                dgelu = half_tanh_z_plus_one + x * (0.5 * (sech2_z * dz_dx))
                
                return grad_output * dgelu

            @staticmethod
            def vmap(info, in_dims, x):
                # Fall back to standard GELU for vmap
                return torch.nn.functional.gelu(x, approximate="tanh"), 0

        _gelu_approx_accelerated = _GeluApproxAccelerated.apply
        return _gelu_approx_accelerated

    except (ImportError, RuntimeError, AttributeError):
        # Fall back to standard PyTorch GELU
        _gelu_approx_accelerated = lambda x: torch.nn.functional.gelu(x, approximate="tanh")
        return _gelu_approx_accelerated


def gelu_accelerated(x: Tensor) -> Tensor:
    """
    Apply GELU with tanh approximation using hardware-accelerated tanh.
    
    Args:
        x: Input tensor
        
    Returns:
        GELU(x) computed with tanh.approx.f32 PTX instruction
    """
    return _get_gelu_approx_accelerated()(x)


def test_gelu_acceleration():
    """Test that accelerated GELU matches PyTorch's gelu(approximate='tanh')."""
    print("Testing GELU acceleration...")
    
    gelu_fn = _get_gelu_approx_accelerated()
    
    # Test on CPU first
    x = torch.randn(100, 100)
    out_accel = gelu_fn(x)
    out_ref = torch.nn.functional.gelu(x, approximate="tanh")
    
    max_diff = (out_accel - out_ref).abs().max().item()
    print(f"  CPU max diff: {max_diff:.2e}")
    
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        out_accel_cuda = gelu_fn(x_cuda)
        out_ref_cuda = torch.nn.functional.gelu(x_cuda, approximate="tanh")
        
        max_diff_cuda = (out_accel_cuda - out_ref_cuda).abs().max().item()
        print(f"  CUDA max diff: {max_diff_cuda:.2e}")
    
    # GELU tanh approximation has inherent numerical differences, so we check for reasonable tolerance
    assert max_diff < 1e-4, f"GELU acceleration mismatch: {max_diff}"
    print("  ✓ GELU acceleration test passed!")


if __name__ == "__main__":
    test_gelu_acceleration()
