import pytest
import torch
import triton
import triton.language as tl

from tests.fixed_point_helpers import requires_cuda
from fxpr_vllm.fixed_point_kernels import rms_norm


@triton.jit
def _rms_norm_float_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride_x,
    hidden_size,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden_size

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    sum_float = tl.sum(x * x, axis=0)
    rrms = 1.0 / tl.sqrt(sum_float / hidden_size + eps)
    y = x * w * rrms
    tl.store(Y_ptr + row * stride_x + cols, y, mask=mask)


def _run_rms_norm_kernel(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    assert x.ndim == 2 and w.ndim == 1
    assert x.shape[1] == w.shape[0]

    batch, hidden = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(max(hidden, 1))

    rms_norm.rms_norm_fxp_kernel[(batch,)](
        x,
        w,
        y,
        x,  # residual buffer (unused; HAS_RESIDUAL=False)
        x.stride(0),
        hidden,
        eps=eps,
        BLOCK=block,
        FRAC_BITS=16,
        FXP_DTYPE=tl.int64,
        HAS_RESIDUAL=False,
    )
    return y


def _run_rms_norm_float_kernel(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    assert x.ndim == 2 and w.ndim == 1
    assert x.shape[1] == w.shape[0]

    batch, hidden = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(max(hidden, 1))

    _rms_norm_float_kernel[(batch,)](
        x,
        w,
        y,
        x.stride(0),
        hidden,
        eps=eps,
        BLOCK=block,
    )
    return y


def _ordered_float16_sum_of_squares(values: list[float]) -> torch.Tensor:
    acc = torch.zeros((), device="cuda", dtype=torch.float16)
    for v in values:
        t = torch.tensor(v, device="cuda", dtype=torch.float16)
        acc = acc + t * t
    return acc


@requires_cuda
@pytest.mark.parametrize("shape", [(2, 7), (3, 16), (4, 31)])
def test_rms_norm_fixed_point_correctness(shape):
    batch, hidden = shape
    g = torch.Generator(device="cuda").manual_seed(0)

    # Keep values comfortably away from per-element fixed-point saturation at Q16.16.
    x = (
        torch.rand((batch, hidden), device="cuda", dtype=torch.float32, generator=g)
        - 0.5
    ) * 32.0
    w = (
        torch.rand((hidden,), device="cuda", dtype=torch.float32, generator=g) - 0.5
    ) * 4.0

    got = _run_rms_norm_kernel(x, w, eps=1e-6)
    ref = _run_rms_norm_float_kernel(x, w, eps=1e-6)

    # Fixed-point sum quantization (Q16.16) should stay close to full-float RMSNorm.
    assert torch.allclose(got, ref, atol=5e-4, rtol=5e-4)


@requires_cuda
def test_rms_norm_associativity_fixed_vs_float16():
    # Squares are [4096, 1, 1, 1, 1], which produce order-dependent fp16 sums:
    # ((4096 + 1) + 1 + 1 + 1) -> 4096, but (1 + 1 + 1 + 1 + 4096) -> 4100.
    order_a = [64.0, 1.0, 1.0, 1.0, 1.0]
    order_b = [1.0, 1.0, 1.0, 1.0, 64.0]

    float16_sum_a = _ordered_float16_sum_of_squares(order_a)
    float16_sum_b = _ordered_float16_sum_of_squares(order_b)
    assert float16_sum_a.item() != float16_sum_b.item()

    x_a = torch.tensor(order_a, device="cuda", dtype=torch.float32)
    perm = torch.tensor([1, 2, 3, 4, 0], device="cuda", dtype=torch.long)
    x_b = x_a[perm]

    x = torch.stack([x_a, x_b], dim=0)
    w = torch.ones((x.shape[1],), device="cuda", dtype=torch.float32)
    y = _run_rms_norm_kernel(x, w, eps=1e-6)

    # With fixed-point sum of squares, the reduction is associative/order-invariant.
    assert torch.equal(y[1], y[0][perm])
