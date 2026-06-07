import pytest
import torch

import fxpr_vllm  # noqa: F401
from .fixed_point_helpers import requires_cuda


def _run_rms_norm_kernel(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    assert x.ndim == 2 and w.ndim == 1
    assert x.shape[1] == w.shape[0]
    return torch.ops.fxpr.rms_norm_fxp(x, w, eps)


def _run_rms_norm_float_kernel(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    assert x.ndim == 2 and w.ndim == 1
    assert x.shape[1] == w.shape[0]

    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    rrms = 1.0 / torch.sqrt(mean_sq + eps)
    return x * w * rrms


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

    # Stay below Q16.16 saturation.
    x = (
        torch.rand((batch, hidden), device="cuda", dtype=torch.float32, generator=g)
        - 0.5
    ) * 32.0
    w = (
        torch.rand((hidden,), device="cuda", dtype=torch.float32, generator=g) - 0.5
    ) * 4.0

    got = _run_rms_norm_kernel(x, w, eps=1e-6)
    ref = _run_rms_norm_float_kernel(x, w, eps=1e-6)

    assert torch.allclose(got, ref, atol=5e-4, rtol=5e-4)


_DTYPE_TOL = {
    torch.float32: (5e-4, 5e-4),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 16), (4, 64)])
def test_rms_norm_native_dtype(dtype, shape):
    batch, hidden = shape
    g = torch.Generator(device="cuda").manual_seed(0)

    x_f32 = (
        torch.rand((batch, hidden), device="cuda", dtype=torch.float32, generator=g)
        - 0.5
    ) * 4.0
    w_f32 = (
        torch.rand((hidden,), device="cuda", dtype=torch.float32, generator=g) - 0.5
    ) * 2.0

    x = x_f32.to(dtype)
    w = w_f32.to(dtype)

    got = torch.ops.fxpr.rms_norm_fxp(x, w, 1e-6)
    assert got.dtype == dtype, f"output dtype {got.dtype} != input dtype {dtype}"

    # Reference computes in fp32 then casts, same as the kernel.
    ref_f32 = _run_rms_norm_float_kernel(x.to(torch.float32), w.to(torch.float32), 1e-6)
    ref = ref_f32.to(dtype)

    atol, rtol = _DTYPE_TOL[dtype]
    assert torch.allclose(got.to(torch.float32), ref.to(torch.float32),
                          atol=atol, rtol=rtol), (
        f"max error = {(got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()} "
        f"at dtype={dtype}"
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_residual_native_dtype(dtype):
    batch, hidden = 4, 32
    g = torch.Generator(device="cuda").manual_seed(11)

    x_f32 = (
        torch.rand((batch, hidden), device="cuda", dtype=torch.float32, generator=g)
        - 0.5
    ) * 2.0
    r_f32 = (
        torch.rand((batch, hidden), device="cuda", dtype=torch.float32, generator=g)
        - 0.5
    ) * 2.0
    w_f32 = (
        torch.rand((hidden,), device="cuda", dtype=torch.float32, generator=g) - 0.5
    ) * 2.0

    x = x_f32.to(dtype)
    r = r_f32.to(dtype)
    r_orig = r.clone()
    w = w_f32.to(dtype)

    out = torch.ops.fxpr.rms_norm_fxp_residual(x, r, w, 1e-6)
    assert out.dtype == dtype
    assert r.dtype == dtype
    # r is updated in place to (x + r).
    expected_r = (x.to(torch.float32) + r_orig.to(torch.float32)).to(dtype)
    atol, rtol = _DTYPE_TOL[dtype]
    assert torch.allclose(r.to(torch.float32), expected_r.to(torch.float32),
                          atol=atol, rtol=rtol)


@requires_cuda
def test_rms_norm_associativity_fixed_vs_float16():
    # fp16 sum-of-squares is order-dependent here; fxp must not be.
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

    assert torch.equal(y[1], y[0][perm])
