import pytest
import torch
import triton
import triton.language as tl

from fxpr_vllm.fixed_point_kernels.fixed_point import (
    float_to_fixed as _float_to_fixed_jit,
    fixed_to_float as _fixed_to_float_jit,
)


@triton.jit
def _float_to_fixed_kernel(
    x_ptr, y_ptr, n, frac_bits: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = _float_to_fixed_jit(x, frac_bits, OUT)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _fixed_to_float_kernel(
    x_ptr, y_ptr, n, frac_bits: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = _fixed_to_float_jit(x, frac_bits, OUT)
    tl.store(y_ptr + offs, y, mask=mask)


_TL = {
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


def float_to_fixed(
    x: torch.Tensor, frac_bits: tl.constexpr, out: torch.dtype
) -> torch.Tensor:
    n = x.numel()
    block = triton.next_power_of_2(max(n, 1))
    y = torch.empty(n, device=x.device, dtype=out)
    _float_to_fixed_kernel[(1,)](
        x.contiguous(), y, n, frac_bits, BLOCK=block, OUT=_TL[out]
    )
    return y.view_as(x)


def fixed_to_float(
    x: torch.Tensor, frac_bits: tl.constexpr, out: torch.dtype
) -> torch.Tensor:
    n = x.numel()
    block = triton.next_power_of_2(max(n, 1))
    y = torch.empty(n, device=x.device, dtype=out)
    _fixed_to_float_kernel[(1,)](
        x.contiguous(), y, n, frac_bits, BLOCK=block, OUT=_TL[out]
    )
    return y.view_as(x)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
