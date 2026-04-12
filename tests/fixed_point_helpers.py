import pytest
import torch
import triton
import triton.language as tl

from triton_vllm_fixed_point_reductions.fixed_point_kernels.fixed_point import (
    flp_2_fxp,
    fxp_to_flp,
)


@triton.jit
def _flp2fxp_kernel(
    x_ptr, y_ptr, n, frac_bits: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = flp_2_fxp(x, frac_bits, OUT)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _fxp2flp_kernel(
    x_ptr, y_ptr, n, frac_bits: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = fxp_to_flp(x, frac_bits, OUT)
    tl.store(y_ptr + offs, y, mask=mask)


_TL = {
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


def flp2fxp(x: torch.Tensor, frac_bits: tl.constexpr, out: torch.dtype) -> torch.Tensor:
    n = x.numel()
    block = triton.next_power_of_2(max(n, 1))
    y = torch.empty(n, device=x.device, dtype=out)
    _flp2fxp_kernel[(1,)](x.contiguous(), y, n, frac_bits, BLOCK=block, OUT=_TL[out])
    return y.view_as(x)


def fxp2flp(x: torch.Tensor, frac_bits: tl.constexpr, out: torch.dtype) -> torch.Tensor:
    n = x.numel()
    block = triton.next_power_of_2(max(n, 1))
    y = torch.empty(n, device=x.device, dtype=out)
    _fxp2flp_kernel[(1,)](x.contiguous(), y, n, frac_bits, BLOCK=block, OUT=_TL[out])
    return y.view_as(x)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
