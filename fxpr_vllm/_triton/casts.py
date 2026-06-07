"""Elementwise float to/from fixed-point casts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fxp import fxp_constants, float_to_fixed, fixed_to_float, exp2_saturating


_BLOCK = 1024

# bf16 is taken directly. The kernel upcasts to fp32 in-register before the cvt
# (no bf16-to-int hardware convert exists anyway), and reading bf16 moves half
# the bytes of an fp32 input on this bandwidth-bound op.
_SUPPORTED_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_SUPPORTED_INT_DTYPES = (torch.int16, torch.int32, torch.int64)

_FLOAT_BITS_TO_TL = {16: tl.float16, 32: tl.float32, 64: tl.float64}
_FLOAT_BITS_TO_TORCH = {16: torch.float16, 32: torch.float32, 64: torch.float64}


@triton.jit
def _float_to_fixed_kernel(
    x_ptr,
    y_ptr,
    n,
    SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = float_to_fixed(x, SCALE, QMIN, QMAX, INT_DTYPE)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _fixed_to_float_kernel(
    x_ptr,
    y_ptr,
    n,
    INV_SCALE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y_fp32 = fixed_to_float(x, INV_SCALE)
    tl.store(y_ptr + offsets, y_fp32.to(OUT_DTYPE), mask=mask)


def float_to_fixed_run(
    x: torch.Tensor, int_bits: int, fxp_frac_bits: int
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("float_to_fixed: input must be CUDA")
    if x.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(
            f"float_to_fixed: unsupported input dtype {x.dtype}; "
            f"expected one of {_SUPPORTED_FLOAT_DTYPES}"
        )

    scale, _, qmin, qmax, tl_int_dtype, torch_int_dtype = fxp_constants(
        int_bits, fxp_frac_bits
    )

    x_c = x.contiguous()
    y = torch.empty_like(x_c, dtype=torch_int_dtype)
    n = x_c.numel()
    if n == 0:
        return y

    grid = (triton.cdiv(n, _BLOCK),)
    _float_to_fixed_kernel[grid](
        x_c,
        y,
        n,
        SCALE=scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        BLOCK=_BLOCK,
    )
    return y


def fixed_to_float_run(
    x: torch.Tensor, float_bits: int, fxp_frac_bits: int
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("fixed_to_float: input must be CUDA")
    if x.dtype not in _SUPPORTED_INT_DTYPES:
        raise TypeError(
            f"fixed_to_float: unsupported input dtype {x.dtype}; "
            f"expected one of {_SUPPORTED_INT_DTYPES}"
        )
    if float_bits not in _FLOAT_BITS_TO_TL:
        raise ValueError(f"float_bits must be 16/32/64, got {float_bits}")

    # frac_bits is unbounded for benchmarking (see fxp_constants).
    # exp2_saturating(-N) gives 2**-N without the `1 << N` domain error.
    inv_scale = exp2_saturating(-fxp_frac_bits)
    tl_out_dtype = _FLOAT_BITS_TO_TL[float_bits]
    torch_out_dtype = _FLOAT_BITS_TO_TORCH[float_bits]

    x_c = x.contiguous()
    y = torch.empty_like(x_c, dtype=torch_out_dtype)
    n = x_c.numel()
    if n == 0:
        return y

    grid = (triton.cdiv(n, _BLOCK),)
    _fixed_to_float_kernel[grid](
        x_c,
        y,
        n,
        INV_SCALE=inv_scale,
        OUT_DTYPE=tl_out_dtype,
        BLOCK=_BLOCK,
    )
    return y
