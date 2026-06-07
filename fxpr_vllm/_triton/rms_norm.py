"""Batch-invariant RMSNorm: one program per row, chunked over hidden.

Grid is (batch,) and never specializes on batch size, so each row reduces in a
fixed order. Sum stays fp32; fixed-point x*x overflowed on real activations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


_BLOCK_N = 1024
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    r_ptr,
    stride_x,
    hidden,
    eps,
    IO_DTYPE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_x
    if HAS_RESIDUAL:
        r_row = r_ptr + row * stride_x

    # Sum of squares in fp32. One program owns the row, so the reduction order
    # is fixed and the result is bit-identical regardless of batch.
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, hidden, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < hidden
        xi = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            ri = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)
            xi = xi + ri
            tl.store(r_row + cols, xi.to(IO_DTYPE), mask=mask)
        sq = xi * xi
        acc += tl.sum(sq, axis=0)

    mean_sq = tl.maximum(acc / hidden.to(tl.float32), 0.0)
    rrms = libdevice.rsqrt(mean_sq + eps)

    for off in range(0, hidden, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < hidden
        if HAS_RESIDUAL:
            xi = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            xi = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        wi = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        yi = xi * wi * rrms
        tl.store(y_row + cols, yi.to(IO_DTYPE), mask=mask)


_TORCH_TO_TL_FLOAT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _common_launch(
    x: torch.Tensor,
    w: torch.Tensor,
    residual: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if not x.is_cuda or not w.is_cuda:
        raise RuntimeError("rms_norm: x and w must be CUDA")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"rms_norm: unsupported input dtype {x.dtype}; "
            f"expected one of {_SUPPORTED_DTYPES}"
        )
    if w.dtype != x.dtype:
        raise TypeError("rms_norm: weight dtype must match input dtype")
    if residual is not None:
        if not residual.is_cuda:
            raise RuntimeError("rms_norm: residual must be CUDA")
        if residual.dtype != x.dtype:
            raise TypeError("rms_norm: residual dtype must match input dtype")
        if residual.shape != x.shape:
            raise ValueError("rms_norm: residual shape must match x")

    x_2d = _as_2d(x).contiguous()
    y_2d = torch.empty_like(x_2d)
    if residual is not None:
        r_2d = _as_2d(residual)
        r_storage = r_2d if r_2d.is_contiguous() else r_2d.contiguous()
    else:
        r_storage = None

    batch, hidden = x_2d.shape
    if batch == 0 or hidden == 0:
        if residual is not None and r_storage.data_ptr() != residual.data_ptr():
            residual.copy_(r_storage.view_as(residual))
        return y_2d.view_as(x)

    io_dtype = _TORCH_TO_TL_FLOAT[x.dtype]

    grid = (batch,)
    _rms_norm_kernel[grid](
        x_2d,
        w,
        y_2d,
        # Triton needs a real tensor here; it is only read when HAS_RESIDUAL.
        r_storage if r_storage is not None else x_2d,
        x_2d.stride(0),
        hidden,
        float(eps),
        IO_DTYPE=io_dtype,
        HAS_RESIDUAL=residual is not None,
        BLOCK_N=_BLOCK_N,
    )

    if residual is not None and r_storage.data_ptr() != residual.data_ptr():
        residual.copy_(r_storage.view_as(residual))
    return y_2d.view_as(x)


def rms_norm_fxp_run(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _common_launch(x, w, None, eps)


def rms_norm_fxp_residual_run(
    x: torch.Tensor,
    residual: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _common_launch(x, w, residual, eps)
