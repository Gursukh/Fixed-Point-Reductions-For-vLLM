import torch
import triton
import triton.language as tl

from .fixed_point import flp_2_fxp, fxp_to_flp


@triton.jit
def log_softmax_fxp_kernel(
    X_ptr,
    Y_ptr,
    stride_xm,
    stride_ym,
    N,
    FRAC_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = X_ptr + row * stride_xm
    y_row = Y_ptr + row * stride_ym

    m = -float("inf")
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float16)
        m = tl.maximum(m, tl.max(x, axis=0).to(tl.float32))

    l_fxp = tl.zeros([1], dtype=FXP_DTYPE)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float16)
        p = tl.exp(x.to(tl.float32) - m)
        p = tl.where(mask, p, 0.0)
        p_fxp = flp_2_fxp(p, FRAC_BITS, FXP_DTYPE)
        l_fxp += tl.sum(p_fxp, axis=0)

    l = fxp_to_flp(l_fxp, FRAC_BITS, tl.float32)
    log_l = tl.log(l)

    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float16)
        y = (x.to(tl.float32) - m) - log_l
        tl.store(y_row + offs, y, mask=mask)


def log_softmax_fxp(
    x: torch.Tensor,
    fxp_dtype,
    dim: int = -1,
    frac_bits: int = 16,
    block_n: int = 1024,
) -> torch.Tensor:

    assert x.is_cuda, "log_softmax_fxp requires a CUDA tensor"

    orig_dtype = x.dtype
    if dim < 0:
        dim += x.ndim

    if dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()
        transposed = True
    else:
        x = x.contiguous()
        transposed = False

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).to(torch.float32)
    rows, N = x2d.shape

    y2d = torch.empty_like(x2d)
    BLOCK_N = min(triton.next_power_of_2(max(N, 1)), block_n)

    log_softmax_fxp_kernel[(rows,)](
        x2d,
        y2d,
        x2d.stride(0),
        y2d.stride(0),
        N,
        FRAC_BITS=frac_bits,
        BLOCK_N=BLOCK_N,
        FXP_DTYPE=fxp_dtype,
    )

    y = y2d.view(orig_shape).to(orig_dtype)
    if transposed:
        y = y.transpose(dim, -1).contiguous()
    return y
