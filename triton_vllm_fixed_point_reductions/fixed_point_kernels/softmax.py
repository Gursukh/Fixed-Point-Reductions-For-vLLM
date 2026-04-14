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
):
    """
    Deterministic log-softmax over the last dimension.

    One program per row. Three sequential passes over the vocab dim:
        1. max reduction (associative in a fixed left-to-right loop)
        2. sum(exp(x - max)) accumulated in Q-format fixed point — bitwise
           reproducible because fixed-point sum is order-invariant and
           exp(x - max) ∈ (0, 1] fits in the fractional range.
        3. y = (x - max) - log(sum)
    """
    row = tl.program_id(0)
    x_row = X_ptr + row * stride_xm
    y_row = Y_ptr + row * stride_ym

    # ---- pass 1: max ----
    m = -float("inf")
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))

    # ---- pass 2: fixed-point sum of exp(x - max) ----
    # Accumulate in int64 because V * 2^FRAC_BITS can easily exceed int32
    # (vocab ~150k, sum(exp(x-max)) approaches V → 150k * 65536 ≈ 9.8e9).
    l_fxp = tl.zeros([1], dtype=tl.int64)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.exp(x - m)
        p = tl.where(mask, p, 0.0)
        p_fxp = flp_2_fxp(p, FRAC_BITS, tl.int64)
        l_fxp += tl.sum(p_fxp, axis=0)

    l = fxp_to_flp(l_fxp, FRAC_BITS, tl.float32)
    log_l = tl.log(l)

    # ---- pass 3: write (x - max) - log(sum) ----
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - m) - log_l
        tl.store(y_row + offs, y, mask=mask)


def log_softmax_fxp(
    x: torch.Tensor,
    dim: int = -1,
    frac_bits: int = 16,
    block_n: int = 1024,
) -> torch.Tensor:
    """
    Deterministic fixed-point log-softmax.

    Casts to fp32 internally; returns a tensor in the original dtype.
    """
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
    )

    y = y2d.view(orig_shape).to(orig_dtype)
    if transposed:
        y = y.transpose(dim, -1).contiguous()
    return y
