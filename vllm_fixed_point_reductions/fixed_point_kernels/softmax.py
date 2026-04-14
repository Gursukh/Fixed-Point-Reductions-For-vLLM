import torch
import triton
import triton.language as tl

from .fixed_point import float_to_fixed, fixed_to_float


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
    row_id = tl.program_id(0)
    x_row_ptr = X_ptr + row_id * stride_xm
    y_row_ptr = Y_ptr + row_id * stride_ym

    row_max = -float("inf")
    for block_start in range(0, N, BLOCK_N):
        col_offs = block_start + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N
        x = tl.load(x_row_ptr + col_offs, mask=col_mask, other=-float("inf")).to(
            tl.float16
        )
        row_max = tl.maximum(row_max, tl.max(x, axis=0).to(tl.float32))

    exp_sum_fxp = tl.zeros([1], dtype=FXP_DTYPE)
    for block_start in range(0, N, BLOCK_N):
        col_offs = block_start + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N
        x = tl.load(x_row_ptr + col_offs, mask=col_mask, other=-float("inf")).to(
            tl.float16
        )
        exp_shifted = tl.exp(x.to(tl.float32) - row_max)
        exp_shifted = tl.where(col_mask, exp_shifted, 0.0)
        exp_shifted_fxp = float_to_fixed(exp_shifted, FRAC_BITS, FXP_DTYPE)
        exp_sum_fxp += tl.sum(exp_shifted_fxp, axis=0)

    exp_sum = fixed_to_float(exp_sum_fxp, FRAC_BITS, tl.float32)
    log_sum = tl.log(exp_sum)

    for block_start in range(0, N, BLOCK_N):
        col_offs = block_start + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N
        x = tl.load(x_row_ptr + col_offs, mask=col_mask, other=0.0).to(tl.float16)
        y = (x.to(tl.float32) - row_max) - log_sum
        tl.store(y_row_ptr + col_offs, y, mask=col_mask)


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
