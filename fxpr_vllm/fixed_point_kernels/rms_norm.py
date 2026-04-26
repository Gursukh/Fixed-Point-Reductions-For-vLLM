import triton
import triton.language as tl

from fxpr_vllm.fixed_point_kernels.fixed_point import (
    float_to_fixed,
    fixed_to_float,
)


@triton.jit
def rms_norm_fxp_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    Residual_ptr,
    stride_x,
    hidden_size,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """One program per row.  BLOCK >= hidden_size (power-of-two).

    When HAS_RESIDUAL is true, the kernel reads Residual_ptr of the
    same shape as X, computes x = x + residual in fp32, writes the new
    residual back through Residual_ptr, and uses the summed value for the
    norm. The fp32 add is bitwise reproducible per program, so determinism is
    preserved.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden_size

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    if HAS_RESIDUAL:
        r = tl.load(Residual_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        x = x + r
        tl.store(Residual_ptr + row * stride_x + cols, x, mask=mask)

    x_sq = x * x

    x_sq_fxp = float_to_fixed(
        x_sq, fractional_bit_width=FRAC_BITS, fixed_point_type=FXP_DTYPE
    )
    sum_fxp = tl.sum(x_sq_fxp, axis=0)
    sum_float = fixed_to_float(
        sum_fxp, fractional_bit_width=FRAC_BITS, floating_point_type=tl.float32
    )

    mean_sq = tl.maximum(sum_float / hidden_size, 0.0)
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    y = x * w * rrms
    tl.store(Y_ptr + row * stride_x + cols, y, mask=mask)
