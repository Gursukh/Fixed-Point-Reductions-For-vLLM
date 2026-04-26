import torch
import triton
from torch.library import triton_op, wrap_triton

from .fixed_point_kernels.fixed_point import fixed_tl_dtype
from .fixed_point_kernels.gemm import gemm_fxp as _gemm_fxp_kernel
from .fixed_point_kernels.rms_norm import rms_norm_fxp_kernel as _rms_norm_fxp_kernel

_GEMM_BLOCK_M = 64
_GEMM_BLOCK_N = 64
_GEMM_BLOCK_K = 64


@triton_op("fxpr::gemm_fxp", mutates_args=())
def gemm_fxp(
    a: torch.Tensor,
    b: torch.Tensor,
    frac_bits: int,
    fxp_int_bits: int,
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    fxp_dtype = fixed_tl_dtype(fxp_int_bits)
    grid = (triton.cdiv(M, _GEMM_BLOCK_M) * triton.cdiv(N, _GEMM_BLOCK_N),)
    wrap_triton(_gemm_fxp_kernel)[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_SIZE_M=_GEMM_BLOCK_M,
        BLOCK_SIZE_N=_GEMM_BLOCK_N,
        BLOCK_SIZE_K=_GEMM_BLOCK_K,
        FRAC_BITS=frac_bits,
        FXP_DTYPE=fxp_dtype,
    )
    return c


@gemm_fxp.register_fake
def _(a, b, frac_bits, fxp_int_bits):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)


@triton_op("fxpr::rms_norm_fxp", mutates_args=())
def rms_norm_fxp(
    x: torch.Tensor,
    weight_fp32: torch.Tensor,
    eps: float,
    frac_bits: int,
    fxp_int_bits: int,
) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    batch, hidden = x2d.shape
    y2d = torch.empty_like(x2d)
    block = triton.next_power_of_2(max(hidden, 1))
    fxp_dtype = fixed_tl_dtype(fxp_int_bits)
    wrap_triton(_rms_norm_fxp_kernel)[(batch,)](
        x2d,
        weight_fp32,
        y2d,
        x2d, 
        x2d.stride(0),
        hidden,
        eps=eps,
        BLOCK=block,
        FRAC_BITS=frac_bits,
        FXP_DTYPE=fxp_dtype,
        HAS_RESIDUAL=False,
    )
    return y2d.view_as(x)


@rms_norm_fxp.register_fake
def _(x, weight_fp32, eps, frac_bits, fxp_int_bits):
    return torch.empty_like(x)


@triton_op("fxpr::rms_norm_fxp_residual", mutates_args=("residual",))
def rms_norm_fxp_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight_fp32: torch.Tensor,
    eps: float,
    frac_bits: int,
    fxp_int_bits: int,
) -> torch.Tensor:
    """Fused residual + RMSNorm. residual is overwritten with x + residual (fp32)."""
    assert x.shape == residual.shape, "x and residual must share shape"
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    r2d = residual.reshape(-1, residual.shape[-1])
    if not r2d.is_contiguous():
        r2d = r2d.contiguous()
    batch, hidden = x2d.shape
    y2d = torch.empty_like(x2d)
    block = triton.next_power_of_2(max(hidden, 1))
    fxp_dtype = fixed_tl_dtype(fxp_int_bits)
    wrap_triton(_rms_norm_fxp_kernel)[(batch,)](
        x2d,
        weight_fp32,
        y2d,
        r2d,
        x2d.stride(0),
        hidden,
        eps=eps,
        BLOCK=block,
        FRAC_BITS=frac_bits,
        FXP_DTYPE=fxp_dtype,
        HAS_RESIDUAL=True,
    )
    if r2d.data_ptr() != residual.data_ptr():
        residual.copy_(r2d.view_as(residual))
    return y2d.view_as(x)


@rms_norm_fxp_residual.register_fake
def _(x, residual, weight_fp32, eps, frac_bits, fxp_int_bits):
    return torch.empty_like(x)
