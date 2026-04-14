import torch
import triton
import triton.language as tl

from .fixed_point import (
    flp_2_fxp,
    fxp_to_flp,
)

@triton.jit
def gemm_fxp_kernel(
    a_row_ptrs,
    b_col_ptrs,
    stride_a_k,
    stride_b_k,
    row_mask,
    col_mask,
    Lk,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    D_CHUNK: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    acc = tl.zeros([ROWS, COLS], dtype=FXP_DTYPE)

    for k_start in tl.range(0, Lk, D_CHUNK):
        k_offs = k_start + tl.arange(0, D_CHUNK)
        k_valid = k_offs < Lk

        a = tl.load(
            a_row_ptrs[:, None] + k_offs[None, :] * stride_a_k,
            mask=row_mask[:, None] & k_valid[None, :],
            other=0.0,
        ).to(tl.float16)

        b = tl.load(
            b_col_ptrs[None, :] + k_offs[:, None] * stride_b_k,
            mask=k_valid[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        prod = a[:, :, None] * b[None, :, :]
        acc += tl.sum(flp_2_fxp(prod, FRAC_BITS, FXP_DTYPE), axis=1)

    return fxp_to_flp(acc, FRAC_BITS, tl.float32)


@triton.jit
def dot_chunk_fxp(a, b, FRAC_BITS: tl.constexpr, FXP_DTYPE: tl.constexpr):
    prod = a[:, :, None] * b[None, :, :]
    return tl.sum(flp_2_fxp(prod, FRAC_BITS, FXP_DTYPE), axis=1)


@triton.jit
def gemm_fxp(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):

    GROUP_SIZE_M: tl.constexpr = 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs_m < M
    col_mask = offs_n < N

    a_row_ptrs = a_ptr + offs_m * stride_am
    b_col_ptrs = b_ptr + offs_n * stride_bn

    c = gemm_fxp_kernel(
        a_row_ptrs,
        b_col_ptrs,
        stride_a_k=stride_ak,
        stride_b_k=stride_bk,
        row_mask=row_mask,
        col_mask=col_mask,
        Lk=K,
        ROWS=BLOCK_SIZE_M,
        COLS=BLOCK_SIZE_N,
        D_CHUNK=BLOCK_SIZE_K,
        FRAC_BITS=FRAC_BITS,
        FXP_DTYPE=FXP_DTYPE,
    )

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _gemm_fxp_tiled_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    D_CHUNK: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs_m < M
    col_mask = offs_n < N

    a_row_ptrs = a_ptr + offs_m * stride_am
    b_col_ptrs = b_ptr + offs_n * stride_bn

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=FXP_DTYPE)

    for k_start in tl.range(0, BLOCK_SIZE_K, D_CHUNK):
        k_offs = k_start + tl.arange(0, D_CHUNK)
        k_valid = k_offs < K

        a = tl.load(
            a_row_ptrs[:, None] + k_offs[None, :] * stride_ak,
            mask=row_mask[:, None] & k_valid[None, :],
            other=0.0,
        ).to(tl.float16)

        b = tl.load(
            b_col_ptrs[None, :] + k_offs[:, None] * stride_bk,
            mask=k_valid[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        partial = tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        acc += flp_2_fxp(partial, FRAC_BITS, FXP_DTYPE)

    c = fxp_to_flp(acc, FRAC_BITS, tl.float32)

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def launch_gemm_fxp(
    a: torch.Tensor,
    b: torch.Tensor,
    frac_bits: int,
    fxp_dtype,
) -> torch.Tensor:
    
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]

    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = triton.next_power_of_2(max(K, 1))
    D_CHUNK = 32  

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _gemm_fxp_tiled_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        D_CHUNK=D_CHUNK,
        FRAC_BITS=frac_bits,
        FXP_DTYPE=fxp_dtype,
    )
    return c
