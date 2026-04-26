import torch
import triton
import triton.language as tl

from .fixed_point import fixed_to_float, float_to_fixed


@triton.jit
def dot_chunk_fxp(a, b, FRAC_BITS: tl.constexpr, FXP_DTYPE: tl.constexpr):
    # test-dot: replace the per-d static_range outer-product loop with tl.dot
    # (hardware MMA). Breaks per-element fixed-point accumulation -> not
    # batch-invariant; used to isolate compile-time cost of the unrolled loop.
    a_h = a.to(tl.float16)
    b_h = b.to(tl.float16)
    outer = tl.dot(a_h, b_h, out_dtype=tl.float32)
    return float_to_fixed(outer, FRAC_BITS, FXP_DTYPE)


@triton.jit
def paged_kv_location(
    page_table,
    stride_page_table_batch,
    batch_index,
    logical_positions,
    mask,
    PAGE_SIZE: tl.constexpr,
):
    """Translate logical token positions into (physical_block, in_block_offset)."""
    # Compute the logical block and in-block offset
    logical_block = logical_positions // PAGE_SIZE
    in_block = logical_positions % PAGE_SIZE

    # Load the physical block from the page table
    physical_block = tl.load(
        page_table + batch_index * stride_page_table_batch + logical_block,
        mask=mask,
        other=0,
    ).to(tl.int64)
    return physical_block, in_block


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
    # Initialize the accumulator
    acc = tl.zeros([ROWS, COLS], dtype=FXP_DTYPE)

    # Main MAC Loop
    for k_start in tl.range(0, Lk, D_CHUNK):
        k_offs = k_start + tl.arange(0, D_CHUNK)
        k_valid = k_offs < Lk

        # Load A chunk [ROWS, D_CHUNK] and B chunk [D_CHUNK, COLS]
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

        # test-dot: hardware MMA in place of the per-d outer-product loop.
        outer = tl.dot(a, b, out_dtype=tl.float32)
        acc += float_to_fixed(outer, FRAC_BITS, FXP_DTYPE)

    # Convert back to float and return to the caller
    return fixed_to_float(acc, FRAC_BITS, tl.float32)


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
    # Grid and Swizzle Logic
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

    # Block Pointers and Masks
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs_m < M
    col_mask = offs_n < N

    a_row_ptrs = a_ptr + offs_m * stride_am
    b_col_ptrs = b_ptr + offs_n * stride_bn

    # Call the device helper function!
    c = gemm_fxp_kernel(
        a_row_ptrs=a_row_ptrs,
        b_col_ptrs=b_col_ptrs,
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

    # Store the resulting block
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def launch_gemm_fxp(
    a: torch.Tensor,
    b: torch.Tensor,
    frac_bits: int = 16,
    fxp_dtype=tl.int32,
) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "launch_gemm_fxp requires CUDA tensors"
    assert a.ndim == 2 and b.ndim == 2, "launch_gemm_fxp expects 2D inputs"
    assert a.shape[1] == b.shape[0], "K dim mismatch between a and b"
    assert a.dtype == b.dtype, f"dtype mismatch: a={a.dtype}, b={b.dtype}"
    assert a.dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ), f"launch_gemm_fxp requires a float input dtype, got {a.dtype}"

    # Ensure inputs are contiguous for strided access in the kernel
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Launch the kernel
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    gemm_fxp[grid](
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
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        FRAC_BITS=frac_bits,
        FXP_DTYPE=fxp_dtype,
    )

    return c
