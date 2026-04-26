import triton
import triton.language as tl

from .fixed_point import fixed_to_float, float_to_fixed


@triton.jit
def dot_chunk_fxp(a, b, FRAC_BITS: tl.constexpr, FXP_DTYPE: tl.constexpr):
    # Iterate over the shared dimension (D = a.shape[1]) one element at a time.
    # The old approach (a[:,:,None] * b[None,:,:]) created an [M,D,N] intermediate
    # that overflowed the register file and spilled to DRAM on every call.
    # Here we use a compile-time select via tl.where to avoid the 3-D tensor;
    # LLVM constant-folds the (arange==d) mask into a plain register move/zero.
    D: tl.constexpr = a.shape[1]
    M: tl.constexpr = a.shape[0]
    N: tl.constexpr = b.shape[1]
    acc = tl.zeros([M, N], dtype=FXP_DTYPE)
    for d in tl.static_range(D):
        sel = tl.arange(0, D) == d  # [D] compile-time bool
        a_col = tl.sum(tl.where(sel[None, :], a, 0.0), 1)  # [M]
        b_row = tl.sum(tl.where(sel[:, None], b, 0.0), 0)  # [N]
        outer = a_col[:, None] * b_row[None, :]  # [M, N]
        acc += float_to_fixed(outer, FRAC_BITS, FXP_DTYPE)
    return acc


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
    # Integer accumulator: K-sums are exact across permutations of K because
    # every per-element product is quantised before any addition.
    acc = tl.zeros([ROWS, COLS], dtype=FXP_DTYPE)

    for k_start in tl.range(0, Lk, D_CHUNK):
        k_offs = k_start + tl.arange(0, D_CHUNK)
        k_valid = k_offs < Lk

        a = tl.load(
            a_row_ptrs[:, None] + k_offs[None, :] * stride_a_k,
            mask=row_mask[:, None] & k_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_col_ptrs[None, :] + k_offs[:, None] * stride_b_k,
            mask=k_valid[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += dot_chunk_fxp(a, b, FRAC_BITS, FXP_DTYPE)

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
