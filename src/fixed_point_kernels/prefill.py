import torch
import triton
import triton.language as tl
from src.fixed_point_kernels.fixed_point import (
    fxp_to_flp,
    flp_2_fxp,
    RCP_LN2,
)
from .gemm import gemm_fxp_kernel, dot_chunk_fxp


@triton.jit
def prefill_fxp_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    D_CHUNK: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start = tl.load(B_Start_Loc + cur_batch)
    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_d = offs_d < Lk

    q_row_ptrs = Q + (cur_batch_start + offs_m) * stride_qbs + cur_head * stride_qh
    q_row_mask = offs_m < cur_batch_seq_len

    v_ptrs_base = V + cur_kv_head * stride_vh

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    end_n = cur_batch_seq_len
    end_n = tl.minimum(end_n, (start_m + 1) * BLOCK_M) if IS_CAUSAL else end_n
    end_n_limit = block_mask * end_n

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, end_n_limit, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        pos_k = start_n + offs_n
        col_mask = pos_k < cur_batch_seq_len

        k_col_ptrs = (
            K + (cur_batch_start + pos_k) * stride_kbs + cur_kv_head * stride_kh
        )

        qk = gemm_fxp_kernel(
            q_row_ptrs,
            k_col_ptrs,
            stride_a_k=1,
            stride_b_k=1,
            row_mask=q_row_mask,
            col_mask=col_mask,
            Lk=Lk,
            ROWS=BLOCK_M,
            COLS=BLOCK_N,
            D_CHUNK=D_CHUNK,
            FRAC_BITS=FRAC_BITS,
            FXP_DTYPE=FXP_DTYPE,
        )

        mask = col_mask[None, :]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= pos_k[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_i = tl.maximum(m_i, tl.max(qk, 1))

    l_i_fxp = tl.zeros([BLOCK_M], dtype=FXP_DTYPE)
    acc_fxp = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=FXP_DTYPE)

    for start_n in range(0, end_n_limit, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        pos_k = start_n + offs_n
        col_mask = pos_k < cur_batch_seq_len

        k_col_ptrs = (
            K + (cur_batch_start + pos_k) * stride_kbs + cur_kv_head * stride_kh
        )

        qk = gemm_fxp_kernel(
            q_row_ptrs,
            k_col_ptrs,
            stride_a_k=1,
            stride_b_k=1,
            row_mask=q_row_mask,
            col_mask=col_mask,
            Lk=Lk,
            ROWS=BLOCK_M,
            COLS=BLOCK_N,
            D_CHUNK=D_CHUNK,
            FRAC_BITS=FRAC_BITS,
            FXP_DTYPE=FXP_DTYPE,
        )

        mask = col_mask[None, :]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= pos_k[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        p = tl.math.exp2(qk - m_i[:, None])

        p_fxp = flp_2_fxp(p, FRAC_BITS, FXP_DTYPE)
        l_i_fxp += tl.sum(p_fxp, axis=1)

        v = tl.load(
            v_ptrs_base
            + (cur_batch_start + pos_k[:, None]) * stride_vbs
            + offs_d[None, :],
            mask=col_mask[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float16)

        acc_fxp += dot_chunk_fxp(p, v, FRAC_BITS, FXP_DTYPE)

    l_i = fxp_to_flp(l_i_fxp, FRAC_BITS, tl.float32)
    acc = fxp_to_flp(acc_fxp, FRAC_BITS, tl.float32)
    l_i_safe = tl.maximum(l_i, 1.0e-6)
    acc = acc / l_i_safe[:, None]

    off_o = (
        (cur_batch_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(
        Out + off_o,
        acc,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
    )


@triton.jit
def prefill_fxp_paged_kernel(
    Q,
    K_cache,
    V_cache,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Block_Table,
    Out,
    stride_qbs,
    stride_qh,
    stride_kc_bs,
    stride_kc_bl,
    stride_kc_h,
    stride_vc_bs,
    stride_vc_bl,
    stride_vc_h,
    stride_obs,
    stride_oh,
    stride_bt_b,
    kv_group_num: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    FRAC_BITS: tl.constexpr,
    D_CHUNK: tl.constexpr,
    FXP_DTYPE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_seq_len = tl.load(B_Seqlen + cur_batch)
    q_start = tl.load(B_Start_Loc + cur_batch)
    q_end = tl.load(B_Start_Loc + cur_batch + 1)
    cur_query_len = q_end - q_start
    cur_ctx_len = cur_seq_len - cur_query_len

    block_start_m = BLOCK_M * start_m
    if block_start_m >= cur_query_len:
        return

    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk

    q_abs = cur_ctx_len + offs_m
    q_row_mask = offs_m < cur_query_len

    q_row_ptrs = Q + (q_start + offs_m) * stride_qbs + cur_head * stride_qh

    if IS_CAUSAL:
        end_n = tl.minimum(cur_seq_len, cur_ctx_len + (start_m + 1) * BLOCK_M)
    else:
        end_n = cur_seq_len

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Pass 1: softmax max
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        pos_k = start_n + offs_n
        col_mask = pos_k < end_n

        logical_blk = pos_k // PAGE_SIZE
        in_blk = pos_k % PAGE_SIZE
        phys_blk = tl.load(
            Block_Table + cur_batch * stride_bt_b + logical_blk,
            mask=col_mask,
            other=0,
        ).to(tl.int64)

        k_col_ptrs = (
            K_cache
            + phys_blk * stride_kc_bs
            + in_blk * stride_kc_bl
            + cur_kv_head * stride_kc_h
        )

        qk = gemm_fxp_kernel(
            q_row_ptrs,
            k_col_ptrs,
            stride_a_k=1,
            stride_b_k=1,
            row_mask=q_row_mask,
            col_mask=col_mask,
            Lk=Lk,
            ROWS=BLOCK_M,
            COLS=BLOCK_N,
            D_CHUNK=D_CHUNK,
            FRAC_BITS=FRAC_BITS,
            FXP_DTYPE=FXP_DTYPE,
        )

        mask = col_mask[None, :]
        if IS_CAUSAL:
            mask = mask & (q_abs[:, None] >= pos_k[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_i = tl.maximum(m_i, tl.max(qk, 1))

    l_i_fxp = tl.zeros([BLOCK_M], dtype=FXP_DTYPE)
    acc_fxp = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=FXP_DTYPE)

    # Pass 2: accumulate
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        pos_k = start_n + offs_n
        col_mask = pos_k < end_n

        logical_blk = pos_k // PAGE_SIZE
        in_blk = pos_k % PAGE_SIZE
        phys_blk = tl.load(
            Block_Table + cur_batch * stride_bt_b + logical_blk,
            mask=col_mask,
            other=0,
        ).to(tl.int64)

        k_col_ptrs = (
            K_cache
            + phys_blk * stride_kc_bs
            + in_blk * stride_kc_bl
            + cur_kv_head * stride_kc_h
        )

        qk = gemm_fxp_kernel(
            q_row_ptrs,
            k_col_ptrs,
            stride_a_k=1,
            stride_b_k=1,
            row_mask=q_row_mask,
            col_mask=col_mask,
            Lk=Lk,
            ROWS=BLOCK_M,
            COLS=BLOCK_N,
            D_CHUNK=D_CHUNK,
            FRAC_BITS=FRAC_BITS,
            FXP_DTYPE=FXP_DTYPE,
        )

        mask = col_mask[None, :]
        if IS_CAUSAL:
            mask = mask & (q_abs[:, None] >= pos_k[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        p = tl.math.exp2(qk - m_i[:, None])

        p_fxp = flp_2_fxp(p, FRAC_BITS, FXP_DTYPE)
        l_i_fxp += tl.sum(p_fxp, axis=1)

        v_ptrs = (
            V_cache
            + phys_blk[:, None] * stride_vc_bs
            + in_blk[:, None] * stride_vc_bl
            + cur_kv_head * stride_vc_h
            + offs_d[None, :]
        )
        v = tl.load(
            v_ptrs,
            mask=col_mask[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float16)

        acc_fxp += dot_chunk_fxp(p, v, FRAC_BITS, FXP_DTYPE)

    l_i = fxp_to_flp(l_i_fxp, FRAC_BITS, tl.float32)
    acc = fxp_to_flp(acc_fxp, FRAC_BITS, tl.float32)
    l_i_safe = tl.maximum(l_i, 1.0e-6)
    acc = acc / l_i_safe[:, None]

    off_o = (
        (q_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(
        Out + off_o,
        acc,
        mask=q_row_mask[:, None] & mask_d[None, :],
    )


def context_attention_fwd_fxp_paged(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    o: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_query_len: int,
    fxp_dtype=tl.int32,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    frac_bits: int = 14,
    block_n: int = 16,
    d_chunk: int = 16,
):
    
    Lq, Lk = q.shape[-1], key_cache.shape[-1]
    assert Lq == Lk, "head_dim mismatch between q and key_cache"
    sm_scale = 1.0 / (Lq**0.5) if softmax_scale is None else softmax_scale
    sm_scale *= RCP_LN2

    BLOCK_M = 64
    num_reqs = seq_lens.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    kv_group_num = num_heads // num_kv_heads
    page_size = key_cache.shape[1]

    grid = (num_reqs, num_heads, triton.cdiv(max_query_len, BLOCK_M))

    prefill_fxp_paged_kernel[grid](
        q,
        key_cache,
        value_cache,
        sm_scale,
        query_start_loc,
        seq_lens,
        block_table,
        o,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        o.stride(0),
        o.stride(1),
        block_table.stride(0),
        kv_group_num=kv_group_num,
        PAGE_SIZE=page_size,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        Lk=Lk,
        FRAC_BITS=frac_bits,
        D_CHUNK=d_chunk,
        FXP_DTYPE=fxp_dtype,
        num_warps=4 if Lk <= 64 else 8,
        num_stages=1,
    )


def context_attention_fwd_fxp_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    fxp_dtype=tl.int32,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    frac_bits: int = 14,
    block_n: int = 16,
    d_chunk: int = 16,
):
    Lq, Lk = q.shape[-1], k.shape[-1]
    sm_scale = 1.0 / (Lq**0.5) if softmax_scale is None else softmax_scale
    sm_scale *= RCP_LN2

    BLOCK_M = 64
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))

    prefill_fxp_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        Lk=Lk,
        FRAC_BITS=frac_bits,
        D_CHUNK=d_chunk,
        FXP_DTYPE=fxp_dtype,
        num_warps=4 if Lk <= 64 else 8,
        num_stages=1,
    )
