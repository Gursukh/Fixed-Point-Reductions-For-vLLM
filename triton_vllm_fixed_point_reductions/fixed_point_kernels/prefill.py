import torch
import triton
import triton.language as tl
from triton_vllm_fixed_point_reductions.fixed_point_kernels.fixed_point import (
    fxp_to_flp,
    flp_2_fxp,
)
from .gemm import gemm_fxp_kernel, dot_chunk_fxp

RCP_LN2 = 1.4426950408889634


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
            K=BLOCK_DMODEL,
            D_CHUNK=D_CHUNK,
            FRAC_BITS=FRAC_BITS,
        )

        mask = col_mask[None, :]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= pos_k[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_i = tl.maximum(m_i, tl.max(qk, 1))

    l_i_fxp = tl.zeros([BLOCK_M], dtype=tl.int32)
    acc_fxp = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.int32)

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
            K=BLOCK_DMODEL,
            D_CHUNK=D_CHUNK,
            FRAC_BITS=FRAC_BITS,
        )

        mask = col_mask[None, :]
        if IS_CAUSAL:
            mask = mask & (offs_m[:, None] >= pos_k[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        p = tl.math.exp2(qk - m_i[:, None])

        p_fxp = flp_2_fxp(p, FRAC_BITS, tl.int32)
        l_i_fxp += tl.sum(p_fxp, axis=1)

        v = tl.load(
            v_ptrs_base
            + (cur_batch_start + pos_k[:, None]) * stride_vbs
            + offs_d[None, :],
            mask=col_mask[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        acc_fxp += dot_chunk_fxp(p, v, FRAC_BITS)

    l_i = fxp_to_flp(l_i_fxp, FRAC_BITS, tl.float32)
    acc = fxp_to_flp(acc_fxp, FRAC_BITS, tl.float32)
    acc = acc / l_i[:, None]

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


def context_attention_fwd_fxp_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
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
        num_warps=4 if Lk <= 64 else 8,
        num_stages=1,
    )
