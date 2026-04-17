import torch
import triton
import triton.language as tl

from fxpr_vllm.fixed_point_kernels.fixed_point import (
    fixed_to_float,
    float_to_fixed,
    RCP_LN2,
)
from .gemm import dot_chunk_fxp, paged_kv_location


def prepare_log2_softmax_scale(head_dim, softmax_scale=None):
    scale = 1.0 / (head_dim**0.5) if softmax_scale is None else softmax_scale
    return scale * RCP_LN2


@triton.jit
def _attention_pass(
    query_row_pointers,
    query_row_mask,
    causal_row_positions,
    key_end_position,
    head_dim_offsets,
    head_dim_mask,
    kv_head_index,
    head_index,
    softmax_scale,
    running_row_max,
    softmax_denominator_fxp,
    output_accumulator_fxp,
    alibi_slopes_ptr,
    PASS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
    KEY_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_CHUNK: tl.constexpr,
    FRACTIONAL_BITS: tl.constexpr,
    FIXED_POINT_DTYPE: tl.constexpr,
    IS_PAGED: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
    K=0,
    stride_key_seq=0,
    stride_key_head=0,
    V=0,
    stride_value_seq=0,
    stride_value_head=0,
    batch_token_start=0,
    K_cache=0,
    stride_key_cache_block=0,
    stride_key_cache_slot=0,
    stride_key_cache_head=0,
    V_cache=0,
    stride_value_cache_block=0,
    stride_value_cache_slot=0,
    stride_value_cache_head=0,
    block_table=0,
    stride_block_table_batch=0,
    request_index=0,
):
    key_offsets = tl.arange(0, KEY_BLOCK_SIZE)

    # Alibi slope is log2-scaled at launch time, so bias can be added
    # directly to softmax-scale-multiplied qk (also in log2 domain).
    if USE_ALIBI:
        alibi_slope = tl.load(alibi_slopes_ptr + head_index)
    else:
        alibi_slope = 0.0

    for key_block_start in range(0, key_end_position, KEY_BLOCK_SIZE):
        key_block_start = tl.multiple_of(key_block_start, KEY_BLOCK_SIZE)
        key_positions = key_block_start + key_offsets
        key_mask = key_positions < key_end_position

        if IS_PAGED:
            physical_block_index, in_block_offset = paged_kv_location(
                block_table,
                stride_block_table_batch,
                request_index,
                key_positions,
                key_mask,
                PAGE_SIZE,
            )
            key_column_pointers = (
                K_cache
                + physical_block_index * stride_key_cache_block
                + in_block_offset * stride_key_cache_slot
                + kv_head_index * stride_key_cache_head
            )
        else:
            key_column_pointers = (
                K
                + (batch_token_start + key_positions) * stride_key_seq
                + kv_head_index * stride_key_head
            )

        qk_fxp = tl.zeros([QUERY_BLOCK_SIZE, KEY_BLOCK_SIZE], dtype=FIXED_POINT_DTYPE)
        for head_dim_chunk_start in tl.range(0, HEAD_DIM, HEAD_DIM_CHUNK):
            head_dim_chunk_offsets = head_dim_chunk_start + tl.arange(0, HEAD_DIM_CHUNK)
            head_dim_chunk_valid = head_dim_chunk_offsets < HEAD_DIM

            query_chunk = tl.load(
                query_row_pointers[:, None] + head_dim_chunk_offsets[None, :],
                mask=query_row_mask[:, None] & head_dim_chunk_valid[None, :],
                other=0.0,
            ).to(tl.float16)
            key_chunk = tl.load(
                key_column_pointers[None, :] + head_dim_chunk_offsets[:, None],
                mask=head_dim_chunk_valid[:, None] & key_mask[None, :],
                other=0.0,
            ).to(tl.float16)

            qk_fxp += dot_chunk_fxp(
                a=query_chunk,
                b=key_chunk,
                FRAC_BITS=FRACTIONAL_BITS,
                FXP_DTYPE=FIXED_POINT_DTYPE,
            )

        qk = fixed_to_float(
            x=qk_fxp,
            fractional_bit_width=FRACTIONAL_BITS,
            floating_point_type=tl.float32,
        )
        qk = qk * softmax_scale
        if USE_ALIBI:
            qk += alibi_slope * (
                key_positions[None, :].to(tl.float32)
                - causal_row_positions[:, None].to(tl.float32)
            )

        score_mask = key_mask[None, :]
        if IS_CAUSAL:
            score_mask = score_mask & (
                causal_row_positions[:, None] >= key_positions[None, :]
            )
        qk = tl.where(score_mask, qk, float("-inf"))

        if PASS == 1:
            running_row_max = tl.maximum(running_row_max, tl.max(qk, 1))
        else:
            if IS_PAGED:
                value_pointers = (
                    V_cache
                    + physical_block_index[:, None] * stride_value_cache_block
                    + in_block_offset[:, None] * stride_value_cache_slot
                    + kv_head_index * stride_value_cache_head
                    + head_dim_offsets[None, :]
                )
            else:
                value_pointers = (
                    V
                    + kv_head_index * stride_value_head
                    + (batch_token_start + key_positions[:, None]) * stride_value_seq
                    + head_dim_offsets[None, :]
                )

            attention_weights = tl.math.exp2(qk - running_row_max[:, None])
            attention_weights_fxp = float_to_fixed(
                x=attention_weights,
                fractional_bit_width=FRACTIONAL_BITS,
                fixed_point_type=FIXED_POINT_DTYPE,
            )
            softmax_denominator_fxp += tl.sum(attention_weights_fxp, axis=1)

            value_chunk = tl.load(
                value_pointers,
                mask=key_mask[:, None] & head_dim_mask[None, :],
                other=0.0,
            ).to(tl.float16)
            output_accumulator_fxp += dot_chunk_fxp(
                a=attention_weights,
                b=value_chunk,
                FRAC_BITS=FRACTIONAL_BITS,
                FXP_DTYPE=FIXED_POINT_DTYPE,
            )

    return running_row_max, softmax_denominator_fxp, output_accumulator_fxp


@triton.jit
def attention_fwd_fxp_body(
    output,
    softmax_scale,
    query_row_pointers,
    query_row_mask,
    causal_row_positions,
    output_row_base,
    key_end_position,
    query_offsets,
    head_dim_offsets,
    head_dim_mask,
    kv_head_index,
    head_index,
    stride_output_seq,
    stride_output_head,
    alibi_slopes_ptr,
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
    KEY_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_CHUNK: tl.constexpr,
    FRACTIONAL_BITS: tl.constexpr,
    FIXED_POINT_DTYPE: tl.constexpr,
    IS_PAGED: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
    K=0,
    stride_key_seq=0,
    stride_key_head=0,
    V=0,
    stride_value_seq=0,
    stride_value_head=0,
    batch_token_start=0,
    K_cache=0,
    stride_key_cache_block=0,
    stride_key_cache_slot=0,
    stride_key_cache_head=0,
    V_cache=0,
    stride_value_cache_block=0,
    stride_value_cache_slot=0,
    stride_value_cache_head=0,
    block_table=0,
    stride_block_table_batch=0,
    request_index=0,
):
    running_row_max = tl.zeros([QUERY_BLOCK_SIZE], dtype=tl.float32) - float("inf")
    softmax_denominator_fxp = tl.zeros([QUERY_BLOCK_SIZE], dtype=FIXED_POINT_DTYPE)
    output_accumulator_fxp = tl.zeros(
        [QUERY_BLOCK_SIZE, HEAD_DIM_PADDED], dtype=FIXED_POINT_DTYPE
    )

    # Pass 1: softmax max. Kept separate from pass 2 to avoid the
    # nondeterminism of online softmax.
    running_row_max, softmax_denominator_fxp, output_accumulator_fxp = _attention_pass(
        query_row_pointers=query_row_pointers,
        query_row_mask=query_row_mask,
        causal_row_positions=causal_row_positions,
        key_end_position=key_end_position,
        head_dim_offsets=head_dim_offsets,
        head_dim_mask=head_dim_mask,
        kv_head_index=kv_head_index,
        head_index=head_index,
        softmax_scale=softmax_scale,
        running_row_max=running_row_max,
        softmax_denominator_fxp=softmax_denominator_fxp,
        output_accumulator_fxp=output_accumulator_fxp,
        alibi_slopes_ptr=alibi_slopes_ptr,
        PASS=1,
        IS_CAUSAL=IS_CAUSAL,
        USE_ALIBI=USE_ALIBI,
        QUERY_BLOCK_SIZE=QUERY_BLOCK_SIZE,
        KEY_BLOCK_SIZE=KEY_BLOCK_SIZE,
        HEAD_DIM=HEAD_DIM,
        HEAD_DIM_CHUNK=HEAD_DIM_CHUNK,
        FRACTIONAL_BITS=FRACTIONAL_BITS,
        FIXED_POINT_DTYPE=FIXED_POINT_DTYPE,
        IS_PAGED=IS_PAGED,
        PAGE_SIZE=PAGE_SIZE,
        K=K,
        stride_key_seq=stride_key_seq,
        stride_key_head=stride_key_head,
        V=V,
        stride_value_seq=stride_value_seq,
        stride_value_head=stride_value_head,
        batch_token_start=batch_token_start,
        K_cache=K_cache,
        stride_key_cache_block=stride_key_cache_block,
        stride_key_cache_slot=stride_key_cache_slot,
        stride_key_cache_head=stride_key_cache_head,
        V_cache=V_cache,
        stride_value_cache_block=stride_value_cache_block,
        stride_value_cache_slot=stride_value_cache_slot,
        stride_value_cache_head=stride_value_cache_head,
        block_table=block_table,
        stride_block_table_batch=stride_block_table_batch,
        request_index=request_index,
    )

    # Pass 2: accumulate softmax-weighted V using running_row_max from pass 1.
    running_row_max, softmax_denominator_fxp, output_accumulator_fxp = _attention_pass(
        query_row_pointers=query_row_pointers,
        query_row_mask=query_row_mask,
        causal_row_positions=causal_row_positions,
        key_end_position=key_end_position,
        head_dim_offsets=head_dim_offsets,
        head_dim_mask=head_dim_mask,
        kv_head_index=kv_head_index,
        head_index=head_index,
        softmax_scale=softmax_scale,
        running_row_max=running_row_max,
        softmax_denominator_fxp=softmax_denominator_fxp,
        output_accumulator_fxp=output_accumulator_fxp,
        alibi_slopes_ptr=alibi_slopes_ptr,
        PASS=2,
        IS_CAUSAL=IS_CAUSAL,
        USE_ALIBI=USE_ALIBI,
        QUERY_BLOCK_SIZE=QUERY_BLOCK_SIZE,
        KEY_BLOCK_SIZE=KEY_BLOCK_SIZE,
        HEAD_DIM=HEAD_DIM,
        HEAD_DIM_CHUNK=HEAD_DIM_CHUNK,
        FRACTIONAL_BITS=FRACTIONAL_BITS,
        FIXED_POINT_DTYPE=FIXED_POINT_DTYPE,
        IS_PAGED=IS_PAGED,
        PAGE_SIZE=PAGE_SIZE,
        K=K,
        stride_key_seq=stride_key_seq,
        stride_key_head=stride_key_head,
        V=V,
        stride_value_seq=stride_value_seq,
        stride_value_head=stride_value_head,
        batch_token_start=batch_token_start,
        K_cache=K_cache,
        stride_key_cache_block=stride_key_cache_block,
        stride_key_cache_slot=stride_key_cache_slot,
        stride_key_cache_head=stride_key_cache_head,
        V_cache=V_cache,
        stride_value_cache_block=stride_value_cache_block,
        stride_value_cache_slot=stride_value_cache_slot,
        stride_value_cache_head=stride_value_cache_head,
        block_table=block_table,
        stride_block_table_batch=stride_block_table_batch,
        request_index=request_index,
    )

    softmax_denominator = fixed_to_float(
        x=softmax_denominator_fxp,
        fractional_bit_width=FRACTIONAL_BITS,
        floating_point_type=tl.float32,
    )
    output_accumulator = fixed_to_float(
        x=output_accumulator_fxp,
        fractional_bit_width=FRACTIONAL_BITS,
        floating_point_type=tl.float32,
    )
    softmax_denominator_safe = tl.maximum(softmax_denominator, 1.0e-6)
    output_accumulator = output_accumulator / softmax_denominator_safe[:, None]

    output_offsets = (
        (output_row_base + query_offsets[:, None]) * stride_output_seq
        + head_index * stride_output_head
        + head_dim_offsets[None, :]
    )
    tl.store(
        output + output_offsets,
        output_accumulator,
        mask=query_row_mask[:, None] & head_dim_mask[None, :],
    )


@triton.jit
def unified_attention_fxp_kernel(
    Q,
    K_cache,
    V_cache,
    softmax_scale,
    batch_start_locations,
    batch_sequence_lengths,
    block_table,
    output,
    alibi_slopes_ptr,
    q_block_to_req,
    q_block_local,
    stride_query_seq,
    stride_query_head,
    stride_key_cache_block,
    stride_key_cache_slot,
    stride_key_cache_head,
    stride_value_cache_block,
    stride_value_cache_slot,
    stride_value_cache_head,
    stride_output_seq,
    stride_output_head,
    stride_block_table_batch,
    kv_group_size: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    KEY_BLOCK_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FRACTIONAL_BITS: tl.constexpr,
    HEAD_DIM_CHUNK: tl.constexpr,
    FIXED_POINT_DTYPE: tl.constexpr,
):
    """Unified prefill+decode attention over a paged KV cache."""
    global_q_block = tl.program_id(0)
    head_index = tl.program_id(1)

    request_index = tl.load(q_block_to_req + global_q_block)
    query_block_index = tl.load(q_block_local + global_q_block)

    kv_head_index = head_index // kv_group_size

    current_sequence_length = tl.load(batch_sequence_lengths + request_index)
    query_start = tl.load(batch_start_locations + request_index)
    query_end = tl.load(batch_start_locations + request_index + 1)
    current_query_length = query_end - query_start
    current_context_length = current_sequence_length - current_query_length

    query_block_start = QUERY_BLOCK_SIZE * query_block_index

    query_offsets = query_block_start + tl.arange(0, QUERY_BLOCK_SIZE)
    head_dim_offsets = tl.arange(0, HEAD_DIM_PADDED)
    head_dim_mask = head_dim_offsets < HEAD_DIM

    query_absolute_positions = current_context_length + query_offsets
    query_row_mask = query_offsets < current_query_length
    query_row_pointers = (
        Q
        + (query_start + query_offsets) * stride_query_seq
        + head_index * stride_query_head
    )

    if IS_CAUSAL:
        key_end_position = tl.minimum(
            current_sequence_length,
            current_context_length + (query_block_index + 1) * QUERY_BLOCK_SIZE,
        )
    else:
        key_end_position = current_sequence_length

    attention_fwd_fxp_body(
        output=output,
        softmax_scale=softmax_scale,
        query_row_pointers=query_row_pointers,
        query_row_mask=query_row_mask,
        causal_row_positions=query_absolute_positions,
        output_row_base=query_start,
        key_end_position=key_end_position,
        query_offsets=query_offsets,
        head_dim_offsets=head_dim_offsets,
        head_dim_mask=head_dim_mask,
        kv_head_index=kv_head_index,
        head_index=head_index,
        stride_output_seq=stride_output_seq,
        stride_output_head=stride_output_head,
        alibi_slopes_ptr=alibi_slopes_ptr,
        IS_CAUSAL=IS_CAUSAL,
        USE_ALIBI=USE_ALIBI,
        QUERY_BLOCK_SIZE=QUERY_BLOCK_SIZE,
        KEY_BLOCK_SIZE=KEY_BLOCK_SIZE,
        HEAD_DIM_PADDED=HEAD_DIM_PADDED,
        HEAD_DIM=HEAD_DIM,
        HEAD_DIM_CHUNK=HEAD_DIM_CHUNK,
        FRACTIONAL_BITS=FRACTIONAL_BITS,
        FIXED_POINT_DTYPE=FIXED_POINT_DTYPE,
        IS_PAGED=True,
        PAGE_SIZE=PAGE_SIZE,
        K_cache=K_cache,
        stride_key_cache_block=stride_key_cache_block,
        stride_key_cache_slot=stride_key_cache_slot,
        stride_key_cache_head=stride_key_cache_head,
        V_cache=V_cache,
        stride_value_cache_block=stride_value_cache_block,
        stride_value_cache_slot=stride_value_cache_slot,
        stride_value_cache_head=stride_value_cache_head,
        block_table=block_table,
        stride_block_table_batch=stride_block_table_batch,
        request_index=request_index,
    )


def unified_attention_fxp(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    o: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    alibi_slopes: torch.Tensor | None = None,
    fxp_dtype=tl.int32,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    frac_bits: int = 14,
    block_n: int = 16,
    d_chunk: int = 16,
):
    """Run unified prefill + decode attention on a vLLM-layout paged KV cache."""

    assert q.is_cuda and kv_cache.is_cuda, "unified_attention_fxp requires CUDA tensors"
    assert kv_cache.ndim == 5 and kv_cache.shape[1] == 2, (
        "kv_cache must be (num_blocks, 2, block_size, num_kv_heads, head_dim); "
        f"got {tuple(kv_cache.shape)}"
    )
    key_cache, value_cache = kv_cache.unbind(1)

    head_dim_query, head_dim = q.shape[-1], key_cache.shape[-1]
    assert head_dim_query == head_dim, "head_dim mismatch between q and key_cache"
    softmax_scale_value = prepare_log2_softmax_scale(head_dim_query, softmax_scale)

    use_alibi = alibi_slopes is not None
    if use_alibi:
        assert alibi_slopes.is_cuda and alibi_slopes.dtype == torch.float32, (
            "alibi_slopes must be a float32 CUDA tensor"
        )

        alibi_slopes_scaled = alibi_slopes * RCP_LN2
    else:
        alibi_slopes_scaled = q

    query_block_size = 64
    num_requests = seq_lens.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    kv_group_size = num_heads // num_kv_heads
    page_size = key_cache.shape[1]

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    blocks_per_req = (query_lens + (query_block_size - 1)) // query_block_size
    total_q_blocks = int(blocks_per_req.sum().item())
    if total_q_blocks == 0:
        return

    device = q.device
    q_block_to_req = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int32),
        blocks_per_req,
    )
    cu_blocks = torch.zeros(num_requests + 1, device=device, dtype=torch.int32)
    torch.cumsum(blocks_per_req, 0, out=cu_blocks[1:])
    q_block_local = torch.arange(
        total_q_blocks, device=device, dtype=torch.int32
    ) - cu_blocks[:-1].index_select(0, q_block_to_req.to(torch.int64))

    grid = (total_q_blocks, num_heads)

    unified_attention_fxp_kernel[grid](
        Q=q,
        K_cache=key_cache,
        V_cache=value_cache,
        softmax_scale=softmax_scale_value,
        batch_start_locations=query_start_loc,
        batch_sequence_lengths=seq_lens,
        block_table=block_table,
        output=o,
        alibi_slopes_ptr=alibi_slopes_scaled,
        q_block_to_req=q_block_to_req,
        q_block_local=q_block_local,
        stride_query_seq=q.stride(0),
        stride_query_head=q.stride(1),
        stride_key_cache_block=key_cache.stride(0),
        stride_key_cache_slot=key_cache.stride(1),
        stride_key_cache_head=key_cache.stride(2),
        stride_value_cache_block=value_cache.stride(0),
        stride_value_cache_slot=value_cache.stride(1),
        stride_value_cache_head=value_cache.stride(2),
        stride_output_seq=o.stride(0),
        stride_output_head=o.stride(1),
        stride_block_table_batch=block_table.stride(0),
        kv_group_size=kv_group_size,
        PAGE_SIZE=page_size,
        QUERY_BLOCK_SIZE=query_block_size,
        HEAD_DIM_PADDED=triton.next_power_of_2(head_dim),
        KEY_BLOCK_SIZE=block_n,
        IS_CAUSAL=is_causal,
        USE_ALIBI=use_alibi,
        HEAD_DIM=head_dim,
        FRACTIONAL_BITS=frac_bits,
        HEAD_DIM_CHUNK=d_chunk,
        FIXED_POINT_DTYPE=fxp_dtype,
        num_warps=4 if head_dim <= 64 else 8,
        num_stages=1,
    )
