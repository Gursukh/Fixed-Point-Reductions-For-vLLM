import torch
import triton
import triton.language as tl

from fxpr_vllm.fixed_point_kernels.fixed_point import RCP_LN2

from .attention import attention_fwd_fxp_body, prepare_log2_softmax_scale


@triton.jit
def prefill_fxp_kernel(
    Q,
    K,
    V,
    softmax_scale,
    batch_start_locations,
    batch_sequence_lengths,
    output,
    alibi_slopes_ptr,
    stride_query_seq,
    stride_query_head,
    stride_key_seq,
    stride_key_head,
    stride_value_seq,
    stride_value_head,
    stride_output_seq,
    stride_output_head,
    kv_group_size: tl.constexpr,
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
    """Non-paged fixed-point prefill. Used for tests where K/V are plain (seq, head, dim)."""
    request_index = tl.program_id(0)
    head_index = tl.program_id(1)
    query_block_index = tl.program_id(2)

    kv_head_index = head_index // kv_group_size

    current_sequence_length = tl.load(batch_sequence_lengths + request_index)
    current_batch_start = tl.load(batch_start_locations + request_index)
    query_block_start = QUERY_BLOCK_SIZE * query_block_index
    if query_block_start >= current_sequence_length:
        return

    head_dim_offsets = tl.arange(0, HEAD_DIM_PADDED)
    query_offsets = query_block_index * QUERY_BLOCK_SIZE + tl.arange(
        0, QUERY_BLOCK_SIZE
    )
    head_dim_mask = head_dim_offsets < HEAD_DIM

    query_row_pointers = (
        Q
        + (current_batch_start + query_offsets) * stride_query_seq
        + head_index * stride_query_head
    )
    query_row_mask = query_offsets < current_sequence_length

    key_end_position = current_sequence_length
    if IS_CAUSAL:
        key_end_position = tl.minimum(
            key_end_position, (query_block_index + 1) * QUERY_BLOCK_SIZE
        )

    attention_fwd_fxp_body(
        output=output,
        softmax_scale=softmax_scale,
        query_row_pointers=query_row_pointers,
        query_row_mask=query_row_mask,
        causal_row_positions=query_offsets,
        output_row_base=current_batch_start,
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
        K=K,
        stride_key_seq=stride_key_seq,
        stride_key_head=stride_key_head,
        V=V,
        stride_value_seq=stride_value_seq,
        stride_value_head=stride_value_head,
        batch_token_start=current_batch_start,
    )


def context_attention_fwd_fxp_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    *,
    alibi_slopes: torch.Tensor | None = None,
    fxp_dtype=tl.int32,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    frac_bits: int = 14,
    block_n: int = 16,
    d_chunk: int = 16,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda, (
        "context_attention_fwd_fxp_kernel requires CUDA tensors"
    )
    assert k.ndim == 3 and v.ndim == 3, (
        "k/v must be 3D (seq, head, dim); "
        f"got k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}"
    )
    head_dim_query, head_dim = q.shape[-1], k.shape[-1]
    assert head_dim_query == head_dim, "head_dim mismatch between q and k"
    softmax_scale_value = prepare_log2_softmax_scale(head_dim_query, softmax_scale)

    use_alibi = alibi_slopes is not None
    if use_alibi:
        assert alibi_slopes.is_cuda and alibi_slopes.dtype == torch.float32, (
            "alibi_slopes must be a float32 CUDA tensor"
        )
        alibi_slopes_scaled = alibi_slopes * RCP_LN2
    else:
        alibi_slopes_scaled = q  # placeholder; kernel guarded by USE_ALIBI.

    query_block_size = 64
    num_batches = b_seq_len.shape[0]
    num_heads = q.shape[1]
    kv_group_size = q.shape[1] // k.shape[1]

    grid = (num_batches, num_heads, triton.cdiv(max_input_len, query_block_size))

    prefill_fxp_kernel[grid](
        Q=q,
        K=k,
        V=v,
        softmax_scale=softmax_scale_value,
        batch_start_locations=b_start_loc,
        batch_sequence_lengths=b_seq_len,
        output=o,
        alibi_slopes_ptr=alibi_slopes_scaled,
        stride_query_seq=q.stride(0),
        stride_query_head=q.stride(1),
        stride_key_seq=k.stride(0),
        stride_key_head=k.stride(1),
        stride_value_seq=v.stride(0),
        stride_value_head=v.stride(1),
        stride_output_seq=o.stride(0),
        stride_output_head=o.stride(1),
        kv_group_size=kv_group_size,
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
