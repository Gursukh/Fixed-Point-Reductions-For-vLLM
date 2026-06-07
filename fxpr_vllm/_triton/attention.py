"""Unified prefill + decode attention over paged KV, tensor-core tiled.

One fused kernel when num_splits == 1, otherwise a two-kernel split path.
tl.dot handles the within-tile QK^T and P*V; cross-tile reductions are integer
fixed-point sums, which keeps results batch-invariant. softmax_scale is
pre-scaled by 1/ln(2) so we can use exp2 in log2 space.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .fxp import fxp_constants, float_to_fixed, fixed_to_float


_RCP_LN2 = 1.4426950408889634
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_TORCH_TO_TL_FLOAT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# Key-tile width sets the determinism granularity. Fixed per dtype; splits and
# causal edges are BLOCK_N-aligned so a tl.dot tile always contracts the same keys.
_BLOCK_N_BY_DTYPE = {
    torch.float16: 64,
    torch.bfloat16: 64,
    torch.float32: 32,
}

_SPLIT_OCCUPANCY = 2  # roughly 2 split programs per SM to hide latency

# KV-split thresholds, tuned on decode sweeps across L4/A100/Blackwell. The old
# heuristic only compared batch to SM count and ignored context length, so it
# split short-context batch=1 decode (about 170% slower) and refused to split
# long-context batched decode (left about 50% on the table). These split only
# when there's enough KV work to pay back the atomic-add combine and the SMs
# aren't already full.
_MIN_KV_TILES_FOR_SPLIT = 32   # below this many KV tiles, just fuse
_MAX_NUM_SPLITS = 8            # past here the combine costs more than it saves
_SPLIT_BASE_OVERSUB = 2        # base over this many * SMs means saturated


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# Cache (compute_capability, SM count) per device since the launcher runs per layer.
_ARCH_CACHE: dict[int, tuple[int, int]] = {}


def _arch_info(device: torch.device) -> tuple[int, int]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    info = _ARCH_CACHE.get(idx)
    if info is None:
        props = torch.cuda.get_device_properties(idx)
        info = (props.major * 10 + props.minor, props.multi_processor_count)
        _ARCH_CACHE[idx] = info
    return info


def _check_arch(dtype: torch.dtype, device: torch.device) -> None:
    cap, _ = _arch_info(device)
    if cap < 75:
        raise RuntimeError(
            f"unified_attention_fxp: requires compute capability >= 7.5 "
            f"(Turing); device is {cap // 10}.{cap % 10}"
        )
    if dtype in (torch.bfloat16, torch.float32) and cap < 80:
        name = "bfloat16" if dtype == torch.bfloat16 else "float32"
        raise RuntimeError(
            f"unified_attention_fxp: {name} inputs require compute capability "
            f">= 8.0 (Ampere) for tensor-core tl.dot; device is "
            f"{cap // 10}.{cap % 10}. Use float16 inputs."
        )


@triton.jit
def _split_tile_range(lo, hi, num_splits, split_index, BLOCK_N: tl.constexpr):
    """BLOCK_N-aligned key range owned by split_index. Whole tiles split evenly,
    and boundaries don't depend on num_splits, so partials are split-invariant."""
    total = hi - lo
    n_tiles = tl.maximum(tl.cdiv(total, BLOCK_N), 0)
    per_split = tl.cdiv(n_tiles, num_splits)
    t_start = split_index * per_split
    t_end = tl.minimum(t_start + per_split, n_tiles)
    chunk_start = lo + t_start * BLOCK_N
    chunk_end = lo + tl.maximum(t_end, t_start) * BLOCK_N
    return chunk_start, chunk_end


@triton.jit
def _qk_post(
    scores,
    softmax_scale_log2,
    softcap_log2,
    alibi_slope,
    n_off,
    q_abs_pos,
    HAS_SOFTCAP: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
):
    """Apply scale, softcap, and alibi to a [BLOCK_M, BLOCK_N] raw qk score tile."""
    scores = scores * softmax_scale_log2
    if HAS_SOFTCAP:
        scores = softcap_log2 * libdevice.tanh(scores / softcap_log2)
    if HAS_ALIBI:
        scores = scores + alibi_slope * (
            n_off[None, :].to(tl.float32) - q_abs_pos[:, None].to(tl.float32)
        )
    return scores


# do_not_specialize stride_block_table_row: it depends on max_model_len, so
# warmup and real requests differ and would trigger a recompile. It only offsets
# a small int32 gather, so specializing it gains nothing.
@triton.jit(
    do_not_specialize=["stride_block_table_row"],
    do_not_specialize_on_alignment=["stride_block_table_row"],
)
def _attn_fused_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    qsl_ptr,
    seq_lens_ptr,
    block_table_ptr,
    alibi_slopes_ptr,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    window_size,
    stride_q_token,
    stride_q_head,
    stride_k_block,
    stride_k_slot,
    stride_k_head,
    stride_v_block,
    stride_v_slot,
    stride_v_head,
    stride_o_token,
    stride_o_head,
    stride_block_table_row,
    softmax_scale_log2,
    softcap_log2,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-kernel attention for num_splits == 1. Each program owns its
    (req, q_block, head) key range and runs both passes (row-max, then denom and
    weighted-V) as two K-loops, with no scratch or atomics. Integer accumulators
    keep it bit-identical to the split path."""
    req = tl.program_id(axis=0)
    q_block = tl.program_id(axis=1)
    head_index = tl.program_id(axis=2)
    kv_group = num_heads // num_kv_heads
    kv_head_index = head_index // kv_group

    q_start = tl.load(qsl_ptr + req)
    q_end = tl.load(qsl_ptr + req + 1)
    q_len = q_end - q_start
    m_base = q_block * BLOCK_M
    if m_base >= q_len:
        return

    seq_len = tl.load(seq_lens_ptr + req)
    context_len = seq_len - q_len

    m_off = tl.arange(0, BLOCK_M)
    q_in_req = m_base + m_off
    row_valid = q_in_req < q_len
    q_token_idx = q_start + q_in_req
    q_abs_pos = context_len + q_in_req

    if IS_CAUSAL:
        hi = tl.minimum(seq_len, context_len + m_base + BLOCK_M)
    else:
        hi = seq_len
    if HAS_WINDOW:
        lo = tl.maximum(0, (context_len + m_base) - window_size + 1)
    else:
        lo = 0

    if IS_CAUSAL:
        key_end_row = q_abs_pos + 1
    else:
        key_end_row = seq_len + 0 * q_abs_pos
    if HAS_WINDOW:
        key_start_row = tl.maximum(0, q_abs_pos - window_size + 1)
    else:
        key_start_row = 0 * q_abs_pos

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes_ptr + head_index)
    else:
        alibi_slope = 0.0

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    q_ptrs = (
        Q_ptr
        + q_token_idx[:, None] * stride_q_token
        + head_index * stride_q_head
        + d_off[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=row_valid[:, None] & d_mask[None, :], other=0.0)
    bt_row = block_table_ptr + req * stride_block_table_row

    # Pass 1: per-row max over the key range.
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for kn in range(lo, hi, BLOCK_N):
        n_off = kn + tl.arange(0, BLOCK_N)
        n_in_seq = n_off < seq_len
        logical_block = n_off // page_size
        slot = n_off % page_size
        phys = tl.load(bt_row + logical_block, mask=n_in_seq, other=0)
        kv_base = phys * stride_k_block + slot * stride_k_slot + kv_head_index * stride_k_head
        k_ptrs = K_ptr + kv_base[None, :] + d_off[:, None]
        k_tile = tl.load(k_ptrs, mask=n_in_seq[None, :] & d_mask[:, None], other=0.0)
        scores = tl.dot(q_tile, k_tile, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
        scores = _qk_post(
            scores, softmax_scale_log2, softcap_log2, alibi_slope,
            n_off, q_abs_pos, HAS_SOFTCAP, HAS_ALIBI,
        )
        valid = (
            (n_off[None, :] >= key_start_row[:, None])
            & (n_off[None, :] < key_end_row[:, None])
            & row_valid[:, None]
        )
        scores = tl.where(valid, scores, -float("inf"))
        row_max = tl.maximum(row_max, tl.max(scores, axis=1))
    # Invalid rows have no keys; pin to 0 so exp2 returns 0 instead of nan.
    global_max = tl.where(row_valid, row_max, 0.0)

    # Pass 2: integer-accumulate the denom and weighted-V.
    denom_acc = tl.zeros((BLOCK_M,), dtype=INT_DTYPE)
    wv_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=INT_DTYPE)
    for kn in range(lo, hi, BLOCK_N):
        n_off = kn + tl.arange(0, BLOCK_N)
        n_in_seq = n_off < seq_len
        logical_block = n_off // page_size
        slot = n_off % page_size
        phys = tl.load(bt_row + logical_block, mask=n_in_seq, other=0)
        k_base = phys * stride_k_block + slot * stride_k_slot + kv_head_index * stride_k_head
        k_ptrs = K_ptr + k_base[None, :] + d_off[:, None]
        k_tile = tl.load(k_ptrs, mask=n_in_seq[None, :] & d_mask[:, None], other=0.0)
        v_base = phys * stride_v_block + slot * stride_v_slot + kv_head_index * stride_v_head
        v_ptrs = V_ptr + v_base[:, None] + d_off[None, :]
        v_tile = tl.load(v_ptrs, mask=n_in_seq[:, None] & d_mask[None, :], other=0.0)
        scores = tl.dot(q_tile, k_tile, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
        scores = _qk_post(
            scores, softmax_scale_log2, softcap_log2, alibi_slope,
            n_off, q_abs_pos, HAS_SOFTCAP, HAS_ALIBI,
        )
        valid = (
            (n_off[None, :] >= key_start_row[:, None])
            & (n_off[None, :] < key_end_row[:, None])
            & row_valid[:, None]
        )
        scores = tl.where(valid, scores, -float("inf"))
        weight = libdevice.exp2(scores - global_max[:, None])
        # tl.sum promotes integer reductions to int32, so cast back to keep the
        # loop-carried accumulator in its declared INT_DTYPE.
        denom_acc += tl.sum(
            float_to_fixed(weight, SCALE, QMIN, QMAX, INT_DTYPE), axis=1
        ).to(INT_DTYPE)
        pv = tl.dot(
            weight.to(IO_DTYPE), v_tile, allow_tf32=ALLOW_TF32, out_dtype=tl.float32
        )
        wv_acc += float_to_fixed(pv, SCALE, QMIN, QMAX, INT_DTYPE)

    denom = tl.maximum(fixed_to_float(denom_acc, INV_SCALE), 1.0e-6)
    num = fixed_to_float(wv_acc, INV_SCALE)
    out = num / denom[:, None]
    o_ptrs = (
        O_ptr
        + q_token_idx[:, None] * stride_o_token
        + head_index * stride_o_head
        + d_off[None, :]
    )
    tl.store(o_ptrs, out.to(IO_DTYPE), mask=row_valid[:, None] & d_mask[None, :])


# See _attn_fused_kernel. num_splits is also excluded so the split count can't
# trigger a recompile (BLOCK_S still varies, but warmup covers it).
@triton.jit(
    do_not_specialize=["stride_block_table_row", "num_splits"],
    do_not_specialize_on_alignment=["stride_block_table_row", "num_splits"],
)
def _attn_split_max_kernel(
    Q_ptr,
    K_ptr,
    partial_max_ptr,
    qsl_ptr,
    seq_lens_ptr,
    block_table_ptr,
    alibi_slopes_ptr,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    num_splits,
    window_size,
    stride_q_token,
    stride_q_head,
    stride_k_block,
    stride_k_slot,
    stride_k_head,
    stride_pmax_token,
    stride_pmax_head,
    stride_pmax_split,
    stride_block_table_row,
    softmax_scale_log2,
    softcap_log2,
    ALLOW_TF32: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(axis=0)
    q_block = tl.program_id(axis=1)
    hs = tl.program_id(axis=2)
    head_index = hs // num_splits
    split_index = hs % num_splits
    kv_group = num_heads // num_kv_heads
    kv_head_index = head_index // kv_group

    q_start = tl.load(qsl_ptr + req)
    q_end = tl.load(qsl_ptr + req + 1)
    q_len = q_end - q_start
    m_base = q_block * BLOCK_M
    if m_base >= q_len:
        return

    seq_len = tl.load(seq_lens_ptr + req)
    context_len = seq_len - q_len

    m_off = tl.arange(0, BLOCK_M)
    q_in_req = m_base + m_off
    row_valid = q_in_req < q_len
    q_token_idx = q_start + q_in_req
    q_abs_pos = context_len + q_in_req

    # Block-uniform key range [lo, hi) covering every row's causal/window span.
    if IS_CAUSAL:
        hi = tl.minimum(seq_len, context_len + m_base + BLOCK_M)
    else:
        hi = seq_len
    if HAS_WINDOW:
        lo = tl.maximum(0, (context_len + m_base) - window_size + 1)
    else:
        lo = 0
    chunk_start, chunk_end = _split_tile_range(lo, hi, num_splits, split_index, BLOCK_N)

    # Per-row bounds for masking the ragged causal/window edge.
    if IS_CAUSAL:
        key_end_row = q_abs_pos + 1
    else:
        key_end_row = seq_len + 0 * q_abs_pos
    if HAS_WINDOW:
        key_start_row = tl.maximum(0, q_abs_pos - window_size + 1)
    else:
        key_start_row = 0 * q_abs_pos

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes_ptr + head_index)
    else:
        alibi_slope = 0.0

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    q_ptrs = (
        Q_ptr
        + q_token_idx[:, None] * stride_q_token
        + head_index * stride_q_head
        + d_off[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=row_valid[:, None] & d_mask[None, :], other=0.0)
    bt_row = block_table_ptr + req * stride_block_table_row

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for kn in range(chunk_start, chunk_end, BLOCK_N):
        n_off = kn + tl.arange(0, BLOCK_N)
        n_in_seq = n_off < seq_len
        logical_block = n_off // page_size
        slot = n_off % page_size
        phys = tl.load(bt_row + logical_block, mask=n_in_seq, other=0)
        kv_base = phys * stride_k_block + slot * stride_k_slot + kv_head_index * stride_k_head
        k_ptrs = K_ptr + kv_base[None, :] + d_off[:, None]
        k_tile = tl.load(
            k_ptrs, mask=n_in_seq[None, :] & d_mask[:, None], other=0.0
        )
        scores = tl.dot(q_tile, k_tile, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
        scores = _qk_post(
            scores, softmax_scale_log2, softcap_log2, alibi_slope,
            n_off, q_abs_pos, HAS_SOFTCAP, HAS_ALIBI,
        )
        valid = (
            (n_off[None, :] >= key_start_row[:, None])
            & (n_off[None, :] < key_end_row[:, None])
            & row_valid[:, None]
        )
        scores = tl.where(valid, scores, -float("inf"))
        row_max = tl.maximum(row_max, tl.max(scores, axis=1))

    pm_ptrs = (
        partial_max_ptr
        + q_token_idx * stride_pmax_token
        + head_index * stride_pmax_head
        + split_index * stride_pmax_split
    )
    tl.store(pm_ptrs, row_max, mask=row_valid)


# See _attn_split_max_kernel.
@triton.jit(
    do_not_specialize=["stride_block_table_row", "num_splits"],
    do_not_specialize_on_alignment=["stride_block_table_row", "num_splits"],
)
def _attn_split_dv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    partial_max_ptr,
    total_denom_ptr,
    total_wv_ptr,
    lock_ptr,
    O_ptr,
    qsl_ptr,
    seq_lens_ptr,
    block_table_ptr,
    alibi_slopes_ptr,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    num_splits,
    window_size,
    stride_q_token,
    stride_q_head,
    stride_k_block,
    stride_k_slot,
    stride_k_head,
    stride_v_block,
    stride_v_slot,
    stride_v_head,
    stride_pmax_token,
    stride_pmax_head,
    stride_pmax_split,
    stride_td_token,
    stride_td_head,
    stride_twv_token,
    stride_twv_head,
    stride_twv_d,
    stride_lock_token,
    stride_lock_head,
    stride_o_token,
    stride_o_head,
    stride_block_table_row,
    softmax_scale_log2,
    softcap_log2,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    req = tl.program_id(axis=0)
    q_block = tl.program_id(axis=1)
    hs = tl.program_id(axis=2)
    head_index = hs // num_splits
    split_index = hs % num_splits
    kv_group = num_heads // num_kv_heads
    kv_head_index = head_index // kv_group

    q_start = tl.load(qsl_ptr + req)
    q_end = tl.load(qsl_ptr + req + 1)
    q_len = q_end - q_start
    m_base = q_block * BLOCK_M
    if m_base >= q_len:
        return

    seq_len = tl.load(seq_lens_ptr + req)
    context_len = seq_len - q_len

    m_off = tl.arange(0, BLOCK_M)
    q_in_req = m_base + m_off
    row_valid = q_in_req < q_len
    q_token_idx = q_start + q_in_req
    q_abs_pos = context_len + q_in_req

    if IS_CAUSAL:
        hi = tl.minimum(seq_len, context_len + m_base + BLOCK_M)
    else:
        hi = seq_len
    if HAS_WINDOW:
        lo = tl.maximum(0, (context_len + m_base) - window_size + 1)
    else:
        lo = 0
    # An empty chunk still has to hit the counter; every split must arrive or
    # the last-arrival epilogue never fires.
    chunk_start, chunk_end = _split_tile_range(lo, hi, num_splits, split_index, BLOCK_N)

    if IS_CAUSAL:
        key_end_row = q_abs_pos + 1
    else:
        key_end_row = seq_len + 0 * q_abs_pos
    if HAS_WINDOW:
        key_start_row = tl.maximum(0, q_abs_pos - window_size + 1)
    else:
        key_start_row = 0 * q_abs_pos

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes_ptr + head_index)
    else:
        alibi_slope = 0.0

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < head_dim
    q_ptrs = (
        Q_ptr
        + q_token_idx[:, None] * stride_q_token
        + head_index * stride_q_head
        + d_off[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=row_valid[:, None] & d_mask[None, :], other=0.0)
    bt_row = block_table_ptr + req * stride_block_table_row

    # Combine the per-split row maxes from pass 1 into one max per row.
    s_off = tl.arange(0, BLOCK_S)
    s_mask = s_off < num_splits
    pm_ptrs = (
        partial_max_ptr
        + q_token_idx[:, None] * stride_pmax_token
        + head_index * stride_pmax_head
        + s_off[None, :] * stride_pmax_split
    )
    pm_vals = tl.load(
        pm_ptrs, mask=row_valid[:, None] & s_mask[None, :], other=-float("inf")
    )
    global_max = tl.max(pm_vals, axis=1)
    # Invalid rows have no stored max; pin to 0 so exp2 returns 0 instead of nan.
    global_max = tl.where(row_valid, global_max, 0.0)

    # Accumulate in ACC_DTYPE. The integer atomic-add combine below has no int16
    # variant, so 16-bit fixed point sums its per-tile int16 quantiles in int32.
    # Integer addition is exact, so this matches the fused path and the
    # cross-split reduction stays order-independent.
    denom_acc = tl.zeros((BLOCK_M,), dtype=ACC_DTYPE)
    wv_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=ACC_DTYPE)

    for kn in range(chunk_start, chunk_end, BLOCK_N):
        n_off = kn + tl.arange(0, BLOCK_N)
        n_in_seq = n_off < seq_len
        logical_block = n_off // page_size
        slot = n_off % page_size
        phys = tl.load(bt_row + logical_block, mask=n_in_seq, other=0)
        k_base = phys * stride_k_block + slot * stride_k_slot + kv_head_index * stride_k_head
        k_ptrs = K_ptr + k_base[None, :] + d_off[:, None]
        k_tile = tl.load(k_ptrs, mask=n_in_seq[None, :] & d_mask[:, None], other=0.0)
        v_base = phys * stride_v_block + slot * stride_v_slot + kv_head_index * stride_v_head
        v_ptrs = V_ptr + v_base[:, None] + d_off[None, :]
        v_tile = tl.load(v_ptrs, mask=n_in_seq[:, None] & d_mask[None, :], other=0.0)

        scores = tl.dot(q_tile, k_tile, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
        scores = _qk_post(
            scores, softmax_scale_log2, softcap_log2, alibi_slope,
            n_off, q_abs_pos, HAS_SOFTCAP, HAS_ALIBI,
        )
        valid = (
            (n_off[None, :] >= key_start_row[:, None])
            & (n_off[None, :] < key_end_row[:, None])
            & row_valid[:, None]
        )
        scores = tl.where(valid, scores, -float("inf"))

        # exp2 of a masked (-inf) score is exactly 0, so it drops out.
        weight = libdevice.exp2(scores - global_max[:, None])
        # Denom: quantize per key, then integer-sum across the tile.
        denom_acc += tl.sum(
            float_to_fixed(weight, SCALE, QMIN, QMAX, INT_DTYPE), axis=1
        ).to(ACC_DTYPE)
        # P*V via fp32 tl.dot (fixed tile shape, so deterministic). Quantize the
        # per-tile partial; the cross-tile sum stays integer.
        pv = tl.dot(
            weight.to(IO_DTYPE), v_tile, allow_tf32=ALLOW_TF32, out_dtype=tl.float32
        )
        wv_acc += float_to_fixed(pv, SCALE, QMIN, QMAX, INT_DTYPE).to(ACC_DTYPE)

    # Add this split into the per-(token, head) integer totals. Integer
    # atomic-add is commutative, so the cross-split reduction is order-free.
    td_ptrs = total_denom_ptr + q_token_idx * stride_td_token + head_index * stride_td_head
    tl.atomic_add(td_ptrs, denom_acc, mask=row_valid, sem="relaxed")
    twv_ptrs = (
        total_wv_ptr
        + q_token_idx[:, None] * stride_twv_token
        + head_index * stride_twv_head
        + d_off[None, :] * stride_twv_d
    )
    tl.atomic_add(
        twv_ptrs, wv_acc, mask=row_valid[:, None] & d_mask[None, :], sem="relaxed"
    )

    # Fused combine: the last split to arrive divides and writes O. The acq_rel
    # increment makes its loads see every split's atomic-add, so the result
    # stays order-independent.
    lk_ptrs = lock_ptr + q_token_idx * stride_lock_token + head_index * stride_lock_head
    arrived = tl.atomic_add(
        lk_ptrs, tl.full((BLOCK_M,), 1, tl.int32), mask=row_valid, sem="acq_rel"
    )
    is_last = (arrived == num_splits - 1) & row_valid

    denom_sum = tl.load(td_ptrs, mask=row_valid, other=0)
    denom = tl.maximum(fixed_to_float(denom_sum, INV_SCALE), 1.0e-6)
    wv_sum = tl.load(twv_ptrs, mask=row_valid[:, None] & d_mask[None, :], other=0)
    num = fixed_to_float(wv_sum, INV_SCALE)
    out = num / denom[:, None]
    o_ptrs = (
        O_ptr
        + q_token_idx[:, None] * stride_o_token
        + head_index * stride_o_head
        + d_off[None, :]
    )
    tl.store(o_ptrs, out.to(IO_DTYPE), mask=is_last[:, None] & d_mask[None, :])


def _pick_launch(max_query_len: int) -> tuple[int, int, int]:
    """Return (BLOCK_M, num_warps, num_stages) for the attention kernels.
    BLOCK_M stays small because the [BLOCK_M, BLOCK_D] integer wv accumulator
    dominates register pressure; decode (one query row) uses 16, the tl.dot
    minimum."""
    if max_query_len <= 1:
        return 16, 4, 2
    return 32, 8, 2


def _pick_num_splits(
    requested: int,
    num_requests: int,
    num_q_blocks: int,
    num_heads: int,
    num_kv_tiles: int,
    device: torch.device,
) -> int:
    # An explicit request wins; tests pass requested >= 1 to force a split count.
    if requested >= 1:
        return requested
    # Everything here is static (batch, q-blocks, heads, KV-tile count from the
    # block-table shape, SM count), so it's safe to bake into a graph.
    _, num_sms = _arch_info(device)
    # Short context: not enough KV work to pay back the atomic-add combine.
    if num_kv_tiles <= _MIN_KV_TILES_FOR_SPLIT:
        return 1
    # The request/q-block/head tiling already fills the device, so splitting
    # would only add combine contention (the large-batch decode case).
    base = max(1, num_requests * num_q_blocks * num_heads)
    if base > _SPLIT_BASE_OVERSUB * num_sms:
        return 1
    # Otherwise split to shorten the KV scan: aim for about
    # _MIN_KV_TILES_FOR_SPLIT tiles per split, capped, rounded down to a power of two.
    split = min(num_kv_tiles // _MIN_KV_TILES_FOR_SPLIT, _MAX_NUM_SPLITS)
    return _next_pow2(split) if split >= 2 else 1


def unified_attention_fxp_run(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    o: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_query_len: int,
    alibi_slopes: torch.Tensor | None,
    is_causal: bool,
    softmax_scale: float | None,
    fxp_int_bits: int,
    fxp_frac_bits: int,
    logit_softcap: float,
    window_size: int,
    num_kv_splits: int,
) -> None:
    if not (q.is_cuda and kv_cache.is_cuda and o.is_cuda):
        raise RuntimeError("unified_attention_fxp: tensors must be CUDA")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"unified_attention_fxp: unsupported q dtype {q.dtype}"
        )
    if kv_cache.dim() != 5 or kv_cache.shape[1] != 2:
        raise ValueError(
            "kv_cache must be (num_blocks, 2, page_size, num_kv_heads, head_dim)"
        )
    if kv_cache.dtype != q.dtype or o.dtype != q.dtype:
        raise TypeError("kv_cache and o must share dtype with q")
    if query_start_loc.dtype != torch.int32:
        raise TypeError("query_start_loc must be int32")
    if seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must be int32")
    if block_table.dtype != torch.int32:
        raise TypeError("block_table must be int32")
    if num_kv_splits < 0:
        raise ValueError(f"num_kv_splits must be >= 0, got {num_kv_splits}")
    if alibi_slopes is not None and alibi_slopes.numel() > 0:
        if alibi_slopes.dtype != torch.float32:
            raise TypeError("alibi_slopes must be float32")
    _check_arch(q.dtype, q.device)

    k_cache = kv_cache.select(1, 0)
    v_cache = kv_cache.select(1, 1)

    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    num_requests = seq_lens.shape[0]

    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if num_requests == 0:
        return

    ss = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    ss_log2 = float(ss) * _RCP_LN2
    softcap_log2 = float(logit_softcap) * _RCP_LN2

    has_alibi = alibi_slopes is not None and alibi_slopes.numel() > 0
    if has_alibi:
        alibi = alibi_slopes.contiguous()
    else:
        # Triton needs a real tensor here; it's only read under HAS_ALIBI.
        alibi = q

    block_m, num_warps, num_stages = _pick_launch(int(max_query_len))
    block_n = _BLOCK_N_BY_DTYPE[q.dtype]
    block_d = max(_next_pow2(head_dim), 16)
    num_q_blocks = max(1, triton.cdiv(max(1, int(max_query_len)), block_m))

    # Estimate KV tiles from the block-table width. block_table is
    # (num_requests, max_blocks_per_seq), so max_blocks_per_seq * page_size is an
    # upper bound on context length. It's a host-side shape (no device sync) and
    # stays fixed across a graph capture, which is what the split needs.
    max_kv_blocks = block_table.shape[1] if block_table.dim() == 2 else 0
    num_kv_tiles = triton.cdiv(max(1, max_kv_blocks * page_size), block_n)

    effective_splits = _pick_num_splits(
        int(num_kv_splits), int(num_requests), int(num_q_blocks),
        int(num_heads), int(num_kv_tiles), q.device,
    )

    scale, inv_scale, qmin, qmax, tl_int_dtype, torch_int_dtype = fxp_constants(
        int(fxp_int_bits), int(fxp_frac_bits)
    )
    io_dtype = _TORCH_TO_TL_FLOAT[q.dtype]
    allow_tf32 = q.dtype == torch.float32

    has_softcap = float(logit_softcap) > 0.0
    has_window = int(window_size) > 0

    # Fast path: one split, so a single fused kernel owns each tile's full key
    # range and runs both passes with no scratch or atomics (prefill uses this).
    if effective_splits == 1:
        grid = (num_requests, num_q_blocks, num_heads)
        _attn_fused_kernel[grid](
            q,
            k_cache,
            v_cache,
            o,
            query_start_loc,
            seq_lens,
            block_table,
            alibi,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
            int(window_size),
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            o.stride(0),
            o.stride(1),
            block_table.stride(0),
            ss_log2,
            softcap_log2,
            SCALE=scale,
            INV_SCALE=inv_scale,
            QMIN=qmin,
            QMAX=qmax,
            INT_DTYPE=tl_int_dtype,
            IO_DTYPE=io_dtype,
            ALLOW_TF32=allow_tf32,
            HAS_ALIBI=has_alibi,
            HAS_SOFTCAP=has_softcap,
            IS_CAUSAL=bool(is_causal),
            HAS_WINDOW=has_window,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    # Split path (num_splits > 1): two kernels with an integer atomic-add
    # combine. split_dv adds each split into the per-(token, head) totals; lock
    # is the per-(token, head) arrival counter for the fused combine.
    T_total = q.shape[0]
    partial_max = torch.full(
        (T_total, num_heads, effective_splits),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    # Triton's integer atomic_add has no int16 variant, so the split combine
    # accumulates 16-bit fixed point in int32 buffers (exact, so identical to the
    # fused path). Other widths use their native integer dtype.
    if torch_int_dtype == torch.int16:
        tl_acc_dtype, torch_acc_dtype = tl.int32, torch.int32
    else:
        tl_acc_dtype, torch_acc_dtype = tl_int_dtype, torch_int_dtype
    total_denom = torch.zeros(
        (T_total, num_heads), device=q.device, dtype=torch_acc_dtype
    )
    total_wv = torch.zeros(
        (T_total, num_heads, head_dim), device=q.device, dtype=torch_acc_dtype
    )
    locks = torch.zeros((T_total, num_heads), device=q.device, dtype=torch.int32)

    block_s = max(_next_pow2(effective_splits), 1)

    grid = (num_requests, num_q_blocks, num_heads * effective_splits)

    _attn_split_max_kernel[grid](
        q,
        k_cache,
        partial_max,
        query_start_loc,
        seq_lens,
        block_table,
        alibi,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        effective_splits,
        int(window_size),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        block_table.stride(0),
        ss_log2,
        softcap_log2,
        ALLOW_TF32=allow_tf32,
        HAS_ALIBI=has_alibi,
        HAS_SOFTCAP=has_softcap,
        IS_CAUSAL=bool(is_causal),
        HAS_WINDOW=has_window,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _attn_split_dv_kernel[grid](
        q,
        k_cache,
        v_cache,
        partial_max,
        total_denom,
        total_wv,
        locks,
        o,
        query_start_loc,
        seq_lens,
        block_table,
        alibi,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        effective_splits,
        int(window_size),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        total_denom.stride(0),
        total_denom.stride(1),
        total_wv.stride(0),
        total_wv.stride(1),
        total_wv.stride(2),
        locks.stride(0),
        locks.stride(1),
        o.stride(0),
        o.stride(1),
        block_table.stride(0),
        ss_log2,
        softcap_log2,
        SCALE=scale,
        INV_SCALE=inv_scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        ACC_DTYPE=tl_acc_dtype,
        IO_DTYPE=io_dtype,
        ALLOW_TF32=allow_tf32,
        HAS_ALIBI=has_alibi,
        HAS_SOFTCAP=has_softcap,
        IS_CAUSAL=bool(is_causal),
        HAS_WINDOW=has_window,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_S=block_s,
        num_warps=num_warps,
        num_stages=num_stages,
    )
