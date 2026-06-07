import pytest
import torch

from .fixed_point_helpers import (
    prefill_fxp_test,
    requires_cuda,
    skip_if_dtype_unsupported,
)

_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_PREFILL_SEQ_LEN = 32
_PREFILL_HEADS = 2
_PREFILL_KV_HEADS = 2
_PREFILL_HEAD_DIM = 32
_PREFILL_SM_SCALE = 1.0 / (_PREFILL_HEAD_DIM**0.5)
# Multiple of every dtype's BLOCK_N (64 for fp16/bf16, 32 for fp32) so a
# permutation at this granularity moves whole K-tiles.
_PERM_BLOCK = 64


def _ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group = num_heads // num_kv_heads
    o = torch.zeros_like(q)

    for b in range(b_seq_len.shape[0]):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()
        for h in range(num_heads):
            kv_h = h // kv_group
            qi = q[start : start + seq_len, h, :]
            ki = k[start : start + seq_len, kv_h, :]
            vi = v[start : start + seq_len, kv_h, :]

            scores = (qi @ ki.T) * sm_scale
            if is_causal:
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores.masked_fill_(mask, -1e8)
            p = torch.softmax(scores, dim=-1)
            o[start : start + seq_len, h, :] = p @ vi
    return o


def _make_inputs(batch, seq_lens, num_heads, num_kv_heads, head_dim, seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(seq_lens)
    q = torch.randn(
        total, num_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )
    k = torch.randn(
        total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )
    v = torch.randn(
        total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )

    b_seq_len = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    b_start_loc = torch.zeros(batch, device="cuda", dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
    return q, k, v, b_start_loc, b_seq_len


def _run_prefill(
    q, k, v, b_start_loc, b_seq_len, max_input_len, dtype,
    *, is_causal=True, num_kv_splits=1,
):
    """Cast fp32 inputs to dtype, run the kernel, return the output as fp32."""
    o = torch.empty(q.shape, device="cuda", dtype=dtype)
    prefill_fxp_test(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        o,
        b_start_loc,
        b_seq_len,
        max_input_len=max_input_len,
        is_causal=is_causal,
        softmax_scale=_PREFILL_SM_SCALE,
        num_kv_splits=num_kv_splits,
    )
    return o.float()


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("batch", [1, 2])
def test_prefill_correctness_causal(dtype, batch):
    skip_if_dtype_unsupported(dtype)
    seq_lens = [_PREFILL_SEQ_LEN] * batch
    q, k, v, b_start_loc, b_seq_len = _make_inputs(
        batch, seq_lens, _PREFILL_HEADS, _PREFILL_KV_HEADS, _PREFILL_HEAD_DIM
    )

    o = _run_prefill(
        q, k, v, b_start_loc, b_seq_len, _PREFILL_SEQ_LEN, dtype, is_causal=True
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=True, sm_scale=_PREFILL_SM_SCALE
    )

    assert torch.allclose(
        o, ref, atol=5e-2, rtol=5e-2
    ), f"max error = {(o - ref).abs().max().item()} ({dtype})"


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
def test_prefill_correctness_non_causal(dtype):
    skip_if_dtype_unsupported(dtype)
    q, k, v, b_start_loc, b_seq_len = _make_inputs(
        1, [_PREFILL_SEQ_LEN], _PREFILL_HEADS, _PREFILL_KV_HEADS, _PREFILL_HEAD_DIM
    )

    o = _run_prefill(
        q, k, v, b_start_loc, b_seq_len, _PREFILL_SEQ_LEN, dtype, is_causal=False
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=False, sm_scale=_PREFILL_SM_SCALE
    )

    assert torch.allclose(
        o, ref, atol=5e-2, rtol=5e-2
    ), f"max error = {(o - ref).abs().max().item()} ({dtype})"


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
def test_prefill_variable_seq_lens(dtype):
    skip_if_dtype_unsupported(dtype)
    # Non-multiples of BLOCK_M/BLOCK_N exercise the ragged-tail masking.
    seq_lens = [70, 128, 100]
    q, k, v, b_start_loc, b_seq_len = _make_inputs(
        3, seq_lens, _PREFILL_HEADS, _PREFILL_KV_HEADS, _PREFILL_HEAD_DIM, seed=99
    )

    o = _run_prefill(
        q, k, v, b_start_loc, b_seq_len, max(seq_lens), dtype, is_causal=True
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=True, sm_scale=_PREFILL_SM_SCALE
    )

    assert torch.allclose(
        o, ref, atol=5e-2, rtol=5e-2
    ), f"max error = {(o - ref).abs().max().item()} ({dtype})"


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("num_kv_splits", [1, 2, 4, 8])
def test_prefill_invariant_across_kv_splits(dtype, num_kv_splits):
    skip_if_dtype_unsupported(dtype)
    # splits=1 takes the fused path, >1 takes the atomic-add split path. Outputs
    # must match.
    seq_lens = [96, 64]
    q, k, v, b_start_loc, b_seq_len = _make_inputs(
        2, seq_lens, _PREFILL_HEADS, _PREFILL_KV_HEADS, _PREFILL_HEAD_DIM, seed=5
    )

    o_ref = _run_prefill(
        q, k, v, b_start_loc, b_seq_len, max(seq_lens), dtype,
        is_causal=True, num_kv_splits=1,
    )
    o_got = _run_prefill(
        q, k, v, b_start_loc, b_seq_len, max(seq_lens), dtype,
        is_causal=True, num_kv_splits=num_kv_splits,
    )

    assert torch.equal(o_ref, o_got), (
        f"prefill diverged at num_kv_splits={num_kv_splits} ({dtype}), "
        f"max diff = {(o_ref - o_got).abs().max().item()}"
    )


def _float_attention_row(
    q_row: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    sm_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    seq_len = k_rows.shape[0]
    head_dim = q_row.shape[0]

    q_c = q_row.to(dtype)
    k_c = k_rows.to(dtype)

    scores = torch.zeros(seq_len, device="cuda", dtype=dtype)
    for j in range(seq_len):
        dot = torch.zeros((), device="cuda", dtype=dtype)
        for d in range(head_dim):
            dot = dot + q_c[d] * k_c[j, d]
        scores[j] = dot * sm_scale

    scores_max = scores.max()
    exp_scores = torch.zeros(seq_len, device="cuda", dtype=dtype)
    exp_sum = torch.zeros((), device="cuda", dtype=dtype)
    for j in range(seq_len):
        exp_scores[j] = torch.exp(scores[j] - scores_max)
        exp_sum = exp_sum + exp_scores[j]
    p = exp_scores / exp_sum

    v_c = v_rows.to(dtype)
    out = torch.zeros(head_dim, device="cuda", dtype=dtype)
    for j in range(seq_len):
        for d in range(head_dim):
            out[d] = out[d] + p[j] * v_c[j, d]
    return out


@requires_cuda
def test_float_accumulation_is_order_dependent():
    seq_len, head_dim = 3, 16
    q_row = torch.ones(head_dim, device="cuda", dtype=torch.float32)
    k_rows = torch.ones(seq_len, head_dim, device="cuda", dtype=torch.float32)

    v_rows = torch.zeros(seq_len, head_dim, device="cuda", dtype=torch.float32)
    v_rows[0] = 1e30
    v_rows[1] = -1e30
    v_rows[2] = 1e-30

    sm_scale = 1.0 / (head_dim**0.5)
    out_fwd = _float_attention_row(q_row, k_rows, v_rows, sm_scale, torch.float32)
    perm = torch.tensor([0, 2, 1], device="cuda")
    out_perm = _float_attention_row(
        q_row, k_rows[perm], v_rows[perm], sm_scale, torch.float32
    )

    assert not torch.equal(out_fwd[perm], out_perm), (
        "fp32 attention should differ with permuted KV order"
    )


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fixedpoint_prefill_is_tile_permutation_invariant(dtype):
    skip_if_dtype_unsupported(dtype)
    # Permuting whole BLOCK_N-aligned KV tiles is bit-exact since the cross-tile
    # reduction is an integer sum. Per-element permutation is not. Same contract
    # as gemm.
    seq_len = 3 * _PERM_BLOCK
    g = torch.Generator(device="cuda").manual_seed(7)
    shape = (seq_len, _PREFILL_HEADS, _PREFILL_HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g) * 0.5

    b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    b_start_loc = torch.tensor([0], device="cuda", dtype=torch.int32)

    o_orig = _run_prefill(
        q, k, v, b_start_loc, b_seq_len, seq_len, dtype, is_causal=False
    )

    n_tiles = seq_len // _PERM_BLOCK
    tile_perm = torch.randperm(n_tiles, device="cuda")
    idx = torch.cat([
        torch.arange(t * _PERM_BLOCK, (t + 1) * _PERM_BLOCK, device="cuda")
        for t in tile_perm
    ])
    o_perm = _run_prefill(
        q, k[idx], v[idx], b_start_loc, b_seq_len, seq_len, dtype, is_causal=False
    )

    assert torch.equal(o_orig, o_perm), (
        f"fxp prefill should be KV-tile-permutation invariant ({dtype}), "
        f"max diff = {(o_orig - o_perm).abs().max().item()}"
    )


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
def test_prefill_deterministic_across_runs(dtype):
    skip_if_dtype_unsupported(dtype)
    q, k, v, b_start_loc, b_seq_len = _make_inputs(
        1,
        [_PREFILL_SEQ_LEN],
        _PREFILL_HEADS,
        _PREFILL_KV_HEADS,
        _PREFILL_HEAD_DIM,
        seed=77,
    )

    results = [
        _run_prefill(
            q, k, v, b_start_loc, b_seq_len, _PREFILL_SEQ_LEN, dtype, is_causal=True
        )
        for _ in range(5)
    ]
    for r in results[1:]:
        assert torch.equal(
            results[0], r
        ), f"Non-deterministic ({dtype}): max diff = {(results[0] - r).abs().max().item()}"
