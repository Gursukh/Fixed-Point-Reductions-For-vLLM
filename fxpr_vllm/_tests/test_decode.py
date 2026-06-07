import pytest
import torch

import fxpr_vllm  # noqa: F401
from .fixed_point_helpers import requires_cuda, skip_if_dtype_unsupported

_NUM_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 32
_PAGE_SIZE = 16
_SM_SCALE = 1.0 / (_HEAD_DIM**0.5)
_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _build_decode_inputs(seq_lens, num_blocks_per_seq, dtype):
    g = torch.Generator(device="cuda").manual_seed(2026)
    num_requests = len(seq_lens)
    total_blocks = num_requests * num_blocks_per_seq

    q = torch.randn(
        (num_requests, _NUM_HEADS, _HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
        generator=g,
    ).to(dtype)
    kv_cache = torch.randn(
        (total_blocks, 2, _PAGE_SIZE, _NUM_KV_HEADS, _HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
        generator=g,
    ).to(dtype)

    query_start_loc = torch.arange(
        num_requests + 1, device="cuda", dtype=torch.int32
    )
    seq_lens_t = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

    block_table = torch.arange(
        total_blocks, device="cuda", dtype=torch.int32
    ).reshape(num_requests, num_blocks_per_seq)

    return q, kv_cache, query_start_loc, seq_lens_t, block_table


def _run_decode(q, kv_cache, query_start_loc, seq_lens, block_table, num_kv_splits):
    o = torch.empty_like(q)
    torch.ops.fxpr.unified_attention_fxp(
        q,
        kv_cache,
        o,
        query_start_loc,
        seq_lens,
        block_table,
        1,
        None,
        True,
        float(_SM_SCALE),
        32,
        16,
        0.0,
        0,
        int(num_kv_splits),
    )
    return o


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("num_kv_splits", [1, 2, 4, 8])
def test_decode_invariant_across_kv_splits(dtype, num_kv_splits):
    skip_if_dtype_unsupported(dtype)
    # Seqs span several BLOCK_N tiles so the split has something to divide.
    seq_lens = [160, 96]
    q, kv_cache, qsl, sl, bt = _build_decode_inputs(
        seq_lens, num_blocks_per_seq=10, dtype=dtype
    )

    o_ref = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=1)
    o_got = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=num_kv_splits)

    assert torch.equal(o_ref, o_got), (
        f"Decode output diverged at num_kv_splits={num_kv_splits}, dtype={dtype}, "
        f"max diff = {(o_ref.float() - o_got.float()).abs().max().item()}"
    )


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
def test_decode_deterministic_across_runs(dtype):
    skip_if_dtype_unsupported(dtype)
    seq_lens = [128, 192, 64]
    q, kv_cache, qsl, sl, bt = _build_decode_inputs(
        seq_lens, num_blocks_per_seq=12, dtype=dtype
    )

    first = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=4)
    for _ in range(4):
        again = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=4)
        assert torch.equal(first, again), (
            f"Non-deterministic decode ({dtype}): "
            f"max diff = {(first.float() - again.float()).abs().max().item()}"
        )


@requires_cuda
@pytest.mark.parametrize("dtype", _DTYPES)
def test_decode_native_dtype(dtype):
    skip_if_dtype_unsupported(dtype)
    seq_lens = [48, 32]
    q, kv_cache, qsl, sl, bt = _build_decode_inputs(
        seq_lens, num_blocks_per_seq=3, dtype=dtype
    )

    o = _run_decode(q, kv_cache, qsl, sl, bt, num_kv_splits=4)
    assert o.dtype == dtype, f"output dtype {o.dtype} != input dtype {dtype}"
    assert o.shape == q.shape
    assert torch.isfinite(o.to(torch.float32)).all()
