import torch

from fxpr_vllm._triton import gemm
from fxpr_vllm._triton.gemm import (
    _CFG_SPLITK,
    _M_BUCKETS,
    _PICKED_CONFIG,
    _gemm_kernel_autotuned,
    _resolve_split_k,
)
from fxpr_vllm.warmup import _gemm_warmed, warmup_gemm

from .fixed_point_helpers import requires_cuda, skip_if_dtype_unsupported


def test_resolve_split_k_uses_cache_then_persistent():
    """_resolve_split_k should use the warmup cache, respect a forced override,
    and fall back to persistent when int16, K-tiles, scratch, or the lock buffer
    don't allow a split. Un-probed shapes stay persistent. No GPU needed."""
    dtype = torch.bfloat16
    key = (3, 4096, 14336, dtype, 32)  # (sk_pid_m, N, K, dtype, int_bits)
    big = 10**12  # scratch/locks big enough to never be the limiter
    tiles = 192   # sk_num_tiles_mn, fits any non-binding lock capacity
    saved = dict(gemm._SPLITK_CHOICE)
    try:
        gemm._SPLITK_FORCE = None
        gemm._SPLITK_CHOICE.clear()

        # cached split-2 is used as-is
        gemm._SPLITK_CHOICE[key] = 2
        assert _resolve_split_k(3, 112, 32, 192, 4096, 14336, dtype, big, tiles, big) == 2

        # cached persistent (1) is used as-is
        gemm._SPLITK_CHOICE[key] = 1
        assert _resolve_split_k(3, 112, 32, 192, 4096, 14336, dtype, big, tiles, big) == 1

        # a forced value beats the cache (how warmup times each path)
        gemm._SPLITK_FORCE = 2
        assert _resolve_split_k(3, 112, 32, 192, 4096, 14336, dtype, big, tiles, big) == 2
        gemm._SPLITK_FORCE = None

        # int16 never splits, regardless of cache
        assert _resolve_split_k(3, 112, 16, 192, 4096, 14336, dtype, big, tiles, big) == 1

        # scratch too small for M*N falls back to persistent
        gemm._SPLITK_CHOICE[key] = 2
        assert _resolve_split_k(3, 112, 32, 192, 4096, 14336, dtype, 1, tiles, big) == 1

        # more (m,n) tiles than the lock buffer holds -> persistent. This is the
        # bug that corrupted Llama: a wide-N layer's tile count outran the locks.
        assert _resolve_split_k(3, 112, 32, 192, 4096, 14336, dtype, big, 5000, 256) == 1

        # fewer K-tiles than the split count -> persistent
        assert _resolve_split_k(3, 1, 32, 192, 4096, 14336, dtype, big, tiles, big) == 1

        # un-probed tile count (empty cache) stays persistent
        gemm._SPLITK_CHOICE.clear()
        assert _resolve_split_k(1, 112, 32, 64, 4096, 14336, dtype, big, tiles, big) == 1
    finally:
        gemm._SPLITK_FORCE = None
        gemm._SPLITK_CHOICE.clear()
        gemm._SPLITK_CHOICE.update(saved)


@requires_cuda
def test_warmup_covers_full_M_sweep():
    """After warmup, run every M in the bucket range with and without bias and
    confirm nothing new shows up in the autotune cache or _PICKED_CONFIG. A new
    entry would mean the runtime hit a JIT path warmup missed."""
    dtype = torch.float16
    skip_if_dtype_unsupported(dtype)

    # Small N widens the split-K range, so this shape hits more binaries.
    K, N = 4096, 64
    # w is the (K, N) transpose of an (out=N, in=K) Linear weight.
    w = torch.zeros(K, N, device="cuda", dtype=dtype)

    # Drop any prior warmup state so we test this run, not a leftover.
    _gemm_warmed.discard((K, N, dtype, 32, 16))

    warmup_gemm(w, 32, 16)

    autotune_keys_after = set(_gemm_kernel_autotuned.cache.keys())
    picked_keys_after = set(_PICKED_CONFIG.keys())

    bias = torch.zeros(N, device="cuda", dtype=dtype)
    # Walk every M, plus block boundaries and bucket edges to be safe.
    sk_block_m = _CFG_SPLITK[0]
    sweep = set()
    for M in range(1, _M_BUCKETS[-1] + 1):
        sweep.add(M)
        if M % sk_block_m == 0 or M % sk_block_m == 1:
            sweep.add(M)
    prev = 0
    for b in _M_BUCKETS:
        sweep.update({prev + 1, b - 1, b})
        prev = b
    sweep = sorted(s for s in sweep if 1 <= s <= _M_BUCKETS[-1])

    for M in sweep:
        a = torch.zeros(M, K, device="cuda", dtype=dtype)
        torch.ops.fxpr.gemm_fxp(a, w, None, 32, 16)
        torch.ops.fxpr.gemm_fxp(a, w, bias, 32, 16)

    new_autotune = set(_gemm_kernel_autotuned.cache.keys()) - autotune_keys_after
    new_picked = set(_PICKED_CONFIG.keys()) - picked_keys_after

    assert not new_autotune, f"warmup missed autotune keys: {new_autotune}"
    assert not new_picked, f"warmup missed _PICKED_CONFIG keys: {new_picked}"


@requires_cuda
def test_warmup_dedup():
    """A second warmup_gemm on the same shape should be a no-op."""
    dtype = torch.float16
    skip_if_dtype_unsupported(dtype)

    K, N = 2048, 128
    w = torch.zeros(K, N, device="cuda", dtype=dtype)
    _gemm_warmed.discard((K, N, dtype, 32, 16))

    warmup_gemm(w, 32, 16)
    autotune_after_first = len(_gemm_kernel_autotuned.cache)
    picked_after_first = len(_PICKED_CONFIG)

    warmup_gemm(w, 32, 16)
    assert len(_gemm_kernel_autotuned.cache) == autotune_after_first
    assert len(_PICKED_CONFIG) == picked_after_first
